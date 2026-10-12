import AVFoundation
import Foundation
import XCTest

@_spi(Benchmarking) @testable import OmniShotCutCoreML

/// The pipeline around the model, with a stub in its place: windows of 100
/// frames started every 90, the last one short and read as padded, each
/// window's cuts stitched once, and segment times taken from the frames.
final class PipelineTests: XCTestCase {

    /// Synthetic frames carrying their own index in their first four bytes,
    /// at 25 frames per second. Read by one pipeline at a time, and by the
    /// test only after it.
    final class CountingSource: FrameSource, @unchecked Sendable {
        let url = URL(fileURLWithPath: "/synthetic.mp4")
        let count: Int
        let frameBytes = 128 * 96 * 3
        private var index = 0

        init(count: Int) { self.count = count }

        var duration: TimeInterval { Double(count) / 25 }

        /// Frames handed out so far; read once the pipeline is done.
        var framesRead: Int { index }

        func nextFrame(into destination: UnsafeMutableRawPointer) throws -> TimeInterval? {
            guard index < count else { return nil }
            destination.initializeMemory(as: UInt8.self, repeating: 0x40, count: frameBytes)
            destination.storeBytes(of: UInt32(index).littleEndian, as: UInt32.self)
            index += 1
            return Double(index - 1) / 25
        }
    }

    /// Ends a segment at every global multiple of `period` inside the window,
    /// labelled by that position, and runs the last one past the window;
    /// without a period, one segment covers the window. Each call is recorded
    /// with the frame indices it was given.
    final class StubModel: WindowModel, @unchecked Sendable {
        struct Call {
            var start: Int
            var count: Int
            var bytes: Int
            /// The frames are consecutive, starting at `start`.
            var consecutive: Bool
        }

        let configuration = WindowConfiguration(windowLength: 100, width: 128, height: 96)
        let period: Int?
        let failing: Int?
        private let lock = NSLock()
        private var recorded: [Call] = []

        init(period: Int?, failing: Int? = nil) {
            self.period = period
            self.failing = failing
        }

        var calls: [Call] { lock.withLock { recorded } }

        static func intra(_ position: Int, period: Int) -> Int { (position / period) % 8 }
        static func inter(_ position: Int, period: Int) -> Int { 1 + (position / period) % 4 }

        func predict(frames: UnsafeRawBufferPointer, count: Int) throws
            -> OmniShotCut.WindowLogits
        {
            let frameBytes = configuration.width * configuration.height * 3
            func index(_ frame: Int) -> Int {
                Int(
                    UInt32(
                        littleEndian: frames.loadUnaligned(
                            fromByteOffset: frame * frameBytes, as: UInt32.self)))
            }
            let start = index(0)
            let consecutive = (0..<count).allSatisfy { index($0) == start + $0 }
            let number = lock.withLock {
                recorded.append(
                    Call(
                        start: start, count: count, bytes: frames.count,
                        consecutive: consecutive))
                return recorded.count - 1
            }
            if number == failing { throw OmniShotCutError.predictionFailed("stub") }

            let window = configuration.windowLength
            var ends: [Int] = []
            if let period {
                ends = Swift.stride(
                    from: (start / period + 1) * period, to: start + count, by: period
                ).map { $0 - start }
            }
            // "Ends after the window", clipped to the frames there are.
            ends.append(window + 1)
            var intra = [0]
            var inter = [0]
            for end in ends.dropLast() {
                intra.append(Self.intra(start + end, period: period!))
                inter.append(Self.inter(start + end, period: period!))
            }
            // The remaining queries are padding, never read.
            while ends.count < 24 {
                ends.append(window + 1)
                intra.append(ShotSegment.Kind.padding.rawValue)
                inter.append(ShotSegment.Boundary.padding.rawValue)
            }
            return OmniShotCut.WindowLogits(intra: intra, inter: inter, end: ends)
        }
    }

    /// The windows `engine.py` forms over `total` frames: full windows every
    /// `window - overlap` frames, each started once a frame past it is read,
    /// then the one that reaches the end.
    private func expectedWindows(total: Int, window: Int = 100, overlap: Int = 10)
        -> [(start: Int, count: Int)]
    {
        let stride = window - overlap
        let full = total > window ? (total - window - 1) / stride + 1 : 0
        return (0..<full).map { ($0 * stride, window) } + [
            (full * stride, total - full * stride)
        ]
    }

    private func check(total: Int, period: Int, overlap: Int = 10) async throws {
        let model = StubModel(period: period)
        let detector = OmniShotCut(windowModel: model, overlap: overlap)
        let segments = try await detector.segments(from: CountingSource(count: total))

        let calls = model.calls
        let expected = expectedWindows(total: total, overlap: overlap)
        XCTAssertEqual(calls.map(\.start), expected.map { $0.start }, "\(total) frames: starts")
        XCTAssertEqual(calls.map(\.count), expected.map { $0.count }, "\(total) frames: counts")
        XCTAssertEqual(calls.map(\.bytes), expected.map { $0.count * 128 * 96 * 3 })
        XCTAssertTrue(calls.allSatisfy(\.consecutive), "\(total) frames: frame order")

        // Every multiple of the period is found once, by whichever window
        // trusts it.
        let cuts = Array(Swift.stride(from: period, to: total, by: period))
        XCTAssertEqual(segments.map(\.frames.lowerBound), [0] + cuts, "\(total) frames")
        XCTAssertEqual(segments.map(\.frames.upperBound), cuts + [total], "\(total) frames")
        XCTAssertEqual(segments.first?.kind, .shot)
        XCTAssertEqual(segments.first?.boundary, .videoStart)
        for segment in segments.dropFirst() {
            let position = segment.frames.lowerBound
            XCTAssertEqual(segment.kind.rawValue, StubModel.intra(position, period: period))
            XCTAssertEqual(segment.boundary.rawValue, StubModel.inter(position, period: period))
        }
        for segment in segments {
            XCTAssertEqual(segment.start, Double(segment.frames.lowerBound) / 25)
            XCTAssertEqual(segment.end, Double(segment.frames.upperBound) / 25)
        }
    }

    func testWindowsStartEveryNinetyFramesAndTheLastIsPadded() async throws {
        // The Sintel trailer's length: 13 full windows, then 83 frames read
        // with 17 of padding.
        try await check(total: 1_253, period: 37)
        XCTAssertEqual(expectedWindows(total: 1_253).count, 14)
        XCTAssertEqual(expectedWindows(total: 1_253).last?.start, 1_170)
        XCTAssertEqual(expectedWindows(total: 1_253).last?.count, 83)
    }

    func testDenseCutsAcrossEveryOverlap() async throws {
        try await check(total: 812, period: 5)
    }

    func testWindowBoundaries() async throws {
        // One short window; exactly one window; a frame past it, which
        // starts a second window of 11 frames.
        for total in [1, 60, 99, 100, 101, 190, 191] {
            try await check(total: total, period: 7)
        }
    }

    func testOtherOverlaps() async throws {
        try await check(total: 500, period: 11, overlap: 0)
        try await check(total: 500, period: 11, overlap: 20)
    }

    func testModelErrorsPropagate() async throws {
        let model = StubModel(period: 37, failing: 2)
        let detector = OmniShotCut(windowModel: model)
        do {
            _ = try await detector.segments(from: CountingSource(count: 1_253))
            XCTFail("expected the stub's error")
        } catch OmniShotCutError.predictionFailed(let detail) {
            XCTAssertEqual(detail, "stub")
        }
    }

    func testAVideoWithoutFramesIsAnError() async throws {
        let detector = OmniShotCut(windowModel: StubModel(period: 37))
        do {
            _ = try await detector.segments(from: CountingSource(count: 0))
            XCTFail("expected an error")
        } catch OmniShotCutError.noFrames {}
    }

    /// The pipeline runs on its own queue; cancelling the awaiting task
    /// stops it at the next frame.
    func testCancellingTheTaskStopsThePipeline() async throws {
        let detector = OmniShotCut(windowModel: StubModel(period: 37))
        let task = Task { try await detector.segments(from: CountingSource(count: .max)) }
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {}
    }

    /// A call waiting for another video on the same detector, cancelled
    /// while it waits, throws when its turn comes without reading a frame.
    func testCancellingAWaitingCallReadsNothing() async throws {
        let detector = OmniShotCut(windowModel: StubModel(period: 37))
        let running = Task { try await detector.segments(from: CountingSource(count: .max)) }
        try await Task.sleep(for: .milliseconds(200))
        let waitingSource = CountingSource(count: 1_253)
        let waiting = Task { try await detector.segments(from: waitingSource) }
        try await Task.sleep(for: .milliseconds(100))
        waiting.cancel()
        running.cancel()
        for task in [running, waiting] {
            do {
                _ = try await task.value
                XCTFail("expected cancellation")
            } catch is CancellationError {}
        }
        XCTAssertEqual(waitingSource.framesRead, 0)
    }

    /// The decoder and the pipeline together, on the Sintel trailer.
    func testSegmentsOfAVideo() async throws {
        let video = TestVideos.directory.appendingPathComponent("sintel_trailer-480p.mp4")
        guard FileManager.default.fileExists(atPath: video.path) else {
            throw XCTSkip("no videos; set OMNISHOTCUT_VIDEOS")
        }
        let model = StubModel(period: nil)
        let segments = try await OmniShotCut(windowModel: model).segments(in: video)
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments.first?.frames, 0..<1_253)
        XCTAssertEqual(segments.first?.start, 0)
        // The last segment ends with the video track, not with the audio.
        let tracks = try await AVURLAsset(url: video).loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let end = CMTimeRangeGetEnd(try await track.load(.timeRange)).seconds
        XCTAssertEqual(segments.first?.end, end)
        XCTAssertEqual(model.calls.map(\.count), expectedWindows(total: 1_253).map { $0.count })
    }
}
