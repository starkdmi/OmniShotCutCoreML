import CoreML
import Foundation
import os

/// Shots and transitions with OmniShotCut v1.5 on Core ML.
///
/// Every frame is read, at the file's own rate, as the official code reads
/// them. Frames stream through one 100-frame window at a time, so an hour of
/// video costs one window of memory, not an hour of frames.
///
/// Windowing and stitching are the official `engine.py`'s, step for step, as
/// in the MLX port: windows of 100 frames overlapping by 10, each window
/// trusted for its half of the overlap, a cut emitted by the window whose
/// valid region holds it, and every segment labelled by the cut that begins
/// it.
///
/// The model is a conversion written by `scripts/convert.py`: the fp16 one
/// runs on the Neural Engine, the fp32 one on the GPU. In exact mode the fp16
/// model runs every window and the fp32 model runs again the windows whose
/// decisions are near-ties, so the segments are the fp32 model's while most
/// windows run on the fp16 model alone.
public actor OmniShotCut {

    /// Exact mode's default near-tie margin, in logits.
    public static let defaultThreshold: Float = 1.0

    /// Frames shared by neighbouring windows. The official default is 10.
    public nonisolated let overlap: Int
    private nonisolated let model: any WindowModel
    /// Runs the model on one window while the next window's frames decode.
    private nonisolated let predictionQueue = DispatchQueue(
        label: "OmniShotCut.prediction", qos: .userInitiated)
    /// Runs one video's pipeline at a time. A pipeline blocks on decoding and
    /// on the model for the length of the video, so it runs on this queue's
    /// thread rather than on Swift's cooperative thread pool.
    private nonisolated let pipelineQueue = DispatchQueue(
        label: "OmniShotCut.pipeline", qos: .userInitiated)

    /// Loads a conversion and runs it on `computeUnits`.
    ///
    /// - Parameters:
    ///   - model: An `.mlpackage`, a compiled `.mlmodelc`, or a directory
    ///     holding one. A package is compiled once into
    ///     `Caches/OmniShotCutCoreML` and the compiled copy is reused: the
    ///     Neural Engine caches its own compilation, seconds on a model's
    ///     first load, per compiled model location.
    ///   - computeUnits: `.cpuAndNeuralEngine` for the fp16 conversion,
    ///     `.cpuAndGPU` for the fp32 one.
    ///   - fastPrediction: Sets `MLOptimizationHints.specializationStrategy`
    ///     to `.fastPrediction` on macOS 15 and iOS 18 and later: faster
    ///     predictions on the Neural Engine, a slower first load.
    ///   - overlap: Frames shared by neighbouring windows.
    public init(
        model: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        fastPrediction: Bool = true, overlap: Int = 10
    ) async throws {
        let predictor = try await WindowPredictor(
            model: model, computeUnits: computeUnits, fastPrediction: fastPrediction)
        self.init(windowModel: predictor, overlap: overlap)
    }

    /// Loads a conversion and a verifier for exact mode: `model`, normally the
    /// fp16 conversion, runs every window on `computeUnits`; `verifier`,
    /// normally the fp32 conversion, runs on the GPU the windows where a
    /// decision the segments depend on is a near-tie, and its logits replace
    /// the model's for those windows.
    ///
    /// A near-tie is a decision whose margin is below `threshold`. The
    /// decisions counted are those `decode_window_segments` reads: each
    /// query's end up to the query that reaches the window's last frame, and
    /// the intra and inter labels of the queries that make a segment. A
    /// label's margin is its best logit minus the second best, among the
    /// classes the argmax reads; an end's is its best logit minus the best of
    /// a class that ends the segment on another frame, since every end at or
    /// past the window's last frame clips to it. If the two models' logits
    /// differ by less than `threshold / 2` everywhere, every window's
    /// segments are the verifier's.
    ///
    /// - Parameters:
    ///   - model: The model run on every window, as in
    ///     ``init(model:computeUnits:fastPrediction:overlap:)``.
    ///   - computeUnits: The model's compute units.
    ///   - verifier: The model run on near-tie windows, on `.cpuAndGPU`. It
    ///     reads and writes the same shapes as `model`.
    ///   - threshold: The smallest decision margin accepted from `model`.
    ///   - fastPrediction: As for ``init(model:computeUnits:fastPrediction:overlap:)``,
    ///     for both models.
    ///   - overlap: Frames shared by neighbouring windows.
    public init(
        model: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, verifier: URL,
        threshold: Float = defaultThreshold, fastPrediction: Bool = true, overlap: Int = 10
    ) async throws {
        let predictor = try await WindowPredictor(
            model: model, computeUnits: computeUnits, verifier: verifier,
            threshold: threshold, fastPrediction: fastPrediction)
        self.init(windowModel: predictor, overlap: overlap)
    }

    /// A detector running `predictor` on each window.
    @_spi(Benchmarking) public init(predictor: WindowPredictor, overlap: Int = 10) {
        self.init(windowModel: predictor, overlap: overlap)
    }

    /// A detector running `windowModel` on each window.
    init(windowModel: some WindowModel, overlap: Int = 10) {
        self.model = windowModel
        self.overlap = max(0, overlap)
    }

    /// The Core ML model wrapper, for its statistics.
    @_spi(Benchmarking) public nonisolated var predictor: WindowPredictor? {
        model as? WindowPredictor
    }

    /// Segments covering the whole video, in order, without gaps.
    ///
    /// The video is decoded and predicted on the detector's own dispatch
    /// queue, one video at a time per detector, not on a thread of Swift's
    /// cooperative pool. Cancelling the calling task stops it at the next
    /// frame with `CancellationError`. A call waiting for another video on the
    /// same detector holds its video open while it waits; cancelled then, it
    /// throws when its turn comes, before decoding a frame. A call cancelled
    /// after the last frame was read returns its segments.
    public func segments(in video: URL) async throws -> [ShotSegment] {
        guard FileManager.default.fileExists(atPath: video.path) else {
            throw OmniShotCutError.videoUnavailable(video)
        }
        let configuration = model.configuration
        let reader = try await FrameReader(
            url: video, width: configuration.width, height: configuration.height)
        return try await segments(from: reader)
    }

    /// The whole pipeline over the model-sized RGB frames of one video, run
    /// on the pipeline queue. Cancelling the task stops it at the next frame.
    func segments(from source: some FrameSource) async throws -> [ShotSegment] {
        let cancellation = CancellationFlag()
        let handoff = Handoff(value: source)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pipelineQueue.async {
                    continuation.resume(
                        with: Result {
                            try self.pipeline(over: handoff.value, cancellation: cancellation)
                        })
                }
            }
        } onCancel: {
            cancellation.cancel()
        }
    }

    /// The pipeline itself, blocking until the last window's segments.
    private nonisolated func pipeline(
        over source: some FrameSource, cancellation: CancellationFlag
    ) throws -> [ShotSegment] {
        // Cancelled while it waited for another video's pipeline.
        if cancellation.isCancelled { throw CancellationError() }
        let configuration = model.configuration
        let window = configuration.windowLength
        let overlap = min(self.overlap, window - 1)
        let frameBytes = configuration.width * configuration.height * 3

        // A window and the frame past it.
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: (window + 1) * frameBytes, alignment: 16)
        defer { buffer.deallocate() }
        var available = 0
        var times: [TimeInterval] = []
        var windowStart = 0
        var cuts: [Cut] = []
        // The model runs on one window while the next one's frames decode.
        var running:
            (prediction: PendingPrediction, start: Int, validLength: Int, bounds: (Int, Int))?

        func finishRunning() throws {
            guard let pending = running else { return }
            running = nil
            let logits = try pending.prediction.wait()
            guard logits.intra.count == logits.end.count, logits.inter.count == logits.end.count
            else {
                throw OmniShotCutError.predictionFailed(
                    "\(logits.intra.count), \(logits.inter.count) and \(logits.end.count) "
                        + "predictions for the intra, inter and end heads")
            }
            let segments = Self.windowSegments(logits: logits, validLength: pending.validLength)
            cuts += Self.cuts(
                from: segments, windowStart: pending.start, validStart: pending.bounds.0,
                validEnd: pending.bounds.1)
        }

        // A full window is started only once a frame past it has been read:
        // the last window is the one that reaches the end, and the official
        // code treats it differently.
        func start(final: Bool) throws {
            let validLength = min(window, available)
            let bounds = Self.validRange(
                windowStart: windowStart, window: window, overlap: overlap,
                total: final ? windowStart + available : Int.max)
            // The window's frames are copied: the buffer takes the next
            // window's frames while the model reads these.
            let bytes = validLength * frameBytes
            let frames = [UInt8](unsafeUninitializedCapacity: bytes) { copy, count in
                UnsafeMutableRawPointer(copy.baseAddress!).copyMemory(
                    from: buffer, byteCount: bytes)
                count = bytes
            }
            let prediction = PendingPrediction(on: predictionQueue) { [model] in
                try frames.withUnsafeBytes { try model.predict(frames: $0, count: validLength) }
            }
            try finishRunning()
            running = (prediction, windowStart, validLength, (bounds.start, bounds.end))
        }

        while let time = try source.nextFrame(into: buffer + available * frameBytes) {
            if cancellation.isCancelled { throw CancellationError() }
            times.append(time)
            available += 1
            if available > window {
                try start(final: false)
                let stride = window - overlap
                // The frames past the stride open the next window.
                memmove(buffer, buffer + stride * frameBytes, (available - stride) * frameBytes)
                available -= stride
                windowStart += stride
            }
        }
        guard !times.isEmpty else { throw OmniShotCutError.noFrames(source.url) }
        try start(final: true)
        try finishRunning()

        return Self.assemble(cuts: cuts, frameCount: times.count).map { range in
            ShotSegment(
                start: times[range.start],
                end: range.end < times.count ? times[range.end] : source.duration,
                frames: range.start..<range.end,
                kind: ShotSegment.Kind(rawValue: range.intra) ?? .shot,
                boundary: ShotSegment.Boundary(rawValue: range.inter) ?? .hardCut)
        }
    }
}

/// The source of a video's model-sized RGB frames, in presentation order.
protocol FrameSource: AnyObject {
    var url: URL { get }
    /// The end of the video track: the end of the last segment.
    var duration: TimeInterval { get }
    /// Copies the next frame's `height × width × 3` RGB bytes to
    /// `destination` and returns its presentation time, `nil` at the end.
    func nextFrame(into destination: UnsafeMutableRawPointer) throws -> TimeInterval?
}

/// The window shape a model reads.
@_spi(Benchmarking) public struct WindowConfiguration: Sendable, Hashable {
    /// Frames per window; 100 for the official model.
    public internal(set) var windowLength: Int
    /// Frame size; 128×96 for the official model.
    public internal(set) var width: Int
    public internal(set) var height: Int
}

/// A model that predicts one window's segments.
///
/// `predict` receives `count` frames, `1...windowLength` of them, each
/// `height × width × 3` RGB bytes row by row, one frame after another. When
/// `count` is short of `windowLength` — the last window of a video, or a video
/// shorter than one window — the model reads the missing frames as black: RGB
/// 0 before normalization, as the official `split_videos` pads with black
/// `uint8` frames.
///
/// It returns each query's argmax over every class but the last, as the
/// official code takes it after its softmax, for the intra-relation, the
/// inter-relation and the end heads, one entry per query in each.
///
/// Calls come from one serial queue, one window at a time, while the pipeline
/// decodes the next window on another thread.
protocol WindowModel: Sendable {
    var configuration: WindowConfiguration { get }
    func predict(frames: UnsafeRawBufferPointer, count: Int) throws -> OmniShotCut.WindowLogits
}

/// Set when the task awaiting a pipeline is cancelled; the pipeline reads it
/// between frames.
private final class CancellationFlag: Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)

    var isCancelled: Bool { state.withLock { $0 } }

    func cancel() { state.withLock { $0 = true } }
}

/// A frame source handed to the pipeline queue, the only code that uses it
/// afterwards.
private struct Handoff<Value>: @unchecked Sendable {
    let value: Value
}

/// One window's prediction, running on the model's queue until waited for.
private final class PendingPrediction: @unchecked Sendable {
    private let done = DispatchSemaphore(value: 0)
    /// Written once on the model's queue before `done` is signalled, read
    /// after it is waited for.
    private var result: Result<OmniShotCut.WindowLogits, any Error>?

    init(
        on queue: DispatchQueue,
        _ work: @escaping @Sendable () throws -> OmniShotCut.WindowLogits
    ) {
        queue.async {
            self.result = Result(catching: work)
            self.done.signal()
        }
    }

    /// Blocks until the prediction has run; called once.
    func wait() throws -> OmniShotCut.WindowLogits {
        done.wait()
        return try result!.get()
    }
}
