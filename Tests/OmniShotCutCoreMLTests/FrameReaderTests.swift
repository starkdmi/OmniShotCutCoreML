import AVFoundation
import Foundation
import XCTest

@testable import OmniShotCutCoreML

/// `FrameReader` against the decode the official code runs,
/// `ffmpeg -s 128x96 -pix_fmt rgb24 -fps_mode passthrough`, frame by frame and
/// byte by byte, on the videos `scripts/fetch-videos.sh` downloads, on rotated
/// and mirrored copies of one of them, and on clips ffmpeg encodes: 8-bit
/// HEVC, 4K H.264, tracks whose range or matrix changes between sample
/// descriptions, and the formats `FrameReader` rejects. Skips without ffmpeg.
final class FrameReaderTests: XCTestCase {

    func testFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        var compared = 0
        for name in ["sintel_trailer-480p.mp4", "trailer_480p.mov", "tears_of_steel_720p.mov"] {
            let video = TestVideos.directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: video.path) else { continue }
            try await compare(video, ffmpeg: ffmpeg)
            compared += 1
        }
        try XCTSkipIf(compared == 0, "no videos; run scripts/fetch-videos.sh")
    }

    /// The last segment ends with the video track, which ends before the
    /// audio track in the Big Buck Bunny trailer.
    func testTheDurationIsTheVideoTracks() async throws {
        let video = TestVideos.directory.appendingPathComponent("trailer_480p.mov")
        guard FileManager.default.fileExists(atPath: video.path) else {
            throw XCTSkip("no videos; run scripts/fetch-videos.sh")
        }
        let reader = try await FrameReader(url: video, width: 128, height: 96)
        XCTAssertEqual(reader.duration, 812.0 / 25, accuracy: 1e-9)
        let asset = try await AVURLAsset(url: video).load(.duration).seconds
        XCTAssertGreaterThan(asset, reader.duration + 0.4)
    }

    /// ffmpeg turns a rotated picture before scaling it, so the horizontal
    /// filter runs along the stored picture's columns. Mirrored display
    /// matrices are flipped as ffmpeg's autorotation flips them.
    func testRotatedFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let source = TestVideos.directory.appendingPathComponent(
            "sintel_trailer-480p.mp4")
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw XCTSkip("no videos; run scripts/fetch-videos.sh")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "omnishotcut-rotation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let matrices: [(String, [String])] = [
            ("rotated-90", ["-display_rotation", "90"]),
            ("rotated-180", ["-display_rotation", "180"]),
            ("rotated-270", ["-display_rotation", "270"]),
            ("hflip", ["-display_hflip"]),
            ("vflip", ["-display_vflip"]),
            ("rotated-90-hflip", ["-display_rotation", "90", "-display_hflip"]),
            ("rotated-180-hflip", ["-display_rotation", "180", "-display_hflip"]),
            ("rotated-270-hflip", ["-display_rotation", "270", "-display_hflip"]),
        ]
        for (name, options) in matrices {
            let rotated = directory.appendingPathComponent("\(name).mp4")
            try Self.run(
                ffmpeg,
                ["-v", "error"] + options + [
                    "-i", source.path, "-t", "12", "-c", "copy", rotated.path,
                ])
            try await compare(rotated, ffmpeg: ffmpeg)
        }
    }

    /// HEVC decoding is bit-exact as well; 8-bit 4:2:0 HEVC is read like
    /// H.264.
    func testHEVCFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clip = try Self.encode(
            directory, "hevc.mp4", ffmpeg: ffmpeg,
            ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"])
        try await compare(clip, ffmpeg: ffmpeg)
    }

    /// 4K frames are converted a few at a time: each batch holds at most
    /// 64 MB of planes.
    func testLargeFramesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clip = try Self.encode(
            directory, "uhd.mp4", ffmpeg: ffmpeg, size: "3840x2160",
            ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
        try await compare(clip, ffmpeg: ffmpeg)
    }

    /// AVFoundation's errors arrive as `OmniShotCutError`s.
    func testUnreadableFilesAreOmniShotCutErrors() async throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let garbage = directory.appendingPathComponent("garbage.mp4")
        try Data(repeating: 0x5a, count: 4_096).write(to: garbage)
        do {
            _ = try await FrameReader(url: garbage, width: 128, height: 96)
            XCTFail("read")
        } catch OmniShotCutError.decodingFailed(let url, _) {
            XCTAssertEqual(url, garbage)
        }
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let audio = directory.appendingPathComponent("audio.m4a")
        try Self.run(
            ffmpeg,
            ["-v", "error", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", audio.path])
        do {
            _ = try await FrameReader(url: audio, width: 128, height: 96)
            XCTFail("read")
        } catch OmniShotCutError.noVideoTrack(let url) {
            XCTAssertEqual(url, audio)
        }
    }

    /// A track that joins clips of different ranges or matrices has a sample
    /// description for each; every frame is converted in its own range and
    /// matrix, as ffmpeg converts it.
    func testColourChangesAreFFmpegs() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func clip(
            _ name: String, matrix: String, range: String,
            codec: [String] = ["-c:v", "libx264"]
        ) throws -> URL {
            try Self.encode(
                directory, name, ffmpeg: ffmpeg,
                codec + [
                    "-bf", "0", "-pix_fmt", "yuv420p", "-colorspace", matrix,
                    "-color_primaries", matrix, "-color_trc", matrix, "-color_range", range,
                ])
        }
        let limited = try clip("limited.mp4", matrix: "smpte170m", range: "tv")
        let full = try clip("full.mp4", matrix: "smpte170m", range: "pc")
        let bt709 = try clip("bt709.mp4", matrix: "bt709", range: "tv")
        let hevc = ["-c:v", "libx265", "-tag:v", "hvc1"]
        let limitedHEVC = try clip(
            "limited-hevc.mp4", matrix: "smpte170m", range: "tv", codec: hevc)
        let fullHEVC = try clip("full-hevc.mp4", matrix: "smpte170m", range: "pc", codec: hevc)
        for (name, parts) in [
            ("limited-full", [limited, full]), ("full-limited", [full, limited]),
            ("bt709-bt601", [bt709, limited]),
            ("hevc-limited-full", [limitedHEVC, fullHEVC]),
            ("hevc-full-limited", [fullHEVC, limitedHEVC]),
        ] {
            let joined = directory.appendingPathComponent("\(name).mov")
            try await Self.join(parts, into: joined)
            let tracks = try await AVURLAsset(url: joined).loadTracks(withMediaType: .video)
            let track = try XCTUnwrap(tracks.first)
            let descriptions = try await track.load(.formatDescriptions)
            XCTAssertEqual(descriptions.count, 2, "\(name): sample descriptions")
            try await compare(joined, ffmpeg: ffmpeg)
        }
    }

    /// AVFoundation converts these to 8-bit 4:2:0 where ffmpeg scales them
    /// from their own depth and subsampling: every frame is read, close to
    /// ffmpeg's but not equal to it.
    func testOtherFormatsAreDecoded() async throws {
        guard let ffmpeg = Self.ffmpeg else { throw XCTSkip("ffmpeg not found on PATH") }
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clips: [(String, [String])] = [
            (
                "hevc-10bit.mp4",
                ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"]
            ),
            ("h264-422.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv422p"]),
            ("h264-10bit.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p10le"]),
            ("prores.mov", ["-c:v", "prores_ks", "-profile:v", "2"]),
        ]
        for (name, options) in clips {
            let clip = try Self.encode(directory, name, ffmpeg: ffmpeg, options)
            try await compare(clip, ffmpeg: ffmpeg, exact: false)
        }
    }

    /// Every frame against ffmpeg's; with `exact`, byte for byte.
    private func compare(_ video: URL, ffmpeg: URL, exact: Bool = true) async throws {
        let name = video.lastPathComponent
        let theirs = try Self.run(
            ffmpeg,
            [
                "-v", "error", "-i", video.path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "128x96", "-fps_mode", "passthrough", "-",
            ], skipOnFailure: false)
        let frameBytes = 128 * 96 * 3
        let reader = try await FrameReader(url: video, width: 128, height: 96)
        var frames = 0
        var different = 0
        var largest = 0
        var offset = 0
        // Time spent in `next()`, comparisons excluded; indicative only.
        let clock = ContinuousClock()
        var reading = Duration.zero
        while true {
            let started = clock.now
            guard let frame = try reader.next() else { break }
            reading += clock.now - started
            guard offset + frameBytes <= theirs.count else { break }
            let expected = theirs[offset..<(offset + frameBytes)]
            if !frame.pixels.elementsEqual(expected) {
                for (a, b) in zip(frame.pixels, expected) where a != b {
                    different += 1
                    largest = max(largest, abs(Int(a) - Int(b)))
                }
            }
            offset += frameBytes
            frames += 1
        }
        let seconds =
            Double(reading.components.seconds)
            + Double(reading.components.attoseconds) * 1e-18
        print(
            "FrameReader \(name): \(frames) frames, \(different) values differ, largest \(largest)"
                + String(format: ", %.0f frames/s read", Double(frames) / max(seconds, 1e-9)))
        XCTAssertEqual(frames * frameBytes, theirs.count, "\(name): frame count")
        if exact { XCTAssertEqual(different, 0, name) }
    }

    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "omnishotcut-clips-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// One second of ffmpeg's `testsrc2` pattern, `size` at 25 frames per
    /// second, encoded with `options`.
    private static func encode(
        _ directory: URL, _ name: String, ffmpeg: URL, size: String = "320x240",
        _ options: [String]
    ) throws -> URL {
        let clip = directory.appendingPathComponent(name)
        let quiet = options.contains("libx265") ? ["-x265-params", "log-level=error"] : []
        try run(
            ffmpeg,
            ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=\(size):rate=25", "-t", "1"]
                + options + quiet + [clip.path])
        return clip
    }

    /// `parts`' video tracks one after another in one track, the samples
    /// copied without re-encoding; each part keeps its sample description.
    private static func join(_ parts: [URL], into destination: URL) async throws {
        var samples: [CMSampleBuffer] = []
        var offset = CMTime.zero
        for part in parts {
            let asset = AVURLAsset(url: part)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            let track = try XCTUnwrap(tracks.first)
            let range = try await track.load(.timeRange)
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
            reader.add(output)
            XCTAssertTrue(reader.startReading())
            while let sample = output.copyNextSampleBuffer() {
                guard CMSampleBufferGetNumSamples(sample) > 0 else { continue }
                let timing = try sample.sampleTimingInfos().map { timing in
                    var timing = timing
                    timing.presentationTimeStamp =
                        timing.presentationTimeStamp - range.start + offset
                    if timing.decodeTimeStamp.isValid {
                        timing.decodeTimeStamp = timing.decodeTimeStamp - range.start + offset
                    }
                    return timing
                }
                samples.append(try CMSampleBuffer(copying: sample, withNewTiming: timing))
            }
            offset = offset + range.duration
        }
        let writer = try AVAssetWriter(outputURL: destination, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video, outputSettings: nil,
            sourceFormatHint: samples.first?.formatDescription)
        input.expectsMediaDataInRealTime = false
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for sample in samples {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(1)) }
            XCTAssertTrue(input.append(sample), "\(writer.error.map { "\($0)" } ?? "")")
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status, .completed, "\(writer.error.map { "\($0)" } ?? "")")
    }

    /// ffmpeg's exit status other than 0, for a reference decode.
    struct FFmpegFailed: Error, CustomStringConvertible {
        var arguments: [String]
        var status: Int32
        var description: String {
            "ffmpeg \(arguments.joined(separator: " ")) exited with \(status)"
        }
    }

    /// Runs ffmpeg and returns its output. A failure skips the test when
    /// `skipOnFailure` (writing a clip: an ffmpeg build can lack an encoder)
    /// and fails it otherwise (decoding the reference frames).
    @discardableResult
    private static func run(
        _ executable: URL, _ arguments: [String], skipOnFailure: Bool = true
    ) throws -> Data {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            if skipOnFailure {
                throw XCTSkip("ffmpeg \(arguments.joined(separator: " ")) failed")
            }
            throw FFmpegFailed(arguments: arguments, status: process.terminationStatus)
        }
        return data
    }

    private static var ffmpeg: URL? {
        let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
        let directories =
            path.split(separator: ":").map(String.init) + [
                "/opt/homebrew/bin", "/usr/local/bin",
            ]
        return directories.map { URL(fileURLWithPath: $0).appendingPathComponent("ffmpeg") }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}
