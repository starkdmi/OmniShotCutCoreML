import CoreML
import CryptoKit
import Foundation
import XCTest

@_spi(Benchmarking) @testable import OmniShotCutCoreML

/// The whole detector against the official PyTorch implementation.
///
/// `omnishotcut-reference.json` is the official code's output for three
/// Blender open movies — `uva-cv-lab/OmniShotCut_v1.5`, fp32 on the CPU,
/// overlap 10 — written by OmniShotCutMLX's `scripts/reference.py`. The port
/// has to find the same cuts with the same labels from its own decoder and
/// converted model.
///
/// The environment names the model; without `OMNISHOTCUT_MODEL` the test
/// skips.
///
/// - `OMNISHOTCUT_MODEL`: an `.mlpackage`, `.mlmodelc` or a directory holding
///   one.
/// - `OMNISHOTCUT_COMPUTE_UNITS`: `ane` (`.cpuAndNeuralEngine`, the default),
///   `gpu`, `cpu` or `all`.
/// - `OMNISHOTCUT_VERIFIER`: a verifier model, normally the fp32 conversion;
///   runs exact mode, the verifier on `.cpuAndGPU`.
/// - `OMNISHOTCUT_THRESHOLD`: exact mode's near-tie margin,
///   ``OmniShotCut/defaultThreshold`` if unset.
/// - `OMNISHOTCUT_VIDEOS`: the videos' directory, `videos/` if unset.
final class ReferenceTests: XCTestCase {

    private struct Reference: Decodable {
        struct Video: Decodable {
            var sha256: String
            var fps: Double
            var frames: Int
            var ranges: [[Int]]
            var intra: [Int]
            var inter: [Int]
        }
        var videos: [String: Video]
    }

    func testSegmentsMatchTheOfficialImplementation() async throws {
        let environment = ProcessInfo.processInfo.environment
        func path(_ key: String) -> URL? {
            environment[key].flatMap {
                $0.isEmpty ? nil : URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
            }
        }
        guard let model = path("OMNISHOTCUT_MODEL") else {
            throw XCTSkip("set OMNISHOTCUT_MODEL to a converted model")
        }
        let computeUnits: MLComputeUnits
        switch environment["OMNISHOTCUT_COMPUTE_UNITS"] ?? "ane" {
        case "ane", "": computeUnits = .cpuAndNeuralEngine
        case "gpu": computeUnits = .cpuAndGPU
        case "cpu": computeUnits = .cpuOnly
        case "all": computeUnits = .all
        case let other:
            XCTFail("OMNISHOTCUT_COMPUTE_UNITS=\(other): expected ane, gpu, cpu or all")
            return
        }
        let threshold =
            environment["OMNISHOTCUT_THRESHOLD"].flatMap(Float.init)
            ?? OmniShotCut.defaultThreshold
        let verifier = path("OMNISHOTCUT_VERIFIER")

        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "omnishotcut-reference", withExtension: "json",
                subdirectory: "Fixtures"))
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))

        let started = Date()
        let detector =
            if let verifier {
                try await OmniShotCut(
                    model: model, computeUnits: computeUnits, verifier: verifier,
                    threshold: threshold)
            } else {
                try await OmniShotCut(model: model, computeUnits: computeUnits)
            }
        let predictor = try XCTUnwrap(detector.predictor)
        print(
            "OmniShotCut model \(predictor.modelURL.lastPathComponent)"
                + (verifier.map { ", verifier \($0.lastPathComponent), threshold \(threshold)" }
                    ?? "")
                + String(format: ": loaded in %.1f s", Date().timeIntervalSince(started)))
        // The fp32 conversion and exact mode reproduce every cut and add
        // none. fp16 arithmetic moves logits by tenths, which decides
        // near-ties: decisions whose two best logits are closer than that,
        // such as a label in the Sintel trailer that fp32 prefers by 0.0016
        // and the MLX port's fp16 weights flip too. Which ones flip depends
        // on the chip — 2 differences on an M4's Neural Engine, 5 on an M1
        // Pro's, all at decisions PyTorch in fp64 takes by under 0.05 — so
        // the fp16 model is not held to a count but to where it differs:
        // only in windows whose own decision margin is below the threshold,
        // those exact mode runs again. A difference anywhere else would get
        // past exact mode too. The model's metadata `precision`, or without
        // it Float32 outputs, marks the fp32 conversion.
        let exact = verifier != nil || predictor.computesInFloat32
        let windowLength = predictor.configuration.windowLength
        let overlap = min(detector.overlap, windowLength - 1)
        /// The window whose cuts are trusted at `position`, numbered in the
        /// order predicted: `split_videos`' `(start, end]`.
        func window(of position: Int, frames: Int) -> Int {
            var index = 0
            while true {
                let valid = OmniShotCut.validRange(
                    windowStart: index * (windowLength - overlap), window: windowLength,
                    overlap: overlap, total: frames)
                if position <= valid.end || valid.end >= frames { return index }
                index += 1
            }
        }

        let videos = TestVideos.directory
        var compared = 0
        var totalCuts = 0
        var totalIdentical = 0
        var totalExtra = 0
        for name in reference.videos.keys.sorted() {
            let expected = reference.videos[name]!
            let video = videos.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: video.path) else {
                print("OmniShotCut: \(name) missing, skipped")
                continue
            }
            guard try Self.sha256(of: video) == expected.sha256 else {
                XCTFail("\(name) is not the file the reference was made from")
                continue
            }
            predictor.resetStatistics()
            let started = Date()
            let segments = try await detector.segments(in: video)
            let elapsed = Date().timeIntervalSince(started)

            XCTAssertEqual(segments.last?.frames.upperBound, expected.frames, name)
            let theirs = Array(
                zip(expected.ranges, zip(expected.intra, expected.inter)).dropFirst())
            let predicted = Array(segments.dropFirst())
            var matched = 0
            var sameFrame = 0
            var sameLabels = 0
            var identical = 0
            // The starts of the predicted segments matched to a reference cut.
            var chosen = Set<Int>()
            // Reference cuts missed or not identical, then cuts added.
            var differing: [Int] = []
            for (range, labels) in theirs {
                guard
                    let nearest = predicted.min(by: {
                        abs($0.frames.lowerBound - range[0])
                            < abs($1.frames.lowerBound - range[0])
                    }),
                    abs(nearest.frames.lowerBound - range[0]) <= 1
                else {
                    differing.append(range[0])
                    continue
                }
                chosen.insert(nearest.frames.lowerBound)
                matched += 1
                let onFrame = nearest.frames.lowerBound == range[0]
                let labelled =
                    nearest.kind.rawValue == labels.0 && nearest.boundary.rawValue == labels.1
                if onFrame { sameFrame += 1 }
                if labelled { sameLabels += 1 }
                if onFrame, labelled { identical += 1 } else { differing.append(range[0]) }
            }
            let extra = predicted.map(\.frames.lowerBound).filter { !chosen.contains($0) }
            differing += extra
            print(
                "OmniShotCut \(name): \(segments.count) segments vs \(expected.ranges.count); "
                    + "of \(theirs.count) cuts \(matched) within a frame, \(sameFrame) on it, "
                    + "\(sameLabels) with the same labels, \(identical) both, "
                    + "\(extra.count) extra \(extra); "
                    + String(format: "%.1f s", elapsed))
            let statistics = predictor.statistics
            let windows = Double(max(statistics.windows, 1))
            print(
                "OmniShotCut \(name): \(statistics.windows) windows"
                    + (verifier == nil
                        ? "" : ", \(statistics.reruns) run again on the verifier")
                    + String(
                        format: "; per window %.2f ms filling, %.2f ms model, %.2f ms verifier",
                        Self.milliseconds(statistics.filling) / windows,
                        Self.milliseconds(statistics.predicting) / windows,
                        Self.milliseconds(statistics.verifying) / windows))
            if exact {
                XCTAssertEqual(segments.count, expected.ranges.count, name)
                XCTAssertEqual(identical, theirs.count, name)
            } else {
                let nearTies = Set(statistics.nearTies)
                let windowsDiffering = Set(
                    differing.map { window(of: $0, frames: expected.frames) })
                print(
                    "OmniShotCut \(name): \(nearTies.count) near-tie windows; "
                        + "differences in windows \(windowsDiffering.sorted())")
                for position in differing {
                    let index = window(of: position, frames: expected.frames)
                    XCTAssertTrue(
                        nearTies.contains(index),
                        "\(name): the cut at frame \(position) differs in window \(index), "
                            + "whose decisions are no near-tie")
                }
            }
            totalCuts += theirs.count
            totalIdentical += identical
            totalExtra += extra.count
            compared += 1
        }
        try XCTSkipIf(
            compared == 0, "no reference videos in \(videos.path); set OMNISHOTCUT_VIDEOS")
        print(
            "OmniShotCut: \(totalIdentical) of \(totalCuts) cuts identical, \(totalExtra) extra"
        )
        if exact {
            XCTAssertEqual(totalIdentical, totalCuts)
            XCTAssertEqual(totalExtra, 0)
        }
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1e3
            + Double(duration.components.attoseconds) * 1e-15
    }

    private static func sha256(of url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let block = try file.read(upToCount: 1 << 20), !block.isEmpty {
            hash.update(data: block)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
