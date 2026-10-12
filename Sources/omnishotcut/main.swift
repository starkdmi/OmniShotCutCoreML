import CoreML
import Foundation
@_spi(Benchmarking) import OmniShotCutCoreML

// omnishotcut --model <path> [--compute-units ane|gpu|cpu|all]
//     [--verifier <path> [--threshold <t>]] [--no-fast-prediction] [--json] <video>...

let usage = """
    usage: omnishotcut --model <path> [--compute-units ane|gpu|cpu|all]
                       [--verifier <path> [--threshold <t>]] [--no-fast-prediction]
                       [--json] <video>...

    Prints the shots and transitions of each video. A model is a Core ML
    conversion of OmniShotCut v1.5: an .mlpackage, a compiled .mlmodelc, or a
    directory holding one.

      --model               the model run on every window
      --compute-units       the model's compute units: ane (.cpuAndNeuralEngine,
                            the default), gpu (.cpuAndGPU), cpu (.cpuOnly), all
      --verifier            exact mode: windows with a near-tie decision run
                            again on this model, on the GPU (the fp32 conversion)
      --threshold           exact mode's near-tie margin in logits, default \(OmniShotCut.defaultThreshold)
      --no-fast-prediction  load without the .fastPrediction specialization
      --json                print JSON
    """

func fail(_ message: String? = nil) -> Never {
    if let message { FileHandle.standardError.write(Data("omnishotcut: \(message)\n".utf8)) }
    FileHandle.standardError.write(Data((usage + "\n").utf8))
    exit(2)
}

func path(_ argument: String) -> URL {
    URL(fileURLWithPath: (argument as NSString).expandingTildeInPath)
}

var model: URL?
var verifier: URL?
var computeUnits = MLComputeUnits.cpuAndNeuralEngine
var threshold: Float?
var fastPrediction = true
var json = false
var videos: [URL] = []
var arguments = CommandLine.arguments.dropFirst()
while let argument = arguments.popFirst() {
    switch argument {
    case "--model":
        guard let value = arguments.popFirst() else { fail("--model needs a path") }
        model = path(value)
    case "--verifier":
        guard let value = arguments.popFirst() else { fail("--verifier needs a path") }
        verifier = path(value)
    case "--compute-units":
        switch arguments.popFirst() {
        case "ane": computeUnits = .cpuAndNeuralEngine
        case "gpu": computeUnits = .cpuAndGPU
        case "cpu": computeUnits = .cpuOnly
        case "all": computeUnits = .all
        default: fail("--compute-units is ane, gpu, cpu or all")
        }
    case "--threshold":
        guard let value = arguments.popFirst().flatMap(Float.init), value >= 0 else {
            fail("--threshold needs a number, 0 or more")
        }
        threshold = value
    case "--no-fast-prediction": fastPrediction = false
    case "--json": json = true
    case "-h", "--help":
        print(usage)
        exit(0)
    default:
        if argument.hasPrefix("--") { fail("unknown option \(argument)") }
        videos.append(path(argument))
    }
}
guard let model, !videos.isEmpty else { fail() }
if threshold != nil, verifier == nil { fail("--threshold needs --verifier") }

struct Output: Encodable {
    struct Segment: Encodable {
        var start: Double
        var end: Double
        var frames: [Int]
        var kind: String
        var boundary: String
    }
    var video: String
    var seconds: Double
    var segments: [Segment]
    /// Exact mode only: windows predicted, and those run again on the
    /// verifier.
    var windows: Int?
    var reruns: Int?
}

do {
    let detector =
        if let verifier {
            try await OmniShotCut(
                model: model, computeUnits: computeUnits, verifier: verifier,
                threshold: threshold ?? OmniShotCut.defaultThreshold,
                fastPrediction: fastPrediction)
        } else {
            try await OmniShotCut(
                model: model, computeUnits: computeUnits, fastPrediction: fastPrediction)
        }
    let predictor = verifier == nil ? nil : detector.predictor
    var outputs: [Output] = []
    for video in videos {
        predictor?.resetStatistics()
        let started = Date()
        let segments = try await detector.segments(in: video)
        let elapsed = Date().timeIntervalSince(started)
        let statistics = predictor?.statistics
        if json {
            outputs.append(
                Output(
                    video: video.path, seconds: elapsed,
                    segments: segments.map {
                        Output.Segment(
                            start: $0.start, end: $0.end,
                            frames: [$0.frames.lowerBound, $0.frames.upperBound],
                            kind: $0.kind.name, boundary: $0.boundary.name)
                    }, windows: statistics?.windows, reruns: statistics?.reruns))
            continue
        }
        let reruns = statistics.map {
            ", \($0.reruns) of \($0.windows) windows run again on the verifier"
        }
        print(
            "\(video.lastPathComponent): \(segments.count) segments in "
                + String(format: "%.1f s", elapsed) + (reruns ?? ""))
        for segment in segments {
            print(
                String(
                    format: "%9.3f %9.3f  %6d %6d  %@ %@", segment.start, segment.end,
                    segment.frames.lowerBound, segment.frames.upperBound,
                    segment.kind.name.padding(toLength: 9, withPad: " ", startingAt: 0),
                    segment.boundary.name))
        }
    }
    if json {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(decoding: try encoder.encode(outputs), as: UTF8.self))
    }
} catch {
    FileHandle.standardError.write(Data("omnishotcut: \(error)\n".utf8))
    exit(1)
}
