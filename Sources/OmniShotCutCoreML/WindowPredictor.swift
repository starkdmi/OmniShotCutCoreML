import CoreML
import Foundation
import os

/// A CoreML conversion of OmniShotCut v1.5, run on one window at a time.
///
/// The model reads `frames`, Float16 `[window, 3, height, width]` of RGB
/// values 0–255 with black frames past the end of the video, and returns the
/// `intra`, `inter` and `end` logits of every query, Float16 or Float32, laid
/// out `[queries, classes]` or `[1, classes, 1, queries]`. Each decision is
/// the argmax over every class but the last.
///
/// With a verifier — normally the fp32 conversion on the GPU next to the fp16
/// one on the Neural Engine — a window is run again on the verifier when one
/// of the decisions its segments depend on has a margin below `threshold`
/// (``WindowScores/decisionMargin(validLength:)``), and the verifier's logits
/// replace the first model's. As long as the two models' logits differ by
/// less than `threshold / 2`, the segments are the verifier's.
///
/// One input array is allocated at load and refilled for every window;
/// predictions are serialized.
@_spi(Benchmarking) public final class WindowPredictor: WindowModel, @unchecked Sendable {

    /// Counts and time spent since load or the last
    /// ``resetStatistics()``.
    public struct Statistics: Sendable, Hashable {
        /// Windows predicted.
        public var windows = 0
        /// Windows run again on the verifier.
        public var reruns = 0
        /// Writing frames into the input array.
        public var filling = Duration.zero
        /// The first model's predictions, outputs read.
        public var predicting = Duration.zero
        /// The verifier's predictions, outputs read.
        public var verifying = Duration.zero
        /// The windows, numbered from 0 in the order predicted, whose first
        /// model's decision margin was below ``WindowPredictor/threshold``:
        /// those run again on the verifier, or that would be with one. They
        /// are the only windows where a model computing in fp16 can decide
        /// differently from fp32 while the two differ by less than half the
        /// threshold.
        public var nearTies: [Int] = []

        public init() {}
    }

    /// Time spent making one model ready.
    public struct Loading: Sendable, Hashable {
        /// Compiling an `.mlpackage`; zero for an `.mlmodelc` or a package
        /// compiled before.
        public var compiling: Duration
        /// `MLModel.load`, which on a model's first load on the Neural Engine
        /// includes the ANE compilation.
        public var loading: Duration
    }

    public let configuration: WindowConfiguration
    let contract: ModelContract
    /// The decision margin below which a window is run on the verifier.
    public let threshold: Float
    /// The compiled models in use.
    public let modelURL: URL
    public let verifierURL: URL?
    public let modelLoading: Loading
    public let verifierLoading: Loading?
    /// Whether the first model computes in fp32: its metadata's `precision`
    /// is `fp32`, or without one, its outputs are Float32.
    let computesInFloat32: Bool

    private let model: MLModel
    private let verifier: MLModel?
    private let input: MLMultiArray
    private let inputFeatures: MLFeatureProvider
    private let options = MLPredictionOptions()
    /// Serializes predictions: they share `input`.
    private let predicting = NSLock()
    private let counters = OSAllocatedUnfairLock(initialState: Statistics())

    /// Loads `model`, and `verifier` if given, and checks both against the
    /// model contract. Each is an `.mlmodelc`, used in place, an
    /// `.mlpackage`, compiled once into `Caches/OmniShotCutCoreML`, or a
    /// directory holding one.
    ///
    /// `fastPrediction` sets `MLOptimizationHints.specializationStrategy` to
    /// `.fastPrediction` for both models on macOS 15 and iOS 18 and later:
    /// faster predictions on the Neural Engine after a slower first load.
    public init(
        model: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, verifier: URL? = nil,
        verifierComputeUnits: MLComputeUnits = .cpuAndGPU,
        threshold: Float = OmniShotCut.defaultThreshold,
        fastPrediction: Bool = true
    ) async throws {
        let first = try await Self.load(
            model, computeUnits: computeUnits, fastPrediction: fastPrediction)
        self.model = first.model
        self.contract = first.contract
        self.configuration = first.contract.window
        self.modelURL = first.url
        self.modelLoading = first.loading
        self.computesInFloat32 = first.computesInFloat32
        if let verifier {
            let second = try await Self.load(
                verifier, computeUnits: verifierComputeUnits, fastPrediction: fastPrediction)
            guard second.contract == first.contract else {
                throw OmniShotCutError.modelUnavailable(
                    verifier.path,
                    "reads \(second.contract), the model \(first.contract)")
            }
            self.verifier = second.model
            self.verifierURL = second.url
            self.verifierLoading = second.loading
        } else {
            self.verifier = nil
            self.verifierURL = nil
            self.verifierLoading = nil
        }
        self.threshold = threshold

        let window = first.contract.window
        input = try MLMultiArray(
            shape: [window.windowLength, 3, window.height, window.width].map {
                NSNumber(value: $0)
            }, dataType: .float16)
        _ = input.withUnsafeMutableBytes { bytes, _ in
            bytes.initializeMemory(as: UInt8.self, repeating: 0)
        }
        inputFeatures = try MLDictionaryFeatureProvider(
            dictionary: ["frames": MLFeatureValue(multiArray: input)])
    }

    public var statistics: Statistics { counters.withLock { $0 } }

    public func resetStatistics() { counters.withLock { $0 = Statistics() } }

    /// One window's decisions: `count` frames of `height × width × 3` RGB
    /// bytes, `1...windowLength` of them, the rest of the window black.
    public func predict(frames: UnsafeRawBufferPointer, count: Int) throws
        -> OmniShotCut.WindowLogits
    {
        let window = configuration
        let frameBytes = window.width * window.height * 3
        guard (1...window.windowLength).contains(count), frames.count >= count * frameBytes
        else {
            throw OmniShotCutError.predictionFailed(
                "\(count) frames in \(frames.count) bytes for a window of "
                    + "\(window.windowLength) frames of \(frameBytes) bytes")
        }
        return try predicting.withLock {
            let clock = ContinuousClock()
            let started = clock.now
            try WindowInput.fill(
                input, frames: frames, count: count, height: window.height, width: window.width)
            let filled = clock.now
            let first = try run(model)
            let predicted = clock.now
            // Taken without a verifier too, for `Statistics.nearTies`.
            let nearTie = first.isNearTie(validLength: count, threshold: threshold)
            var verified: WindowScores?
            if let verifier, nearTie {
                verified = try run(verifier)
            }
            let finished = clock.now
            let scores = verified ?? first
            let rerun = verified != nil
            // An argmax over NaN returns class 0 and passes for a decision.
            guard scores.isFinite else {
                let source = rerun ? verifierURL ?? modelURL : modelURL
                throw OmniShotCutError.predictionFailed(
                    "\(source.lastPathComponent) returned logits that are not finite")
            }
            counters.withLock {
                if nearTie { $0.nearTies.append($0.windows) }
                $0.windows += 1
                if rerun { $0.reruns += 1 }
                $0.filling += filled - started
                $0.predicting += predicted - filled
                $0.verifying += finished - predicted
            }
            return scores.decisions
        }
    }

    /// Runs `model` on the input array and reads its three outputs.
    private func run(_ model: MLModel) throws -> WindowScores {
        let outputs: MLFeatureProvider
        do {
            outputs = try model.prediction(from: inputFeatures, options: options)
        } catch {
            throw OmniShotCutError.predictionFailed("\(error)")
        }
        func head(_ name: String, classes: Int) throws -> WindowScores.Head {
            guard let array = outputs.featureValue(for: name)?.multiArrayValue else {
                throw OmniShotCutError.predictionFailed("no \(name) output")
            }
            return try WindowScores.Head(
                array, queries: contract.queries, classes: classes, name: name)
        }
        return WindowScores(
            intra: try head("intra", classes: contract.intraClasses),
            inter: try head("inter", classes: contract.interClasses),
            end: try head("end", classes: contract.endClasses))
    }

    private static func load(
        _ url: URL, computeUnits: MLComputeUnits, fastPrediction: Bool
    ) async throws -> (
        model: MLModel, url: URL, contract: ModelContract, computesInFloat32: Bool,
        loading: Loading
    ) {
        let compiled = try await ModelFiles.compiled(url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        if fastPrediction, #available(macOS 15, iOS 18, *) {
            var hints = configuration.optimizationHints
            hints.specializationStrategy = .fastPrediction
            configuration.optimizationHints = hints
        }
        let clock = ContinuousClock()
        let started = clock.now
        let model: MLModel
        do {
            model = try await MLModel.load(
                contentsOf: compiled.url, configuration: configuration)
        } catch {
            throw OmniShotCutError.modelUnavailable(compiled.url.path, "\(error)")
        }
        let loading = Loading(compiling: compiled.compiling, loading: clock.now - started)
        let description = model.modelDescription
        let metadata = description.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let contract = try ModelContract(metadata: metadata, model: url.path)
        let outputType = try contract.validate(description, model: url.path)
        let float32 = metadata["precision"].map { $0 == "fp32" } ?? (outputType == .float32)
        return (model, compiled.url, contract, float32, loading)
    }
}

/// The shapes a model reads and writes.
struct ModelContract: Sendable, Hashable, CustomStringConvertible {
    var window: WindowConfiguration
    var queries: Int
    var intraClasses: Int
    var interClasses: Int
    var endClasses: Int

    /// The official model's: windows of 100 frames of 128×96, 24 queries,
    /// 10 intra-relation, 7 inter-relation and 102 end classes.
    static let official = ModelContract(
        window: WindowConfiguration(windowLength: 100, width: 128, height: 96), queries: 24,
        intraClasses: 10, interClasses: 7, endClasses: 102)

    /// The creator-defined metadata keys read, each an integer; the official
    /// value for any key missing.
    static let metadataKeys = [
        "window_length", "width", "height", "queries", "intra_classes", "inter_classes",
        "end_classes",
    ]

    init(
        window: WindowConfiguration, queries: Int, intraClasses: Int, interClasses: Int,
        endClasses: Int
    ) {
        self.window = window
        self.queries = queries
        self.intraClasses = intraClasses
        self.interClasses = interClasses
        self.endClasses = endClasses
    }

    /// The official contract, with the values `metadata` sets.
    init(metadata: [String: String], model: String) throws {
        self = .official
        for key in Self.metadataKeys {
            guard let text = metadata[key] else { continue }
            guard let value = Int(text.trimmingCharacters(in: .whitespaces)), value > 0 else {
                throw OmniShotCutError.modelUnavailable(
                    model, "metadata \(key) = \(text) is not a positive integer")
            }
            switch key {
            case "window_length": window.windowLength = value
            case "width": window.width = value
            case "height": window.height = value
            case "queries": queries = value
            case "intra_classes": intraClasses = value
            case "inter_classes": interClasses = value
            default: endClasses = value
            }
        }
    }

    var description: String {
        "frames [\(window.windowLength), 3, \(window.height), \(window.width)], "
            + "\(queries) queries of \(intraClasses), \(interClasses) and \(endClasses) classes"
    }

    /// Checks the model's input and outputs against the contract and returns
    /// the outputs' data type.
    func validate(_ description: MLModelDescription, model: String) throws
        -> MLMultiArrayDataType
    {
        func fail(_ detail: String) -> OmniShotCutError {
            OmniShotCutError.modelUnavailable(model, detail)
        }
        let expected = [window.windowLength, 3, window.height, window.width]
        guard description.inputDescriptionsByName.count == 1,
            let frames = description.inputDescriptionsByName["frames"]?.multiArrayConstraint
        else {
            throw fail(
                "inputs \(description.inputDescriptionsByName.keys.sorted()), expected frames")
        }
        guard frames.dataType == .float16, frames.shape.map(\.intValue) == expected else {
            throw fail(
                "frames is \(WindowScores.Head.describe(frames.dataType)) \(frames.shape), "
                    + "expected Float16 \(expected)")
        }
        var types: [MLMultiArrayDataType] = []
        for (name, classes) in [
            ("intra", intraClasses), ("inter", interClasses), ("end", endClasses),
        ] {
            guard let output = description.outputDescriptionsByName[name]?.multiArrayConstraint
            else { throw fail("no multi-array output \(name)") }
            let shape = output.shape.map(\.intValue)
            guard [.float16, .float32].contains(output.dataType),
                WindowScores.Head.axes(shape: shape, queries: queries, classes: classes) != nil
            else {
                throw fail(
                    "\(name) is \(WindowScores.Head.describe(output.dataType)) \(shape), "
                        + "expected Float16 or Float32 [\(queries), \(classes)] or "
                        + "[1, \(classes), 1, \(queries)]")
            }
            types.append(output.dataType)
        }
        return types.contains(.float32) ? .float32 : .float16
    }
}
