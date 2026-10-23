import Foundation
import Metal

/// The Metal device and queue every frame reader shares, and the scaler's
/// kernels, compiled once per process for each shape.
final class MetalContext: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let lock = NSLock()
    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// `nil` without a Metal device.
    static let shared = MetalContext()

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue()
        else { return nil }
        self.device = device
        self.queue = queue
    }

    /// The kernel `name` of `source`, with `constants` defined as
    /// preprocessor macros.
    func pipeline(_ name: String, source: String, constants: [(String, Int)]) throws
        -> MTLComputePipelineState
    {
        let key = ([name] + constants.map { "\($0.0)=\($0.1)" }).joined(separator: " ")
        lock.lock()
        defer { lock.unlock() }
        if let pipeline = pipelines[key] { return pipeline }
        let options = MTLCompileOptions()
        options.preprocessorMacros = Dictionary(
            constants.map { ($0.0, NSNumber(value: $0.1)) }, uniquingKeysWith: { $1 })
        let pipeline: MTLComputePipelineState
        do {
            let library = try device.makeLibrary(source: source, options: options)
            guard let function = library.makeFunction(name: name) else {
                throw OmniShotCutError.metalUnavailable("\(name) missing from its library")
            }
            pipeline = try device.makeComputePipelineState(function: function)
        } catch let error as OmniShotCutError {
            throw error
        } catch {
            throw OmniShotCutError.metalUnavailable("\(name): \(error)")
        }
        pipelines[key] = pipeline
        return pipeline
    }
}
