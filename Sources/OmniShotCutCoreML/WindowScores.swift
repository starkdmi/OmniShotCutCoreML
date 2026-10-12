import Accelerate
import CoreML
import Foundation

/// One window's logits, read out of the model's three outputs, and the
/// decisions `engine.py` takes from them.
struct WindowScores: Sendable, Hashable {

    /// One head's logits: `queries × classes` values, query by query.
    struct Head: Sendable, Hashable {
        var classes: Int
        var values: [Float]

        var queries: Int { values.count / classes }

        /// The best and the second-best class among all but the last, the
        /// classes the official argmax reads. The best is the first maximum,
        /// as `torch.argmax` returns it; an exact tie makes the second its
        /// equal. Without a second class it is `-1`, at `-infinity`.
        func top2(query: Int) -> (best: Int, bestValue: Float, second: Int, secondValue: Float)
        {
            let row = query * classes
            var best = 0
            var bestValue = values[row]
            var second = -1
            var secondValue = -Float.infinity
            for label in 1..<max(1, classes - 1) {
                let value = values[row + label]
                if value > bestValue {
                    (second, secondValue) = (best, bestValue)
                    (best, bestValue) = (label, value)
                } else if value > secondValue {
                    (second, secondValue) = (label, value)
                }
            }
            return (best, bestValue, second, secondValue)
        }

        /// Every query's argmax over every class but the last.
        var argmaxes: [Int] { (0..<queries).map { top2(query: $0).best } }

        /// The end decision's margin over `validLength` real frames: the
        /// best logit minus the best logit of a class that ends the segment
        /// on another frame. Classes at or past `validLength` all clip to
        /// it, so when the best and the second-best both do, the rival is
        /// the best class before it.
        func endMargin(query: Int, validLength: Int) -> Float {
            let ends = top2(query: query)
            if ends.best < validLength || ends.second < validLength {
                return ends.bestValue - ends.secondValue
            }
            let row = query * classes
            var rival = -Float.infinity
            for label in 0..<min(validLength, classes - 1) {
                rival = max(rival, values[row + label])
            }
            return ends.bestValue - rival
        }
    }

    var intra: Head
    var inter: Head
    var end: Head

    var queries: Int { end.queries }

    /// The argmaxes `decode_window_segments` reads.
    var decisions: OmniShotCut.WindowLogits {
        OmniShotCut.WindowLogits(
            intra: intra.argmaxes, inter: inter.argmaxes, end: end.argmaxes)
    }

    /// The smallest top-1 to top-2 gap among the decisions that decide the
    /// window's segments, `infinity` if there is none, `-infinity` if a logit
    /// is not finite (an fp16 overflow: neither the argmax nor a margin can be
    /// trusted).
    ///
    /// Queries are walked as `decode_window_segments` walks them, over
    /// `validLength` real frames. Every query read has its end decision
    /// counted; a query that produces a segment has its intra and inter
    /// decisions counted too; queries after the one that reaches the last
    /// frame are not read. An end decision's margin is taken against the
    /// best class that ends the segment on another frame
    /// (``Head/endMargin(query:validLength:)``): end frames at or past
    /// `validLength` all clip to it.
    ///
    /// If every logit differs from another model's by less than half the
    /// margin, no decision counted here differs between the two, and neither
    /// do the segments.
    func decisionMargin(validLength: Int) -> Float {
        guard isFinite else { return -.infinity }
        var margin = Float.infinity
        var start = 0
        for query in 0..<queries {
            margin = min(margin, end.endMargin(query: query, validLength: validLength))
            let segmentEnd = min(end.top2(query: query).best, validLength)
            if start >= segmentEnd { continue }
            for head in [intra, inter] {
                let labels = head.top2(query: query)
                margin = min(margin, labels.bestValue - labels.secondValue)
            }
            start = segmentEnd
            if segmentEnd >= validLength { break }
        }
        return margin
    }

    /// Whether every logit is finite. One that is not is an fp16 overflow:
    /// neither the argmax nor a margin can be trusted.
    var isFinite: Bool {
        [intra, inter, end].allSatisfy { $0.values.allSatisfy(\.isFinite) }
    }

    /// Whether a decision the segments depend on has a margin below
    /// `threshold`: the window is run again on the verifier.
    func isNearTie(validLength: Int, threshold: Float) -> Bool {
        decisionMargin(validLength: validLength) < threshold
    }
}

extension WindowScores.Head {

    /// Reads one output of `queries × classes` logits, Float16 or Float32,
    /// laid out `[queries, classes]` or `[1, classes, 1, queries]`, through
    /// the array's own strides.
    init(_ array: MLMultiArray, queries: Int, classes: Int, name: String) throws {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard let axes = Self.axes(shape: shape, queries: queries, classes: classes) else {
            throw OmniShotCutError.predictionFailed(
                "\(name) has shape \(shape), expected [\(queries), \(classes)] or "
                    + "[1, \(classes), 1, \(queries)]")
        }
        let queryStride = strides[axes.query]
        let classStride = strides[axes.classes]
        var values = [Float](repeating: 0, count: queries * classes)
        func read(_ value: (UnsafeRawPointer, Int) -> Float) {
            array.withUnsafeBytes { bytes in
                let base = bytes.baseAddress!
                for query in 0..<queries {
                    for label in 0..<classes {
                        values[query * classes + label] = value(
                            base, query * queryStride + label * classStride)
                    }
                }
            }
        }
        switch array.dataType {
        case .float16:
            read { WindowInput.float(half: $0.load(fromByteOffset: 2 * $1, as: UInt16.self)) }
        case .float32:
            read { $0.load(fromByteOffset: 4 * $1, as: Float.self) }
        default:
            throw OmniShotCutError.predictionFailed(
                "\(name) is \(Self.describe(array.dataType)), expected Float16 or Float32")
        }
        self.init(classes: classes, values: values)
    }

    /// The query and class axes of `[queries, classes]` or
    /// `[1, classes, 1, queries]`, `nil` for any other shape.
    static func axes(shape: [Int], queries: Int, classes: Int) -> (query: Int, classes: Int)? {
        if shape == [queries, classes] { return (0, 1) }
        if shape == [1, classes, 1, queries] { return (3, 1) }
        return nil
    }

    static func describe(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "Float16"
        case .float32: return "Float32"
        case .double: return "Float64"
        case .int32: return "Int32"
        case .int8: return "Int8"
        default: return "data type \(type.rawValue)"
        }
    }
}

/// The model's input: frames as planes of raw RGB values, `[window, 3,
/// height, width]` in Float16.
///
/// Float16 values are handled as their bit patterns: Swift's `Float16` is
/// unavailable on x86_64 macOS.
enum WindowInput {

    /// Writes `count` frames of `height × width × 3` RGB bytes into `input`
    /// as Float16 values 0–255, channel planes apart, through the array's own
    /// strides, and zeros the frames from `count` to the end of the window:
    /// black, as `split_videos` pads before normalizing. Every byte is exact
    /// in Float16.
    ///
    /// With unit column stride, vImage splits each frame into planes and
    /// converts them through Float32 (0–255 maps to 0.0–255.0 exactly), which
    /// costs the same in unoptimized builds as in optimized ones.
    static func fill(
        _ input: MLMultiArray, frames: UnsafeRawBufferPointer, count: Int, height: Int,
        width: Int
    ) throws {
        let window = input.shape[0].intValue
        let source = frames.baseAddress!.assumingMemoryBound(to: UInt8.self)
        try input.withUnsafeMutableBytes { bytes, strides in
            let base = bytes.baseAddress!.assumingMemoryBound(to: UInt16.self)
            let (frameStride, channelStride, rowStride, columnStride) = (
                strides[0], strides[1], strides[2], strides[3]
            )
            if columnStride == 1 {
                try convert(
                    source, count: count, height: height, width: width,
                    into: base, frameStride: frameStride, channelStride: channelStride,
                    rowStride: rowStride)
            } else {
                for frame in 0..<count {
                    for channel in 0..<3 {
                        for row in 0..<height {
                            let line =
                                base + frame * frameStride + channel * channelStride
                                + row * rowStride
                            let pixels = source + (frame * height + row) * width * 3 + channel
                            for column in 0..<width {
                                line[column * columnStride] = half(pixels[3 * column])
                            }
                        }
                    }
                }
            }
            for frame in count..<window {
                for channel in 0..<3 {
                    for row in 0..<height {
                        let line =
                            base + frame * frameStride + channel * channelStride
                            + row * rowStride
                        if columnStride == 1 {
                            line.update(repeating: 0, count: width)
                        } else {
                            for column in 0..<width { line[column * columnStride] = 0 }
                        }
                    }
                }
            }
        }
    }

    /// vImage: RGB888 to three Planar8 planes, each to PlanarF 0.0–255.0,
    /// then to Planar16F in place in the array.
    private static func convert(
        _ source: UnsafePointer<UInt8>, count: Int, height: Int, width: Int,
        into base: UnsafeMutablePointer<UInt16>, frameStride: Int, channelStride: Int,
        rowStride: Int
    ) throws {
        let planes = UnsafeMutablePointer<UInt8>.allocate(capacity: 3 * height * width)
        let floats = UnsafeMutablePointer<Float>.allocate(capacity: height * width)
        defer {
            planes.deallocate()
            floats.deallocate()
        }
        let rows = vImagePixelCount(height)
        let columns = vImagePixelCount(width)
        let flags = vImage_Flags(kvImageDoNotTile)
        func check(_ status: vImage_Error) throws {
            guard status == kvImageNoError else {
                throw OmniShotCutError.predictionFailed(
                    "vImage error \(status) filling the input")
            }
        }
        for frame in 0..<count {
            var rgb = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: source + frame * height * width * 3),
                height: rows, width: columns, rowBytes: width * 3)
            var split = (0..<3).map {
                vImage_Buffer(
                    data: planes + $0 * height * width, height: rows, width: columns,
                    rowBytes: width)
            }
            try check(
                vImageConvert_RGB888toPlanar8(&rgb, &split[0], &split[1], &split[2], flags))
            for channel in 0..<3 {
                var float = vImage_Buffer(
                    data: floats, height: rows, width: columns, rowBytes: width * 4)
                try check(
                    vImageConvert_Planar8toPlanarF(&split[channel], &float, 255, 0, flags))
                var half = vImage_Buffer(
                    data: base + frame * frameStride + channel * channelStride, height: rows,
                    width: columns, rowBytes: rowStride * 2)
                try check(vImageConvert_PlanarFtoPlanar16F(&float, &half, flags))
            }
        }
    }

    /// The Float16 bit pattern of `value`: its Float32 exponent rebiased from
    /// 127 to 15 and the top 10 bits of its mantissa, exact for 8-bit
    /// integers, which have at most 8 significant bits.
    @inline(__always)
    static func half(_ value: UInt8) -> UInt16 {
        let bits = Float(value).bitPattern
        let rebiased = (((bits >> 23) &- 112) << 10) | ((bits >> 13) & 0x3ff)
        return value == 0 ? 0 : UInt16(truncatingIfNeeded: rebiased)
    }

    /// The value of a Float16 bit pattern.
    static func float(half bits: UInt16) -> Float {
        let sign: Float = bits & 0x8000 == 0 ? 1 : -1
        let exponent = UInt32(bits >> 10) & 0x1f
        let mantissa = UInt32(bits & 0x3ff)
        switch exponent {
        case 0: return sign * Float(mantissa) * 0x1p-24
        case 0x1f: return mantissa == 0 ? sign * .infinity : .nan
        default:
            return sign * Float(bitPattern: ((exponent + 112) << 23) | (mantissa << 13))
        }
    }
}
