import Foundation
import Metal

// Scales decoded 8-bit 4:2:0 frames to ffmpeg's `-s W×H -pix_fmt rgb24`
// output, byte for byte, following docs/frame-scaler-spec.md. The integer
// weights and colour coefficients are built on the CPU once per scaler; the
// per-pixel work runs in two Metal kernels compiled at run time.

/// A frame's YCbCr→RGB matrix.
enum YCbCrMatrix: Sendable, Hashable {
    case bt601, bt709, smpte240m, bt2020

    /// Red from Cr, blue from Cb, and the magnitudes of green from Cb and
    /// from Cr, in units of 2^-16, for chroma whose 224 steps span the 255
    /// output levels. These are the values ffmpeg's output is made with;
    /// BT.2020's red is one below what its luma weights give.
    fileprivate var coefficients:
        (red: Int64, blue: Int64, greenFromBlue: Int64, greenFromRed: Int64)
    {
        switch self {
        case .bt601: (104597, 132201, 25675, 53279)
        case .bt709: (117489, 138438, 13975, 34925)
        case .smpte240m: (117579, 136230, 16907, 35559)
        case .bt2020: (110013, 140363, 12277, 42626)
        }
    }
}

/// Scales and converts batches of one frame size and colour on the GPU.
///
/// Every frame is read as it is displayed, turned and mirrored, and
/// resampled one axis at a time. The first kernel filters every displayed
/// row of the luma plane to the output width and every displayed row of the
/// chroma plane to half of it, into 16-bit values in the scratch buffer. The
/// second filters those columns to the output height and converts each pixel
/// to RGB.
final class FrameScaler {
    /// The stored plane sizes this scaler was built for.
    let layout: PlaneLayout
    private let width: Int
    private let height: Int
    private let lumaRows: RowPass
    private let chromaRows: RowPass
    private let columns: MTLComputePipelineState
    /// One record per output row: the luma and chroma vertical filters'
    /// first source rows, the rounding term, then the luma weights and the
    /// chroma weights.
    private let columnTable: MTLBuffer

    init(
        layout: PlaneLayout, quarterTurns: Int, flipped: Bool, width: Int, height: Int,
        matrix: YCbCrMatrix, fullRange: Bool, metal: MetalContext
    ) throws {
        precondition(width > 0 && height > 0, "empty output size")
        precondition(
            layout.width > 0 && layout.height > 0 && layout.chromaWidth > 0
                && layout.chromaHeight > 0, "empty planes")
        self.layout = layout
        self.width = width
        self.height = height
        let turns = (quarterTurns % 4 + 4) % 4
        let luma = DisplayedPlane(
            storedColumns: layout.width, storedRows: layout.height, quarterTurns: turns,
            flipped: flipped)
        let chroma = DisplayedPlane(
            storedColumns: layout.chromaWidth, storedRows: layout.chromaHeight,
            quarterTurns: turns, flipped: flipped)

        let lumaAcross = AxisFilter(
            sourceLength: luma.columns, outputLength: width, tapMultiple: 4, unity: 1 << 14)
        let chromaAcross = AxisFilter(
            sourceLength: chroma.columns, outputLength: (width + 1) / 2, tapMultiple: 4,
            unity: 1 << 14)
        let lumaDown = AxisFilter(
            sourceLength: luma.rows, outputLength: height, tapMultiple: 2, unity: 1 << 12)
        let chromaDown = AxisFilter(
            sourceLength: chroma.rows, outputLength: height, tapMultiple: 2, unity: 1 << 12)

        lumaRows = try RowPass(plane: luma, filter: lumaAcross, paired: false, metal: metal)
        chromaRows = try RowPass(
            plane: chroma, filter: chromaAcross, paired: true, metal: metal)

        // The rounding term is dropped from rows where both vertical filters
        // are plain two-tap interpolations.
        let twoTaps = lumaDown.taps == 2 && chromaDown.taps == 2
        func interpolates(_ filter: AxisFilter, _ row: Int) -> Bool {
            let first = filter.weights[row * 2]
            let second = filter.weights[row * 2 + 1]
            return first + second == 1 << 12 && 0 <= second && second <= 1 << 12
        }
        let lumaTaps = lumaDown.storedTaps
        let chromaTaps = chromaDown.storedTaps
        var table: [Int32] = []
        table.reserveCapacity(height * (3 + lumaTaps + chromaTaps))
        for row in 0..<height {
            let rounding =
                twoTaps && interpolates(lumaDown, row) && interpolates(chromaDown, row)
                ? 0 : 1 << 18
            table += [
                Int32(lumaDown.starts[row]), Int32(chromaDown.starts[row]), Int32(rounding),
            ]
            table += lumaDown.storedWeights(position: row)
            table += chromaDown.storedWeights(position: row)
        }
        columnTable = try Self.buffer(table, device: metal.device)

        let colour = ColourConversion(matrix: matrix, fullRange: fullRange)
        columns = try metal.pipeline(
            "scaleColumnsToRGB", source: Self.columnSource,
            constants: [
                ("WIDTH", width), ("HEIGHT", height),
                ("LUMA_ROWS", luma.rows), ("CHROMA_ROWS", chroma.rows),
                ("CHROMA_COLUMNS", chromaAcross.outputLength),
                ("LUMA_TAPS", lumaTaps), ("CHROMA_TAPS", chromaTaps),
                ("RED_FROM_CR", Int(colour.redFromCr)),
                ("BLUE_FROM_CB", Int(colour.blueFromCb)),
                ("GREEN_FROM_CB", Int(colour.greenFromCb)),
                ("GREEN_FROM_CR", Int(colour.greenFromCr)),
                ("LEVEL_SCALE", Int(colour.levelScale)),
                ("LEVEL_OFFSET", Int(colour.levelOffset)),
                ("LEVEL_BASE", Int(colour.levelBase)),
            ])
    }

    /// Bytes of GPU-private scratch `encode` needs for `frames` frames.
    func scratchBytes(frames: Int) -> Int {
        chromaScratchOffset(frames: frames) + frames * chromaRows.filteredBytesPerFrame
    }

    /// The filtered chroma rows follow every frame's filtered luma rows,
    /// aligned to 256 bytes.
    private func chromaScratchOffset(frames: Int) -> Int {
        (frames * lumaRows.filteredBytesPerFrame + 255) & ~255
    }

    /// Encodes the conversion of `frames` frames (section 1's layouts) from
    /// `luma` and `chroma` into `rgb`, using `scratch` for intermediate
    /// values, on `commandBuffer`. It does not commit the command buffer.
    func encode(
        frames: Int, luma: MTLBuffer, chroma: MTLBuffer, scratch: MTLBuffer,
        rgb: MTLBuffer, into commandBuffer: MTLCommandBuffer
    ) throws {
        guard frames > 0 else { return }
        precondition(luma.length >= frames * layout.lumaBytes, "luma buffer too small")
        precondition(chroma.length >= frames * layout.chromaBytes, "chroma buffer too small")
        precondition(scratch.length >= scratchBytes(frames: frames), "scratch too small")
        precondition(rgb.length >= frames * width * height * 3, "RGB buffer too small")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw OmniShotCutError.metalUnavailable("could not create a compute encoder")
        }
        let chromaOffset = chromaScratchOffset(frames: frames)
        lumaRows.encode(frames: frames, plane: luma, filtered: scratch, offset: 0, encoder)
        chromaRows.encode(
            frames: frames, plane: chroma, filtered: scratch, offset: chromaOffset, encoder)

        encoder.setComputePipelineState(columns)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        encoder.setBuffer(scratch, offset: chromaOffset, index: 1)
        encoder.setBuffer(columnTable, offset: 0, index: 2)
        encoder.setBuffer(rgb, offset: 0, index: 3)
        let across = min(columns.threadExecutionWidth, width)
        let down = max(1, min(columns.maxTotalThreadsPerThreadgroup, 256) / across)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: frames),
            threadsPerThreadgroup: MTLSize(width: across, height: down, depth: 1))
        encoder.endEncoding()
    }

    /// A shared buffer holding `values`.
    fileprivate static func buffer(_ values: [Int32], device: MTLDevice) throws -> MTLBuffer {
        let bytes = max(values.count * MemoryLayout<Int32>.stride, 4)
        let made = values.withUnsafeBytes { raw -> MTLBuffer? in
            guard let base = raw.baseAddress else {
                return device.makeBuffer(length: bytes, options: .storageModeShared)
            }
            return device.makeBuffer(bytes: base, length: bytes, options: .storageModeShared)
        }
        guard let made else {
            throw OmniShotCutError.metalUnavailable("could not allocate \(bytes) bytes")
        }
        return made
    }

    /// Resamples every displayed row of one plane, from `plane` to `OUTPUTS`
    /// filtered values per row. `table` holds each output's first source
    /// column, then the weights, all outputs' first taps, then all outputs'
    /// second taps, and so on. With `PAIRED`, every sample is a Cb, Cr pair
    /// filtered with the same weights.
    fileprivate static let rowSource = """
        #include <metal_stdlib>
        using namespace metal;

        #if PAIRED
        typedef uchar2 Sample;
        typedef int2 Total;
        typedef short2 Filtered;
        #else
        typedef uchar Sample;
        typedef int Total;
        typedef short Filtered;
        #endif

        kernel void scaleDisplayedRows(
            device const Sample *plane [[buffer(0)]],
            device const int *table [[buffer(1)]],
            device Filtered *filtered [[buffer(2)]],
            uint3 position [[thread_position_in_grid]])
        {
            const int column = int(position.x);
            const int row = int(position.y);
            const int frame = int(position.z);
            if (column >= OUTPUTS || row >= ROWS) return;
            const int first = frame * (PLANE_SAMPLES) + (ORIGIN) + row * (ROW_STEP)
                + table[column] * (COLUMN_STEP);
            device const Sample *source = plane + first;
            device const int *weights = table + OUTPUTS + column;
            Total total = Total(0);
            for (int tap = 0; tap < TAPS; ++tap) {
                total += Total(source[tap * (COLUMN_STEP)]) * weights[tap * OUTPUTS];
            }
            filtered[(frame * ROWS + row) * OUTPUTS + column] =
                Filtered(min(total >> 7, Total(32767)));
        }
        """

    /// Resamples the filtered rows down every output column and converts
    /// each output pixel to RGB. `table` holds one record per output row: the
    /// first luma and chroma rows, the rounding term, the luma weights and
    /// the chroma weights. Chroma is read at half the output width.
    fileprivate static let columnSource = """
        #include <metal_stdlib>
        using namespace metal;

        // A chroma value's share of an output level, centred on chroma 128.
        static inline int chromaTerm(int coefficient, int chroma) {
            return -(coefficient >> 9) + ((chroma * coefficient) >> 16);
        }

        // The output byte for a luma value plus its chroma terms.
        static inline uchar outputLevel(int index) {
            return uchar(clamp((index * (LEVEL_SCALE) + (LEVEL_OFFSET)) >> 16, 0, 255));
        }

        kernel void scaleColumnsToRGB(
            device const short *luma [[buffer(0)]],
            device const short2 *chroma [[buffer(1)]],
            device const int *table [[buffer(2)]],
            device uchar *rgb [[buffer(3)]],
            uint3 position [[thread_position_in_grid]])
        {
            const int x = int(position.x);
            const int y = int(position.y);
            const int frame = int(position.z);
            if (x >= WIDTH || y >= HEIGHT) return;
            device const int *record = table + y * (3 + LUMA_TAPS + CHROMA_TAPS);
            const int rounding = record[2];

            device const short *lumaColumn =
                luma + (frame * LUMA_ROWS + record[0]) * WIDTH + x;
            int lumaTotal = rounding;
            device const int *lumaWeights = record + 3;
            for (int tap = 0; tap < LUMA_TAPS; ++tap) {
                lumaTotal += int(lumaColumn[tap * WIDTH]) * lumaWeights[tap];
            }
            device const short2 *chromaColumn =
                chroma + (frame * CHROMA_ROWS + record[1]) * CHROMA_COLUMNS + x / 2;
            int2 chromaTotal = int2(rounding);
            device const int *chromaWeights = record + 3 + LUMA_TAPS;
            for (int tap = 0; tap < CHROMA_TAPS; ++tap) {
                chromaTotal += int2(chromaColumn[tap * CHROMA_COLUMNS]) * chromaWeights[tap];
            }

            const int lumaValue = clamp(lumaTotal >> 19, -512, 767);
            const int2 chromaValue = clamp(chromaTotal >> 19, int2(0), int2(255));
            const int index = LEVEL_BASE + lumaValue;
            device uchar *pixel = rgb + ((frame * HEIGHT + y) * WIDTH + x) * 3;
            pixel[0] = outputLevel(index + chromaTerm(RED_FROM_CR, chromaValue.y));
            pixel[1] = outputLevel(
                index + chromaTerm(GREEN_FROM_CB, chromaValue.x)
                    + chromaTerm(GREEN_FROM_CR, chromaValue.y));
            pixel[2] = outputLevel(index + chromaTerm(BLUE_FROM_CB, chromaValue.x));
        }
        """
}

/// The horizontal pass over one plane: its kernel, its weight table, and
/// the shape of its output.
private struct RowPass {
    let pipeline: MTLComputePipelineState
    let table: MTLBuffer
    /// Filtered values per row.
    let outputs: Int
    /// Displayed rows per frame.
    let rows: Int
    /// Whether a displayed row runs along the stored rows, so that
    /// neighbouring outputs read neighbouring bytes.
    let alongStoredRows: Bool
    /// Bytes per filtered value: a 16-bit value, or a pair for chroma.
    let valueBytes: Int

    var filteredBytesPerFrame: Int { rows * outputs * valueBytes }

    init(plane: DisplayedPlane, filter: AxisFilter, paired: Bool, metal: MetalContext) throws {
        outputs = filter.outputLength
        rows = plane.rows
        alongStoredRows = abs(plane.columnStep) == 1
        valueBytes = paired ? 4 : 2
        let taps = filter.storedTaps
        var values = filter.starts.map { Int32($0) }
        values.reserveCapacity(outputs * (1 + taps))
        for tap in 0..<taps {
            for position in 0..<outputs {
                values.append(Int32(filter.weights[position * filter.taps + tap]))
            }
        }
        table = try FrameScaler.buffer(values, device: metal.device)
        pipeline = try metal.pipeline(
            "scaleDisplayedRows", source: FrameScaler.rowSource,
            constants: [
                ("PAIRED", paired ? 1 : 0), ("OUTPUTS", outputs), ("ROWS", rows),
                ("TAPS", taps), ("PLANE_SAMPLES", plane.storedSamples),
                ("ORIGIN", plane.origin), ("ROW_STEP", plane.rowStep),
                ("COLUMN_STEP", plane.columnStep),
            ])
    }

    func encode(
        frames: Int, plane: MTLBuffer, filtered: MTLBuffer, offset: Int,
        _ encoder: MTLComputeCommandEncoder
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(plane, offset: 0, index: 0)
        encoder.setBuffer(table, offset: 0, index: 1)
        encoder.setBuffer(filtered, offset: offset, index: 2)
        let threads = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        // When a displayed row runs down the stored columns, a SIMD group
        // takes neighbouring displayed rows of one output instead, which
        // read neighbouring bytes.
        let across =
            alongStoredRows ? min(pipeline.threadExecutionWidth, outputs) : 1
        let down = max(1, threads / across)
        encoder.dispatchThreads(
            MTLSize(width: outputs, height: rows, depth: frames),
            threadsPerThreadgroup: MTLSize(width: across, height: down, depth: 1))
    }
}

/// A plane stored `storedColumns × storedRows` as it is displayed: its
/// displayed size, and where displayed sample `(row, column)` is stored, at
/// `origin + row · rowStep + column · columnStep` samples from the start of
/// the plane.
private struct DisplayedPlane {
    let columns: Int
    let rows: Int
    let origin: Int
    let rowStep: Int
    let columnStep: Int
    let storedSamples: Int

    /// `quarterTurns` clockwise quarter turns display the stored picture,
    /// mirrored too when `flipped`.
    init(storedColumns: Int, storedRows: Int, quarterTurns: Int, flipped: Bool) {
        let lastRow = storedRows - 1
        let lastColumn = storedColumns - 1
        // The stored row and column of displayed (row, column), each as
        // (constant, per displayed row, per displayed column).
        let storedRow: (Int, Int, Int)
        let storedColumn: (Int, Int, Int)
        switch (quarterTurns, flipped) {
        case (0, false): (storedRow, storedColumn) = ((0, 1, 0), (0, 0, 1))
        case (0, true): (storedRow, storedColumn) = ((lastRow, -1, 0), (0, 0, 1))
        case (1, false): (storedRow, storedColumn) = ((lastRow, 0, -1), (0, 1, 0))
        case (1, true): (storedRow, storedColumn) = ((0, 0, 1), (0, 1, 0))
        case (2, false): (storedRow, storedColumn) = ((lastRow, -1, 0), (lastColumn, 0, -1))
        case (2, true): (storedRow, storedColumn) = ((0, 1, 0), (lastColumn, 0, -1))
        case (3, false): (storedRow, storedColumn) = ((0, 0, 1), (lastColumn, -1, 0))
        default: (storedRow, storedColumn) = ((lastRow, 0, -1), (lastColumn, -1, 0))
        }
        let sideways = quarterTurns % 2 == 1
        columns = sideways ? storedRows : storedColumns
        rows = sideways ? storedColumns : storedRows
        origin = storedRow.0 * storedColumns + storedColumn.0
        rowStep = storedRow.1 * storedColumns + storedColumn.1
        columnStep = storedRow.2 * storedColumns + storedColumn.2
        storedSamples = storedColumns * storedRows
    }
}

/// Integer resampling weights along one axis: output position `i` is the
/// sum of `taps` consecutive source samples from `starts[i]`, weighted by
/// `weights[i · taps ..< (i + 1) · taps]`, which sum to about `unity`.
///
/// The kernel is the Mitchell–Netravali cubic with B = 0 and C = 0.6,
/// evaluated in fixed point and stretched by the reduction when shrinking.
/// Negligible weights are trimmed from both ends, the width is rounded up to
/// a multiple, taps past either edge are folded onto the edge sample, and
/// each position is quantized to `unity` with its rounding error carried
/// from tap to tap.
private struct AxisFilter {
    let sourceLength: Int
    let outputLength: Int
    /// Taps per output position.
    let taps: Int
    let starts: [Int]
    let weights: [Int]

    /// Taps the kernels read: `taps`, but no more than the source has. When
    /// there are more taps than source samples, every position starts at 0
    /// and the taps past the source have weight 0.
    var storedTaps: Int { min(taps, sourceLength) }

    /// Position `position`'s first `storedTaps` weights.
    func storedWeights(position: Int) -> [Int32] {
        (0..<storedTaps).map { Int32(weights[position * taps + $0]) }
    }

    init(sourceLength: Int, outputLength: Int, tapMultiple: Int, unity: Int64) {
        self.sourceLength = sourceLength
        self.outputLength = outputLength
        let source = Int64(sourceLength)
        let output = Int64(outputLength)
        let positions = outputLength
        let one: Int64 = 1 << 16
        // The source step per output sample in 16.16 fixed point.
        let step = ((source << 16) + output / 2) / output

        // Candidate taps per position, their first source index, and their
        // raw weights: 6 · 2^54 times the kernel, divided by 2^reduction.
        let candidates: Int
        var first: [Int64]
        var raw: [Int64]
        var reduction = 0
        if abs(step - one) < 10 {
            candidates = 1
            first = Array(0..<Int64(positions))
            raw = Array(repeating: 1 << 54, count: positions)
        } else {
            let shrinking = step > one
            var count = shrinking ? 1 + (4 * source + output - 1) / output : 5
            count = max(min(count, source - 2), 1)
            candidates = Int(count)
            let ratio = source / output
            if ratio >= 1 {
                reduction = min(Int64.bitWidth - 1 - ratio.leadingZeroBitCount, 8)
            }
            first = []
            first.reserveCapacity(positions)
            raw = []
            raw.reserveCapacity(positions * candidates)
            for position in 0..<Int64(positions) {
                // The output sample's centre in units of 2^-17 source samples.
                let centre = (2 * position + 1) * step - one
                // Truncated, not floored, where it is negative.
                let start = (centre - (count - 2) * one) / (1 << 17)
                first.append(start)
                for tap in 0..<count {
                    // Distance from the centre in units of 2^-30 samples.
                    var distance = abs((start + tap) * (1 << 17) - centre) * (1 << 13)
                    if shrinking { distance = distance * output / source }
                    raw.append(Self.cubic(distance) / (1 << Int64(reduction)))
                }
            }
        }

        // Trim taps that together weigh no more than 0.002 · 2^(54 - reduction)
        // from both ends, last position first, so that each sees its
        // successor's new start.
        let threshold = 0.002 * Double(Int64(1) << Int64(54 - reduction))
        var widest = 0
        for position in stride(from: positions - 1, through: 0, by: -1) {
            let base = position * candidates
            var dropped: Int64 = 0
            for _ in 0..<candidates {
                dropped += abs(raw[base])
                if Double(dropped) > threshold { break }
                if position < positions - 1, first[position] >= first[position + 1] { break }
                for tap in 0..<(candidates - 1) { raw[base + tap] = raw[base + tap + 1] }
                raw[base + candidates - 1] = 0
                first[position] += 1
            }
            var tail: Int64 = 0
            var length = candidates
            for tap in stride(from: candidates - 1, through: 1, by: -1) {
                tail += abs(raw[base + tap])
                if Double(tail) > threshold { break }
                length -= 1
            }
            widest = max(widest, length)
        }

        // Round the width up to the multiple; a one-tap vertical filter
        // stays one tap wide.
        let multiple = tapMultiple == 2 && widest == 1 ? 1 : tapMultiple
        let taps = (widest + multiple - 1) / multiple * multiple
        self.taps = taps

        var starts: [Int] = []
        starts.reserveCapacity(positions)
        var weights: [Int] = []
        weights.reserveCapacity(positions * taps)
        for position in 0..<positions {
            var row = [Int64](repeating: 0, count: taps)
            for tap in 0..<min(taps, candidates) {
                row[tap] = raw[position * candidates + tap]
            }
            var start = first[position]
            // Taps before the first sample fold onto it.
            if start < 0 {
                var folded = [Int64](repeating: 0, count: taps)
                for tap in 0..<taps {
                    folded[Int(max(start + Int64(tap), 0))] += row[tap]
                }
                row = folded
                start = 0
            }
            // Taps past the last sample fold onto it, and the position moves
            // back to end there.
            if start + Int64(taps) > source {
                let moved = max(source - Int64(taps), 0)
                var folded = [Int64](repeating: 0, count: taps)
                for tap in 0..<taps {
                    folded[Int(min(start + Int64(tap), source - 1) - moved)] += row[tap]
                }
                row = folded
                start = moved
            }
            starts.append(Int(start))
            weights += Self.quantize(row, unity: unity).map { Int($0) }
        }
        self.starts = starts
        self.weights = weights
    }

    /// The Mitchell–Netravali cubic with B = 0 and C = 0.6 at `distance`
    /// (units of 2^-30 samples), times 6 · 2^54.
    private static func cubic(_ distance: Int64) -> Int64 {
        let unit: Int64 = 1 << 30
        if distance >= 2 * unit { return 0 }
        // B and C in units of 2^-24.
        let one: Int64 = 1 << 24
        let b: Int64 = 0
        let c: Int64 = 10_066_329
        let squared = (distance * distance) >> 30
        let cubed = (squared * distance) >> 30
        if distance < unit {
            let cubic = 12 * one - 9 * b - 6 * c
            let quadratic = -18 * one + 12 * b + 6 * c
            let constant = 6 * one - 2 * b
            return cubic * cubed + quadratic * squared + constant * unit
        }
        let cubic = -b - 6 * c
        let quadratic = 6 * b + 30 * c
        let linear = -12 * b - 48 * c
        let constant = 8 * b + 24 * c
        return cubic * cubed + quadratic * squared + linear * distance + constant * unit
    }

    /// `row` rescaled to sum to `unity`, every tap rounded half away from
    /// zero and its rounding error carried into the next.
    private static func quantize(_ row: [Int64], unity: Int64) -> [Int64] {
        let total = row.reduce(0, +)
        var divisor = (total + unity / 2) / unity
        if divisor == 0 { divisor = 1 }
        let half = floorDivide(divisor, 2)
        var carry: Int64 = 0
        return row.map { weight in
            let value = weight + carry
            let magnitude = floorDivide(abs(value) + half, divisor)
            let quantized = value < 0 ? -magnitude : magnitude
            carry = value - quantized * divisor
            precondition(Int64(Int32.min) <= quantized && quantized <= Int64(Int32.max))
            return quantized
        }
    }

    private static func floorDivide(_ a: Int64, _ b: Int64) -> Int64 {
        let quotient = a / b
        return (a % b != 0) && ((a < 0) != (b < 0)) ? quotient - 1 : quotient
    }
}

/// The YCbCr→RGB arithmetic for one matrix and range: chroma coefficients in
/// units of 2^-16 per chroma step, and the luma scale and offset of the
/// final level mapping.
private struct ColourConversion {
    let redFromCr: Int64
    let blueFromCb: Int64
    let greenFromCb: Int64
    let greenFromCr: Int64
    /// Output levels per luma step, in units of 2^-16.
    let levelScale: Int64
    /// The level mapping's constant, rounding included, in units of 2^-16.
    let levelOffset: Int64
    /// Added to the luma value before the level mapping.
    let levelBase: Int64

    init(matrix: YCbCrMatrix, fullRange: Bool) {
        let given = matrix.coefficients
        var values = [given.red, given.blue, -given.greenFromBlue, -given.greenFromRed]
        let scale: Int64
        let lumaOffset: Int64
        if fullRange {
            // Chroma spans 255 steps, not 224.
            values = values.map { $0 * 224 / 255 }
            scale = 1 << 16
            lumaOffset = 0
            levelBase = 896
        } else {
            scale = (255 << 16) / 219
            lumaOffset = 16 << 16
            levelBase = 838
        }
        // Truncated toward zero, the negative green terms too.
        values = values.map { ($0 * (1 << 16) + (1 << 15)) / scale }
        redFromCr = values[0]
        blueFromCb = values[1]
        greenFromCb = values[2]
        greenFromCr = values[3]
        levelScale = scale
        levelOffset = -384 * (1 << 16) - 512 * scale - lumaOffset + (1 << 15)
    }
}
