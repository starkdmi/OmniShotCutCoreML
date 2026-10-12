import Foundation
import Metal

// libswscale as `ffmpeg -s 128x96 -pix_fmt rgb24` runs it on 8-bit 4:2:0 video
// (ffmpeg 8.0.1, libswscale 9.1.100, the version `FrameReaderTests` compares
// against): the legacy scaler, SWS_BICUBIC, horizontal then vertical filtering in
// fixed point, and YUV→RGB through integer tables. Every step is integer
// arithmetic, so the output is bit-exact, not close (`FrameReaderTests`).
//
// The model is sensitive to the difference. A float bicubic with the same
// kernel matched 99.7% of ffmpeg's values and moved logits by 0.04 — more
// than rounding every weight to fp16.

/// `initFilter()` for SWS_BICUBIC with the default parameters (B = 0,
/// C = 0.6), as ffmpeg builds it on arm64 and x86: coefficients computed in
/// 64-bit fixed point, near-zero taps trimmed, the size padded to the SIMD
/// alignment, taps past the edges folded onto the edge pixel, and each row
/// normalized to `one` with the rounding error carried along the row.
struct SwscaleFilter: Sendable {
    /// Taps per output pixel.
    let size: Int
    /// First source pixel of each output pixel's taps.
    let positions: [Int]
    /// `positions.count × size`, each row summing to `one`.
    let coefficients: [Int]

    /// `increment` is swscale's 16.16 step, `((source << 16) + destination / 2)
    /// / destination`; positions are in 1/256 of a pixel from the ideal left
    /// edge (`get_local_pos`).
    init(
        increment: Int, source: Int, destination: Int, alignment: Int, one: Int,
        sourcePosition: Int, destinationPosition: Int
    ) {
        let xInc = Int64(increment)
        let fone = Int64(1) << (54 - min(Self.log2(source / destination), 8))
        var filterSize: Int
        var filter: [[Int64]]
        var positions: [Int]

        if abs(xInc - 0x10000) < 10 && sourcePosition == destinationPosition {
            filterSize = 1
            filter = Array(repeating: [fone], count: destination)
            positions = Array(0..<destination)
        } else {
            let sizeFactor = 4
            filterSize =
                xInc <= 1 << 16
                ? 1 + sizeFactor : 1 + (sizeFactor * source + destination - 1) / destination
            filterSize = max(min(filterSize, source - 2), 1)
            let b: Int64 = 0
            let c = Int64(0.6 * Double(1 << 24))
            filter = []
            positions = []
            var xDstInSrc =
                ((Int64(destinationPosition) * xInc) >> 7)
                - ((Int64(sourcePosition) * 0x10000) >> 7)
            for _ in 0..<destination {
                // C division: truncated toward zero, which Swift's is too.
                var xx = (xDstInSrc - Int64(filterSize - 2) * (1 << 16)) / (1 << 17)
                positions.append(Int(xx))
                var row: [Int64] = []
                for _ in 0..<filterSize {
                    var d = abs(xx * (1 << 17) - xDstInSrc) << 13
                    if xInc > 1 << 16 { d = d * Int64(destination) / Int64(source) }
                    var coefficient: Int64
                    if d >= 1 << 31 {
                        coefficient = 0
                    } else {
                        let dd = (d * d) >> 30
                        let ddd = (dd * d) >> 30
                        if d < 1 << 30 {
                            coefficient =
                                (12 * (1 << 24) - 9 * b - 6 * c) * ddd
                                + (-18 * (1 << 24) + 12 * b + 6 * c) * dd
                                + (6 * (1 << 24) - 2 * b) * (1 << 30)
                        } else {
                            coefficient =
                                (-b - 6 * c) * ddd + (6 * b + 30 * c) * dd
                                + (-12 * b - 48 * c) * d + (8 * b + 24 * c) * (1 << 30)
                        }
                    }
                    row.append(coefficient / ((1 << 54) / fone))
                    xx += 1
                }
                filter.append(row)
                xDstInSrc += 2 * xInc
            }
        }

        // Trim near-zero taps: shift each row left past them, then count the
        // ones on the right.
        let cutoff = 0.002 * Double(fone)  // SWS_MAX_REDUCE_CUTOFF
        var minFilterSize = 0
        for i in stride(from: destination - 1, through: 0, by: -1) {
            var size = filterSize
            var sum: Int64 = 0
            for _ in 0..<filterSize {
                sum += abs(filter[i][0])
                if Double(sum) > cutoff { break }
                // The core cannot handle a filter that moves backwards.
                if i < destination - 1 && positions[i] >= positions[i + 1] { break }
                filter[i] = Array(filter[i].dropFirst()) + [0]
                positions[i] += 1
            }
            sum = 0
            for j in stride(from: filterSize - 1, to: 0, by: -1) {
                sum += abs(filter[i][j])
                if Double(sum) > cutoff { break }
                size -= 1
            }
            minFilterSize = max(minFilterSize, size)
        }

        // NEON and MMX pad horizontal filters to 4 taps and vertical ones to
        // 2, except an unscaled vertical filter.
        let align = minFilterSize == 1 && alignment == 2 ? 1 : alignment
        let size = (minFilterSize + align - 1) & ~(align - 1)
        var f = filter.map { row in (0..<size).map { $0 < filterSize ? row[$0] : 0 } }

        // Fold taps outside the source onto its edge pixels.
        for i in 0..<destination {
            if positions[i] < 0 {
                for j in 1..<max(size, 1) {
                    let left = max(j + positions[i], 0)
                    f[i][left] += f[i][j]
                    f[i][j] = 0
                }
                positions[i] = 0
            }
            if positions[i] + size > source {
                let shift = positions[i] + min(size - source, 0)
                var accumulated: Int64 = 0
                for j in stride(from: size - 1, through: 0, by: -1)
                where positions[i] + j >= source {
                    accumulated += f[i][j]
                    f[i][j] = 0
                }
                for j in stride(from: size - 1, through: 0, by: -1) {
                    f[i][j] = j < shift ? 0 : f[i][j - shift]
                }
                positions[i] -= shift
                f[i][source - 1 - positions[i]] += accumulated
            }
        }

        // Normalize to `one`, carrying each tap's rounding error to the next.
        var coefficients: [Int] = []
        coefficients.reserveCapacity(destination * size)
        for i in 0..<destination {
            var sum = (f[i].reduce(0, +) + Int64(one / 2)) / Int64(one)
            if sum == 0 { sum = 1 }
            var error: Int64 = 0
            for j in 0..<size {
                let v = f[i][j] + error
                let rounded = v >= 0 ? (v + (sum >> 1)) / sum : (v - (sum >> 1)) / sum
                coefficients.append(Int(rounded))
                error = v - rounded * sum
            }
        }
        self.size = size
        self.positions = positions
        self.coefficients = coefficients
    }

    /// `av_log2`: the index of the highest set bit, 0 for 0.
    private static func log2(_ value: Int) -> Int {
        value > 0 ? Int.bitWidth - 1 - value.leadingZeroBitCount : 0
    }
}

/// libswscale's YUV→RGB for 8-bit, 24-bit output (`ff_yuv2rgb_c_init_tables`,
/// `yuv2rgb_X_c_template`): one clipped luma table, indexed by luma plus an
/// integer offset per chroma value.
struct SwscaleRGBTables: Sendable {
    let luma: [UInt8]
    let redV: [Int]
    let greenU: [Int]
    let greenV: [Int]
    let blueU: [Int]

    /// `ff_yuv2rgb_coeffs`, by the stream's matrix.
    enum Matrix: Sendable {
        case bt709, bt601, smpte240m, bt2020, fcc

        var coefficients: (Int64, Int64, Int64, Int64) {
            switch self {
            case .bt709: return (117_489, 138_438, 13_975, 34_925)
            case .bt601: return (104_597, 132_201, 25_675, 53_279)
            case .smpte240m: return (117_579, 136_230, 16_907, 35_559)
            case .bt2020: return (110_013, 140_363, 12_277, 42_626)
            case .fcc: return (104_448, 132_798, 24_759, 53_109)
            }
        }
    }

    init(matrix: Matrix, fullRange: Bool) {
        // C division truncates toward zero; Swift's does too.
        var (crv, cbu, cgu, cgv) = matrix.coefficients
        cgu = -cgu
        cgv = -cgv
        var cy: Int64 = 1 << 16
        var oy: Int64 = 0
        if fullRange {
            crv = crv * 224 / 255
            cbu = cbu * 224 / 255
            cgu = cgu * 224 / 255
            cgv = cgv * 224 / 255
        } else {
            cy = cy * 255 / 219
            oy = 16 << 16
        }
        crv = (crv * 65_536 + 0x8000) / cy
        cbu = (cbu * 65_536 + 0x8000) / cy
        cgu = (cgu * 65_536 + 0x8000) / cy
        cgv = (cgv * 65_536 + 0x8000) / cy

        let headroom: Int64 = 512
        var yb = -(384 << 16) - headroom * cy - oy
        var luma = [UInt8](repeating: 0, count: 2_048)
        for index in luma.indices {
            luma[index] = UInt8(clamping: (yb + 0x8000) >> 16)
            yb += cy
        }
        self.luma = luma
        let offset = (fullRange ? 384 : 326) + Int(headroom)
        func table(_ increment: Int64, base: Int) -> [Int] {
            (0..<256).map { value in
                base - Int(increment >> 9) + Int((Int64(value) * increment) >> 16)
            }
        }
        redV = table(crv, base: offset)
        greenU = table(cgu, base: offset)
        greenV = table(cgv, base: 0)
        blueU = table(cbu, base: offset)
    }
}

/// The stored size of a frame's luma plane and of its interleaved chroma
/// plane, in pixels.
struct PlaneLayout: Sendable, Hashable {
    var width: Int
    var height: Int
    var chromaWidth: Int
    var chromaHeight: Int

    var lumaBytes: Int { width * height }
    var chromaBytes: Int { chromaWidth * chromaHeight * 2 }
}

/// The scale and conversion of a batch of 8-bit 4:2:0 frames on the GPU, exactly as
/// swscale computes it: two Metal kernels that are swscale's C loops in int32
/// — `hScale8To15_c`, then `yuv2rgb_X_c_template` with the table lookups. The
/// integer arithmetic is the MLX port's kernels', line for line.
///
/// The planes are read as stored and turned by quarter turns, and mirrored,
/// as the horizontal pass reads them, so the filters run along the displayed
/// axes, as ffmpeg turns the picture before it scales it.
final class PlaneScaler {
    let layout: PlaneLayout
    let quarterTurns: Int
    let flipped: Bool
    /// The displayed (turned) size of the luma plane.
    let sourceWidth: Int
    let sourceHeight: Int
    let width: Int
    let height: Int
    private let chromaWidth: Int
    private let chromaHeight: Int
    private let chromaOutputWidth: Int
    private let lumaHorizontal: Pass
    private let chromaHorizontal: Pass
    private let vertical: MTLComputePipelineState
    private let lumaVertical: (positions: MTLBuffer, coefficients: MTLBuffer)
    private let chromaVertical: (positions: MTLBuffer, coefficients: MTLBuffer)
    /// `1 << 18` per output row, or 0 on a row that swscale's two-tap path
    /// (`yuv2rgb_2_c_template`) writes without rounding.
    private let rounding: MTLBuffer
    /// The luma table, then the four chroma offset tables, as one array.
    private let tables: MTLBuffer

    /// A horizontal pass: its pipeline, compiled for one plane's shape, and
    /// its filter.
    private struct Pass {
        let pipeline: MTLComputePipelineState
        let positions: MTLBuffer
        let coefficients: MTLBuffer
    }

    init(
        layout: PlaneLayout, quarterTurns: Int, flipped: Bool = false, width: Int, height: Int,
        tables: SwscaleRGBTables, metal: MetalContext
    ) throws {
        self.layout = layout
        self.quarterTurns = quarterTurns
        self.flipped = flipped
        let turned = quarterTurns % 2 == 1
        sourceWidth = turned ? layout.height : layout.width
        sourceHeight = turned ? layout.width : layout.height
        chromaWidth = turned ? layout.chromaHeight : layout.chromaWidth
        chromaHeight = turned ? layout.chromaWidth : layout.chromaHeight
        self.width = width
        self.height = height
        // Packed RGB keeps chroma at half the output width and full height.
        chromaOutputWidth = (width + 1) / 2
        func increment(_ source: Int, _ destination: Int) -> Int {
            ((source << 16) + (destination >> 1)) / destination
        }
        // Chroma is sited at the centre whatever the stream says: ffmpeg's
        // scale filter overwrites each frame's chroma location with its own
        // option, unspecified by default, and swscale reads unspecified as
        // centre. Every position is then 128, the ideal left edge.
        func filter(_ source: Int, _ destination: Int, alignment: Int, one: Int)
            -> SwscaleFilter
        {
            SwscaleFilter(
                increment: increment(source, destination), source: source,
                destination: destination, alignment: alignment, one: one,
                sourcePosition: 128, destinationPosition: 128)
        }
        let lumaH = filter(sourceWidth, width, alignment: 4, one: 1 << 14)
        let lumaV = filter(sourceHeight, height, alignment: 2, one: 1 << 12)
        let chromaH = filter(chromaWidth, chromaOutputWidth, alignment: 4, one: 1 << 14)
        let chromaV = filter(chromaHeight, height, alignment: 2, one: 1 << 12)

        func upload(_ values: [Int]) throws -> MTLBuffer {
            let words = values.map { Int32($0) }
            guard
                let buffer = words.withUnsafeBytes({
                    metal.device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                })
            else { throw OmniShotCutError.metalUnavailable("could not allocate a filter") }
            return buffer
        }
        func horizontal(
            _ filter: SwscaleFilter, width: Int, rows: Int, outputWidth: Int, channels: Int,
            storedWidth: Int, storedHeight: Int
        ) throws -> Pass {
            Pass(
                pipeline: try metal.pipeline(
                    "swscale_hscale8to15", source: Self.horizontalSource,
                    constants: [
                        ("W", width), ("OUT_W", outputWidth), ("SIZE", filter.size),
                        ("CHANNELS", channels), ("ROWS", rows), ("STORED_W", storedWidth),
                        ("STORED_H", storedHeight), ("TURNS", quarterTurns),
                        ("FLIP", flipped ? 1 : 0),
                    ]),
                positions: try upload(filter.positions),
                coefficients: try upload(filter.coefficients))
        }
        lumaHorizontal = try horizontal(
            lumaH, width: sourceWidth, rows: sourceHeight, outputWidth: width, channels: 1,
            storedWidth: layout.width, storedHeight: layout.height)
        chromaHorizontal = try horizontal(
            chromaH, width: chromaWidth, rows: chromaHeight, outputWidth: chromaOutputWidth,
            channels: 2, storedWidth: layout.chromaWidth, storedHeight: layout.chromaHeight)
        lumaVertical = (try upload(lumaV.positions), try upload(lumaV.coefficients))
        chromaVertical = (try upload(chromaV.positions), try upload(chromaV.coefficients))
        vertical = try metal.pipeline(
            "swscale_yuv2rgb24_x", source: Self.verticalSource,
            constants: [
                ("H", sourceHeight), ("CH", chromaHeight), ("OUT_W", width), ("OUT_H", height),
                ("CHROMA_W", chromaOutputWidth), ("LUMA_SIZE", lumaV.size),
                ("CHROMA_SIZE", chromaV.size),
            ])

        let twoTap = lumaV.size == 2 && chromaV.size == 2
        let rows = (0..<height).map { row -> Int in
            func sumsToOne(_ filter: SwscaleFilter) -> Bool {
                let first = filter.coefficients[2 * row]
                let second = filter.coefficients[2 * row + 1]
                return first + second == 4096 && UInt32(truncatingIfNeeded: second) <= 4096
            }
            return twoTap && sumsToOne(lumaV) && sumsToOne(chromaV) ? 0 : 1 << 18
        }
        rounding = try upload(rows)
        self.tables = try upload(
            tables.luma.map(Int.init) + tables.redV + tables.greenU + tables.greenV
                + tables.blueU)
    }

    /// Bytes of the luma pass's 15-bit lines for `frames` frames.
    func lineBytes(frames: Int) -> Int { frames * sourceHeight * width * 4 }

    /// Bytes of the chroma pass's 15-bit lines for `frames` frames.
    func chromaLineBytes(frames: Int) -> Int {
        frames * chromaHeight * chromaOutputWidth * 2 * 4
    }

    /// Encodes the conversion of `frames` frames: `luma` holds them as
    /// stored, `[frames, height, width]` bytes, `chroma` `[frames,
    /// chromaHeight, chromaWidth, 2]`; `rgb` receives `[frames, height,
    /// width, 3]` bytes at the output size. `lines` and `chromaLines` hold
    /// the horizontal passes' output.
    func encode(
        frames: Int, luma: MTLBuffer, chroma: MTLBuffer, lines: MTLBuffer,
        chromaLines: MTLBuffer, rgb: MTLBuffer, into commandBuffer: MTLCommandBuffer
    ) throws {
        // A serial compute encoder: each dispatch sees the previous one's
        // writes.
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw OmniShotCutError.metalUnavailable("could not create a compute encoder")
        }
        func dispatch(
            _ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer], grid: MTLSize,
            group: MTLSize
        ) {
            encoder.setComputePipelineState(pipeline)
            for (index, buffer) in buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
        }
        dispatch(
            lumaHorizontal.pipeline,
            [luma, lumaHorizontal.positions, lumaHorizontal.coefficients, lines],
            grid: MTLSize(width: width, height: sourceHeight, depth: frames),
            group: MTLSize(width: min(width, 64), height: 4, depth: 1))
        dispatch(
            chromaHorizontal.pipeline,
            [chroma, chromaHorizontal.positions, chromaHorizontal.coefficients, chromaLines],
            grid: MTLSize(width: chromaOutputWidth, height: chromaHeight, depth: frames),
            group: MTLSize(width: min(chromaOutputWidth, 64), height: 4, depth: 1))
        dispatch(
            vertical,
            [
                lines, chromaLines, lumaVertical.positions, lumaVertical.coefficients,
                chromaVertical.positions, chromaVertical.coefficients, rounding, tables, rgb,
            ],
            grid: MTLSize(width: width, height: height, depth: frames),
            group: MTLSize(width: min(width, 32), height: 8, depth: 1))
        encoder.endEncoding()
    }

    /// `hScale8To15_c`: each output value is `min(Σ pixel · coefficient >> 7,
    /// 2^15 - 1)`. Taps past the edge carry zero coefficients; their reads are
    /// clamped to the line. `W` and `ROWS` are the displayed plane's width and
    /// height, `STORED_W` and `STORED_H` the stored plane's, `TURNS` the
    /// quarter turns clockwise from stored to displayed, `FLIP` 1 when the
    /// display matrix also mirrors the picture.
    private static let horizontalSource = """
        #include <metal_stdlib>
        using namespace metal;

        // The stored pixel shown at (row, column), turned and mirrored as the
        // filters ffmpeg's autorotation inserts: transpose=clock, hflip,vflip
        // and transpose=cclock for the rotations; vflip, transpose=cclock_flip,
        // hflip and transpose=clock_flip for the mirrored matrices.
        static inline size_t stored_pixel(int row, int column) {
        #if FLIP
        #if TURNS == 1
            return size_t(column) * STORED_W + row;
        #elif TURNS == 2
            return size_t(row) * STORED_W + (STORED_W - 1 - column);
        #elif TURNS == 3
            return size_t(STORED_H - 1 - column) * STORED_W + (STORED_W - 1 - row);
        #else
            return size_t(STORED_H - 1 - row) * STORED_W + column;
        #endif
        #else
        #if TURNS == 1
            return size_t(STORED_H - 1 - column) * STORED_W + row;
        #elif TURNS == 2
            return size_t(STORED_H - 1 - row) * STORED_W + (STORED_W - 1 - column);
        #elif TURNS == 3
            return size_t(column) * STORED_W + (STORED_W - 1 - row);
        #else
            return size_t(row) * STORED_W + column;
        #endif
        #endif
        }

        kernel void swscale_hscale8to15(
            const device uint8_t* source [[buffer(0)]],
            const device int* positions [[buffer(1)]],
            const device int* coefficients [[buffer(2)]],
            device int* lines [[buffer(3)]],
            uint3 thread_position_in_grid [[thread_position_in_grid]])
        {
            uint x = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            uint frame = thread_position_in_grid.z;
            if (x >= uint(OUT_W) || row >= uint(ROWS)) return;
            const device uint8_t* plane =
                source + size_t(frame) * STORED_W * STORED_H * CHANNELS;
            int position = positions[x];
            for (int c = 0; c < CHANNELS; c++) {
                int sum = 0;
                for (int j = 0; j < SIZE; j++) {
                    int tap = min(position + j, W - 1);
                    sum += int(plane[stored_pixel(int(row), tap) * CHANNELS + c])
                        * coefficients[x * SIZE + j];
                }
                lines[((size_t(frame) * ROWS + row) * OUT_W + x) * CHANNELS + c] =
                    min(sum >> 7, 32767);
            }
        }
        """

    /// `yuv2rgb_X_c_template` for RGB24: `(rounding + Σ line · coefficient)
    /// >> 19` for luma and both chroma planes, chroma shared by each pair of
    /// pixels, then the tables — chroma clipped to 0...255, luma within the
    /// table's 512 entries of headroom either side.
    private static let verticalSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void swscale_yuv2rgb24_x(
            const device int* lines [[buffer(0)]],
            const device int* chroma_lines [[buffer(1)]],
            const device int* luma_positions [[buffer(2)]],
            const device int* luma_coefficients [[buffer(3)]],
            const device int* chroma_positions [[buffer(4)]],
            const device int* chroma_coefficients [[buffer(5)]],
            const device int* rounding [[buffer(6)]],
            const device int* tables [[buffer(7)]],
            device uint8_t* rgb [[buffer(8)]],
            uint3 thread_position_in_grid [[thread_position_in_grid]])
        {
            uint x = thread_position_in_grid.x;
            uint y = thread_position_in_grid.y;
            uint frame = thread_position_in_grid.z;
            if (x >= uint(OUT_W) || y >= uint(OUT_H)) return;
            int luma = rounding[y];
            int position = luma_positions[y];
            for (int j = 0; j < LUMA_SIZE; j++) {
                int line = min(position + j, H - 1);
                luma += lines[(size_t(frame) * H + line) * OUT_W + x]
                    * luma_coefficients[y * LUMA_SIZE + j];
            }
            int u = rounding[y];
            int v = rounding[y];
            position = chroma_positions[y];
            for (int j = 0; j < CHROMA_SIZE; j++) {
                int line = min(position + j, CH - 1);
                size_t index = ((size_t(frame) * CH + line) * CHROMA_W + (x >> 1)) * 2;
                int coefficient = chroma_coefficients[y * CHROMA_SIZE + j];
                u += chroma_lines[index] * coefficient;
                v += chroma_lines[index + 1] * coefficient;
            }
            luma = clamp(luma >> 19, -512, 767);
            u = clamp(u >> 19, 0, 255);
            v = clamp(v >> 19, 0, 255);
            // tables: luma[2048], red V[256], green U[256], green V[256], blue U[256]
            const device int* red_v = tables + 2048;
            const device int* green_u = red_v + 256;
            const device int* green_v = green_u + 256;
            const device int* blue_u = green_v + 256;
            size_t out = ((size_t(frame) * OUT_H + y) * OUT_W + x) * 3;
            rgb[out] = uint8_t(tables[red_v[v] + luma]);
            rgb[out + 1] = uint8_t(tables[green_u[u] + green_v[v] + luma]);
            rgb[out + 2] = uint8_t(tables[blue_u[u] + luma]);
        }
        """
}

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
