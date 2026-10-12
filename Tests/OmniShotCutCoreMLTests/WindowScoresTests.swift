import CoreML
import XCTest

@_spi(Benchmarking) @testable import OmniShotCutCoreML

/// The model wrapper without a model: the near-tie gate, the input array's
/// contents and the reading of both output layouts.
final class WindowScoresTests: XCTestCase {

    // MARK: - Near-tie gate

    /// 24 queries of 10, 7 and 102 classes. Query `q` ends at `ends[q]` and
    /// takes intra class 0 and inter class 1; every best logit is 10 and
    /// every other 0, so every decision's margin is 10. Queries past `ends`
    /// end at frame 100.
    private func scores(ends: [Int]) -> WindowScores {
        func head(_ classes: Int, best: (Int) -> Int) -> WindowScores.Head {
            var values = [Float](repeating: 0, count: 24 * classes)
            for query in 0..<24 { values[query * classes + best(query)] = 10 }
            return WindowScores.Head(classes: classes, values: values)
        }
        return WindowScores(
            intra: head(10) { _ in 0 }, inter: head(7) { _ in 1 },
            end: head(102) { $0 < ends.count ? ends[$0] : 100 })
    }

    func testDecisionsAreArgmaxesOverEveryClassButTheLast() {
        var window = scores(ends: [30, 60, 100])
        // The last class is padding and never read, however large.
        window.intra.values[0 * 10 + 9] = 50
        window.end.values[1 * 102 + 101] = 50
        // An exact tie goes to the first class, as torch.argmax.
        window.inter.values[2 * 7 + 0] = 10
        let decisions = window.decisions
        XCTAssertEqual(Array(decisions.end.prefix(4)), [30, 60, 100, 100])
        XCTAssertEqual(decisions.intra[0], 0)
        XCTAssertEqual(decisions.inter[2], 0)
        XCTAssertEqual(decisions.inter[1], 1)
        XCTAssertEqual(window.decisionMargin(validLength: 100), 0)
    }

    func testAClearWindowIsNotRunAgain() {
        let window = scores(ends: [30, 60, 100])
        XCTAssertEqual(window.decisionMargin(validLength: 100), 10)
        XCTAssertFalse(window.isNearTie(validLength: 100, threshold: 1))
    }

    func testMarginsJustBelowAndAboveTheThreshold() {
        for (head, label) in [(0, 3), (1, 4), (2, 70)] {
            for (gap, nearTie) in [(Float(0.99), true), (1.01, false)] {
                var window = scores(ends: [30, 60, 100])
                // Query 1's runner-up, `gap` below its best.
                switch head {
                case 0: window.intra.values[1 * 10 + label] = 10 - gap
                case 1: window.inter.values[1 * 7 + label] = 10 - gap
                default: window.end.values[1 * 102 + label] = 10 - gap
                }
                XCTAssertEqual(
                    window.decisionMargin(validLength: 100), gap, accuracy: 1e-5,
                    "head \(head)")
                XCTAssertEqual(
                    window.isNearTie(validLength: 100, threshold: 1), nearTie,
                    "head \(head), gap \(gap)")
            }
        }
    }

    /// Two end frames at or past the last real frame both clip to it.
    func testEndTiesPastTheLastFrameAreIgnored() {
        var window = scores(ends: [30, 80])
        window.end.values[1 * 102 + 70] = 9.9
        XCTAssertEqual(window.decisionMargin(validLength: 60), 10)
        XCTAssertFalse(window.isNearTie(validLength: 60, threshold: 1))
        // With all 100 frames real the two ends differ.
        XCTAssertTrue(window.isNearTie(validLength: 100, threshold: 1))
        // A third frame before the last one, close behind both: it would
        // end the segment elsewhere, so it sets the margin.
        window.end.values[1 * 102 + 50] = 9.5
        XCTAssertEqual(window.decisionMargin(validLength: 60), 0.5, accuracy: 1e-5)
        XCTAssertTrue(window.isNearTie(validLength: 60, threshold: 1))
        // One of the two before the last frame: the decision matters.
        window.end.values[1 * 102 + 50] = 9.95
        XCTAssertEqual(window.decisionMargin(validLength: 60), 0.05, accuracy: 1e-5)
    }

    /// A query that does not move forward makes no segment: its labels are
    /// not read, its end is.
    func testSkippedQueriesLabelsAreIgnored() {
        var window = scores(ends: [30, 30, 100])
        window.intra.values[1 * 10 + 5] = 10
        window.inter.values[1 * 7 + 2] = 10
        XCTAssertEqual(window.decisionMargin(validLength: 100), 10)
        window.end.values[1 * 102 + 40] = 9.5
        XCTAssertEqual(window.decisionMargin(validLength: 100), 0.5, accuracy: 1e-5)
    }

    /// Decoding stops at the query that reaches the last real frame.
    func testQueriesAfterTheLastFrameAreNotRead() {
        var window = scores(ends: [50, 100, 20])
        for query in 2..<24 {
            window.intra.values[query * 10 + 1] = 10
            window.end.values[query * 102 + 7] = 10
        }
        XCTAssertEqual(window.decisionMargin(validLength: 100), 10)
        // A short window: the first query already reaches its last frame.
        window.intra.values[1 * 10 + 1] = 10
        XCTAssertEqual(window.decisionMargin(validLength: 40), 10)
        XCTAssertEqual(window.decisionMargin(validLength: 100), 0)
    }

    /// An overflowed logit anywhere, read or not, runs the window again, and
    /// marks the scores as not finite, which the predictor reports rather
    /// than decide from.
    func testNonFiniteLogitsAreNearTies() {
        for value in [Float.nan, .infinity, -.infinity] {
            var window = scores(ends: [100])
            XCTAssertTrue(window.isFinite)
            XCTAssertFalse(window.isNearTie(validLength: 100, threshold: 1))
            window.end.values[0 * 102 + 100] = value
            XCTAssertFalse(window.isFinite, "\(value)")
            XCTAssertEqual(window.decisionMargin(validLength: 100), -.infinity, "\(value)")
            XCTAssertTrue(window.isNearTie(validLength: 100, threshold: 1), "\(value)")
            window = scores(ends: [100])
            window.intra.values[23 * 10 + 4] = value
            XCTAssertFalse(window.isFinite, "\(value)")
            XCTAssertTrue(window.isNearTie(validLength: 100, threshold: 1), "\(value)")
        }
    }

    // MARK: - Input

    func testFramesBecomePlanesOfRawValuesAndPaddingIsBlack() throws {
        let (window, height, width) = (4, 8, 16)
        let input = try MLMultiArray(shape: [4, 3, 8, 16], dataType: .float16)
        // Every byte value, differing along frames, rows, columns and
        // channels.
        let bytes = (0..<(window * height * width * 3)).map {
            UInt8(truncatingIfNeeded: $0 * 7 + 3)
        }
        XCTAssertEqual(Set(bytes).count, 256)
        try bytes.withUnsafeBytes {
            try WindowInput.fill(input, frames: $0, count: 4, height: height, width: width)
        }
        try checkInput(input, bytes: bytes, count: 4, height: height, width: width)
        try bytes.withUnsafeBytes {
            try WindowInput.fill(input, frames: $0, count: 2, height: height, width: width)
        }
        try checkInput(input, bytes: bytes, count: 2, height: height, width: width)
    }

    /// Input arrays with padded rows, and with every other column: the fill
    /// follows the strides and leaves the gaps alone.
    func testFillFollowsTheArraysStrides() throws {
        for (height, width, strides) in [
            (8, 16, [3 * 8 * 24, 8 * 24, 24, 1]), (2, 4, [3 * 2 * 16, 2 * 16, 16, 2]),
        ] {
            let window = 3
            let storage = UnsafeMutablePointer<UInt16>.allocate(capacity: window * strides[0])
            storage.initialize(repeating: 0xffff, count: window * strides[0])
            let input = try MLMultiArray(
                dataPointer: storage,
                shape: [window, 3, height, width].map { NSNumber(value: $0) },
                dataType: .float16, strides: strides.map { NSNumber(value: $0) },
                deallocator: { $0.deallocate() })
            let bytes = (0..<(window * height * width * 3)).map {
                UInt8(truncatingIfNeeded: $0 * 11)
            }
            try bytes.withUnsafeBytes {
                try WindowInput.fill(input, frames: $0, count: 2, height: height, width: width)
            }
            try checkInput(input, bytes: bytes, count: 2, height: height, width: width)
            var written = 0
            for index in 0..<(window * strides[0]) where storage[index] != 0xffff {
                written += 1
            }
            XCTAssertEqual(written, window * 3 * height * width, "strides \(strides)")
        }
    }

    func testEveryByteIsExactInFloat16() {
        for value in 0...255 {
            let half = WindowInput.half(UInt8(value))
            XCTAssertEqual(WindowInput.float(half: half), Float(value))
            #if arch(arm64)
                XCTAssertEqual(half, Float16(value).bitPattern)
            #endif
        }
    }

    func testFloat16BitPatterns() {
        XCTAssertEqual(WindowInput.float(half: 0x3c00), 1)
        XCTAssertEqual(WindowInput.float(half: 0xc000), -2)
        XCTAssertEqual(WindowInput.float(half: 0x3555), 0.333_251_953_125)
        XCTAssertEqual(WindowInput.float(half: 0x0001), 0x1p-24)
        XCTAssertEqual(WindowInput.float(half: 0x8000), 0)
        XCTAssertEqual(WindowInput.float(half: 0x7bff), 65504)
        XCTAssertEqual(WindowInput.float(half: 0xfc00), -.infinity)
        XCTAssertTrue(WindowInput.float(half: 0x7e00).isNaN)
    }

    /// Element `[frame, channel, row, column]` of `input` against byte
    /// `frame, row, column, channel` of `bytes`, zero from frame `count`.
    private func checkInput(
        _ input: MLMultiArray, bytes: [UInt8], count: Int, height: Int, width: Int
    ) throws {
        let strides = input.strides.map(\.intValue)
        var wrong = 0
        input.withUnsafeBytes { raw in
            for frame in 0..<input.shape[0].intValue {
                for channel in 0..<3 {
                    for row in 0..<height {
                        for column in 0..<width {
                            let offset =
                                frame * strides[0] + channel * strides[1] + row * strides[2]
                                + column * strides[3]
                            let value = WindowInput.float(
                                half: raw.load(fromByteOffset: 2 * offset, as: UInt16.self))
                            let expected =
                                frame < count
                                ? Float(
                                    bytes[
                                        ((frame * height + row) * width + column) * 3 + channel]
                                )
                                : 0
                            if value != expected { wrong += 1 }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(wrong, 0, "\(count) frames")
    }

    // MARK: - Outputs

    /// Logit `query + class / 4`, exact in Float16.
    private func logit(_ query: Int, _ label: Int) -> Float { Float(query) + Float(label) / 4 }

    func testOutputsAreReadInBothLayoutsAndTypes() throws {
        let (queries, classes) = (24, 10)
        // [24, 10] Float16 with rows padded to 16, as the fp16 model returns
        // them.
        let padded = UnsafeMutablePointer<UInt16>.allocate(capacity: queries * 16)
        padded.initialize(repeating: 0x7e00, count: queries * 16)
        for query in 0..<queries {
            for label in 0..<classes {
                padded[query * 16 + label] = Self.half(logit(query, label))
            }
        }
        let rows = try MLMultiArray(
            dataPointer: padded, shape: [24, 10], dataType: .float16, strides: [16, 1],
            deallocator: { $0.deallocate() })

        // [1, 10, 1, 24] Float32 and Float16, contiguous.
        let columns32 = try MLMultiArray(shape: [1, 10, 1, 24], dataType: .float32)
        let columns16 = try MLMultiArray(shape: [1, 10, 1, 24], dataType: .float16)
        columns32.withUnsafeMutableBytes { raw, strides in
            for query in 0..<queries {
                for label in 0..<classes {
                    raw.storeBytes(
                        of: logit(query, label),
                        toByteOffset: 4 * (label * strides[1] + query * strides[3]),
                        as: Float.self)
                }
            }
        }
        columns16.withUnsafeMutableBytes { raw, strides in
            for query in 0..<queries {
                for label in 0..<classes {
                    raw.storeBytes(
                        of: Self.half(logit(query, label)),
                        toByteOffset: 2 * (label * strides[1] + query * strides[3]),
                        as: UInt16.self)
                }
            }
        }

        let expected = (0..<queries).flatMap { query in
            (0..<classes).map { logit(query, $0) }
        }
        for array in [rows, columns32, columns16] {
            let head = try WindowScores.Head(array, queries: 24, classes: 10, name: "intra")
            XCTAssertEqual(head.values, expected, "\(array.shape) \(array.dataType.rawValue)")
            XCTAssertEqual(head.argmaxes, [Int](repeating: 8, count: 24))
        }
    }

    func testOutputsOfTheWrongShapeOrTypeAreErrors() throws {
        for (shape, type) in [
            ([10, 24], MLMultiArrayDataType.float16), ([1, 24, 1, 10], .float32),
            ([24, 10, 1], .float32), ([24, 10], .double),
        ] {
            let array = try MLMultiArray(
                shape: shape.map { NSNumber(value: $0) }, dataType: type)
            XCTAssertThrowsError(
                try WindowScores.Head(array, queries: 24, classes: 10, name: "intra"),
                "\(shape)")
        }
    }

    /// The Float16 bit pattern of a value exact in Float16.
    private static func half(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        guard value != 0 else { return 0 }
        return UInt16(
            truncatingIfNeeded: (((bits >> 23) &- 112) << 10) | ((bits >> 13) & 0x3ff))
    }
}
