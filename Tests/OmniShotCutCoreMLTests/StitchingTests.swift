import XCTest

@_spi(Benchmarking) @testable import OmniShotCutCoreML

/// `engine.py`'s windowing and stitching, without the model.
final class StitchingTests: XCTestCase {

    func testWindowSegmentsFollowQueriesUntilTheWindowEnds() {
        let logits = OmniShotCut.WindowLogits(
            intra: [0, 1, 0, 6, 0], inter: [0, 2, 1, 2, 1], end: [30, 30, 60, 101, 5])
        let segments = OmniShotCut.windowSegments(logits: logits, validLength: 100)
        // The second query does not move forward and is skipped; the fourth
        // runs past the frames and is clipped; the fifth is never read.
        XCTAssertEqual(segments.map(\.start), [0, 30, 60])
        XCTAssertEqual(segments.map(\.end), [30, 60, 100])
        XCTAssertEqual(segments.map(\.intra), [0, 0, 6])
    }

    func testValidRangesSplitTheOverlapInHalf() {
        let first = OmniShotCut.validRange(
            windowStart: 0, window: 100, overlap: 10, total: .max)
        let middle = OmniShotCut.validRange(
            windowStart: 90, window: 100, overlap: 10, total: .max)
        let last = OmniShotCut.validRange(
            windowStart: 180, window: 100, overlap: 10, total: 250)
        XCTAssertEqual(first.start, 0)
        XCTAssertEqual(first.end, 95)
        XCTAssertEqual(middle.start, 95)
        XCTAssertEqual(middle.end, 185)
        XCTAssertEqual(last.start, 185)
        XCTAssertEqual(last.end, 250)
    }

    func testCutsSkipTheWindowsOpeningSegment() {
        let segments = [
            OmniShotCut.FrameRange(start: 0, end: 30, intra: 0, inter: 0),
            OmniShotCut.FrameRange(start: 30, end: 60, intra: 1, inter: 2),
            OmniShotCut.FrameRange(start: 60, end: 100, intra: 0, inter: 1),
        ]
        let cuts = OmniShotCut.cuts(
            from: segments, windowStart: 90, validStart: 95, validEnd: 185)
        XCTAssertEqual(cuts.map(\.position), [120, 150])
        XCTAssertEqual(cuts.map(\.intra), [1, 0])
    }

    func testRangesTakeTheLabelsOfTheCutThatStartsThem() {
        typealias Cut = OmniShotCut.Cut
        let ranges = OmniShotCut.assemble(
            cuts: [
                Cut(position: 150, intra: 0, inter: 1), Cut(position: 120, intra: 6, inter: 2),
                Cut(position: 120, intra: 0, inter: 1), Cut(position: 0, intra: 1, inter: 1),
                Cut(position: 300, intra: 1, inter: 1),
            ], frameCount: 200)
        XCTAssertEqual(ranges.map(\.start), [0, 120, 150])
        XCTAssertEqual(ranges.map(\.end), [120, 150, 200])
        XCTAssertEqual(ranges.map(\.intra), [0, 6, 0])
        XCTAssertEqual(ranges.map(\.inter), [0, 2, 1])
    }
}
