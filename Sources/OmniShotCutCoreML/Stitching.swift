import Foundation

// The official `engine.py`'s windowing and stitching, step for step. The
// logic is the MLX port's, unchanged; `StitchingTests` covers it.

extension OmniShotCut {

    /// One window's per-query argmaxes: `intra` and `inter` are class ids,
    /// `end` the frame each query's segment ends at within the window.
    @_spi(Benchmarking) public struct WindowLogits: Sendable, Hashable {
        public var intra: [Int]
        public var inter: [Int]
        public var end: [Int]

        public init(intra: [Int], inter: [Int], end: [Int]) {
            self.intra = intra
            self.inter = inter
            self.end = end
        }
    }

    struct Cut: Sendable, Hashable {
        var position: Int
        var intra: Int
        var inter: Int
    }

    struct FrameRange: Sendable, Hashable {
        var start: Int
        var end: Int
        var intra: Int
        var inter: Int
    }

    /// `split_videos`: the global frame positions `(start, end]` a window's
    /// cuts are trusted in. `total` is `Int.max` until the window that
    /// reaches the end of the video.
    static func validRange(windowStart: Int, window: Int, overlap: Int, total: Int)
        -> (start: Int, end: Int)
    {
        let left = overlap / 2
        let right = overlap - left
        let windowEnd = windowStart + window
        let start = windowStart == 0 ? 0 : windowStart + left
        let end = windowEnd >= total ? total : windowEnd - right
        return (start, end)
    }

    /// `decode_window_segments`: queries in order, each ending where it says
    /// and starting where the previous one ended; a query that does not move
    /// forward is skipped, and the window stops at its last real frame.
    static func windowSegments(logits: WindowLogits, validLength: Int) -> [FrameRange] {
        var segments: [FrameRange] = []
        var start = 0
        for query in logits.end.indices {
            let end = min(logits.end[query], validLength)
            if start >= end { continue }
            segments.append(
                FrameRange(
                    start: start, end: end, intra: logits.intra[query],
                    inter: logits.inter[query]))
            start = end
            if end >= validLength { break }
        }
        return segments
    }

    /// `collect_cuts_from_window`: every segment start but the window's first
    /// becomes a cut, if it falls in `(validStart, validEnd]`.
    static func cuts(
        from segments: [FrameRange], windowStart: Int, validStart: Int, validEnd: Int
    ) -> [Cut] {
        segments.dropFirst().compactMap { segment in
            let position = windowStart + segment.start
            guard validStart < position, position <= validEnd else { return nil }
            return Cut(position: position, intra: segment.intra, inter: segment.inter)
        }
    }

    /// `assemble_ranges`: sorted, de-duplicated cuts partition the video; the
    /// first range is the opening shot and every later one takes the labels
    /// of the cut that starts it.
    static func assemble(cuts: [Cut], frameCount: Int) -> [FrameRange] {
        var kept: [Cut] = []
        for cut in cuts.sorted(by: { $0.position < $1.position }) {
            if let last = kept.last, cut.position == last.position { continue }
            if cut.position <= 0 || cut.position >= frameCount { continue }
            kept.append(cut)
        }
        let opening = (
            intra: ShotSegment.Kind.shot.rawValue,
            inter: ShotSegment.Boundary.videoStart.rawValue
        )
        var ranges: [FrameRange] = []
        var previous = 0
        for (index, cut) in kept.enumerated() where cut.position > previous {
            let labels = index == 0 ? opening : (kept[index - 1].intra, kept[index - 1].inter)
            ranges.append(
                FrameRange(start: previous, end: cut.position, intra: labels.0, inter: labels.1)
            )
            previous = cut.position
        }
        if previous < frameCount {
            let labels = kept.last.map { ($0.intra, $0.inter) } ?? opening
            ranges.append(
                FrameRange(start: previous, end: frameCount, intra: labels.0, inter: labels.1))
        }
        return ranges
    }
}
