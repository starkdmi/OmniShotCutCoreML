import Foundation

/// One stretch of a video between two boundaries: a shot, or the transition
/// between two shots.
///
/// ``kind`` says what the stretch itself is — a shot, or a dissolve, wipe or
/// fade on its way to the next one — and ``boundary`` says how it began. A
/// dissolve's middle frame is two pictures at once and a fade's is half
/// black, which is why a frame picker should take frames from shots only.
public struct ShotSegment: Sendable, Hashable, Codable {

    /// What the segment is. Raw values are OmniShotCut's intra-relation ids.
    public enum Kind: Int, Sendable, Hashable, Codable, CaseIterable {
        case shot = 0
        case dissolve = 1
        case wipe = 2
        case push = 3
        case slide = 4
        case zoom = 5
        /// Fade to or from black or white, including a dip between shots.
        case fade = 6
        case doorway = 7
        /// The model's class for the queries past a window's last segment. The
        /// model sometimes gives it to a real segment, and the official code
        /// keeps it there, as this package does: 2 of the 222 segments of the
        /// three reference videos, the Sintel trailer's last among them.
        case padding = 8

        /// A stretch on its way between pictures. Neither a shot nor
        /// ``padding`` is: padding says nothing about the picture.
        public var isTransition: Bool { self != .shot && self != .padding }

        /// The official label.
        public var name: String {
            [
                "General", "Dissolve", "Wipes", "Push", "Slide", "Zoom", "Fade", "Doorway",
                "Padding",
            ][
                rawValue]
        }
    }

    /// How the segment starts. Raw values are OmniShotCut's inter-relation ids.
    public enum Boundary: Int, Sendable, Hashable, Codable, CaseIterable {
        case videoStart = 0
        case hardCut = 1
        case transitionSource = 2
        case transition = 3
        /// A jump within one camera setup — a cut in an interview.
        case suddenJump = 4
        /// The model's class for the queries past a window's last segment,
        /// kept where it labels a real one, as ``Kind/padding`` is.
        case padding = 5

        /// The official label.
        public var name: String {
            [
                "New_Start", "Hard_Cut", "Transition_Source", "Transition", "Sudden_Jump",
                "Padding",
            ][
                rawValue]
        }
    }

    /// Presentation time of the first frame, in seconds.
    public var start: TimeInterval
    /// Presentation time of the next segment's first frame, or the end of
    /// the video track for the last segment.
    public var end: TimeInterval
    /// Decoded frame indices, `start..<end`, as the official code numbers
    /// them.
    public var frames: Range<Int>
    public var kind: Kind
    public var boundary: Boundary

    public init(
        start: TimeInterval, end: TimeInterval, frames: Range<Int>, kind: Kind,
        boundary: Boundary
    ) {
        self.start = start
        self.end = end
        self.frames = frames
        self.kind = kind
        self.boundary = boundary
    }

    public var duration: TimeInterval { end - start }
}

public enum OmniShotCutError: LocalizedError, Sendable, Equatable, CustomStringConvertible {
    case videoUnavailable(URL)
    case noVideoTrack(URL)
    case noFrames(URL)
    case decodingFailed(URL, String)
    case unsupportedPixelFormat(URL)
    /// The model's path, and why it could not be found, compiled or loaded.
    case modelUnavailable(String, String)
    /// The model failed on a window, or returned outputs of the wrong shape
    /// or logits that are not finite.
    case predictionFailed(String)
    /// No Metal device, or the frame scaler's kernels did not compile.
    case metalUnavailable(String)

    public var description: String {
        switch self {
        case .videoUnavailable(let url): return "\(url.path): no such file"
        case .noVideoTrack(let url): return "\(url.lastPathComponent): no video track"
        case .noFrames(let url): return "\(url.lastPathComponent): no readable frames"
        case .decodingFailed(let url, let detail):
            return "\(url.lastPathComponent): \(detail)"
        case .unsupportedPixelFormat(let url):
            return "\(url.lastPathComponent): decoder did not return 4:2:0 YUV"
        case .modelUnavailable(let model, let detail): return "\(model): \(detail)"
        case .predictionFailed(let detail): return "prediction failed: \(detail)"
        case .metalUnavailable(let detail): return "Metal: \(detail)"
        }
    }

    /// The description, so that `localizedDescription` says what failed.
    public var errorDescription: String? { description }
}
