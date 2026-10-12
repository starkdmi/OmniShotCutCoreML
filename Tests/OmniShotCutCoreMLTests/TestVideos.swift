import Foundation

/// The reference videos: `OMNISHOTCUT_VIDEOS`, or `videos/` in the package.
enum TestVideos {
    static var directory: URL {
        if let path = ProcessInfo.processInfo.environment["OMNISHOTCUT_VIDEOS"], !path.isEmpty {
            return URL(
                fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("videos", isDirectory: true)
    }
}
