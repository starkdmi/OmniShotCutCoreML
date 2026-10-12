import CoreML
import CryptoKit
import Foundation

/// Where a model's compiled `.mlmodelc` is.
///
/// The ANE compiles a model on its first load and caches the result per
/// compiled model location; a later load from the same location takes
/// milliseconds instead of seconds. So a package is compiled once into
/// `Caches/OmniShotCutCoreML/<name>-<origin>-<fingerprint>.mlmodelc` and every
/// later load uses that copy. The origin is a hash of the package's path, so
/// same-named packages in different directories keep separate copies. The
/// fingerprint covers the contents of the package's files up to 4 MiB, the
/// sizes and modification dates of larger ones, and the OS version, with
/// symbolic links followed. A changed package is compiled again. On every
/// load, the copies compiled earlier from the same path, the copies whose
/// package is gone from the path they were compiled from — a moved folder, an
/// app reinstalled at a new path, a pruned Hugging Face snapshot — and staging
/// directories a compile left behind are deleted once they have been neither
/// compiled nor used for ten minutes, so a copy another process has just
/// compiled or found is not deleted before that process loads it.
enum ModelFiles {

    /// The file in a compiled copy that holds the path of the package it was
    /// compiled from. Without it, a copy whose package moved would be kept
    /// forever: the origin is only a hash of the path.
    static let sourceFileName = "omnishotcut-source"

    /// How long a stale compiled copy — of an earlier version of a package, or
    /// of one that is gone — is kept after it was last compiled or used.
    static let staleAge: TimeInterval = 600

    /// Files up to this size enter the fingerprint by their contents, larger
    /// ones by size and modification date.
    static let hashedFileSize = 4 << 20

    /// The model `url` names: an `.mlmodelc` or `.mlpackage` (or `.mlmodel`)
    /// itself, or the one in a directory, its `.mlmodelc` preferred.
    static func locate(_ url: URL) throws -> URL {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw OmniShotCutError.modelUnavailable(url.path, "no such file")
        }
        if ["mlmodelc", "mlpackage", "mlmodel"].contains(url.pathExtension.lowercased()) {
            return url
        }
        guard isDirectory.boolValue else {
            throw OmniShotCutError.modelUnavailable(
                url.path, "not an .mlpackage, an .mlmodelc or a directory holding one")
        }
        let contents = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }
        let compiled = contents.filter { $0.pathExtension.lowercased() == "mlmodelc" }
        let packages = contents.filter {
            ["mlpackage", "mlmodel"].contains($0.pathExtension.lowercased())
        }
        let candidates = compiled.isEmpty ? packages : compiled
        guard candidates.count == 1 else {
            let detail =
                candidates.isEmpty
                ? "holds no .mlmodelc or .mlpackage"
                : "holds several models: "
                    + candidates.map(\.lastPathComponent).joined(separator: ", ")
            throw OmniShotCutError.modelUnavailable(url.path, detail)
        }
        return candidates[0]
    }

    /// The compiled model for `url`, as ``locate(_:)`` finds it: an
    /// `.mlmodelc` in place, a package compiled into the cache, or the copy
    /// compiled there before.
    static func compiled(_ url: URL) async throws -> (url: URL, compiling: Duration) {
        let model = try locate(url)
        if model.pathExtension.lowercased() == "mlmodelc" { return (model, .zero) }

        let name = model.deletingPathExtension().lastPathComponent
        let origin = Self.origin(of: model)
        let fingerprint = try Self.fingerprint(of: model)
        let cache = try cacheDirectory()
        let destination = cache.appendingPathComponent(
            "\(name)-\(origin)-\(fingerprint).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            // Its modification date is its last use, which keeps it from
            // being deleted as stale.
            try? FileManager.default.setAttributes(
                [.modificationDate: Date()], ofItemAtPath: destination.path)
            removeStaleCopies(
                in: cache, name: name, origin: origin, keeping: destination.lastPathComponent)
            return (destination, .zero)
        }

        let clock = ContinuousClock()
        let started = clock.now
        let temporary: URL
        do {
            temporary = try await MLModel.compileModel(at: model)
        } catch {
            throw OmniShotCutError.modelUnavailable(model.path, "\(error)")
        }
        let compiling = clock.now - started

        // Moved in under a unique name first, then renamed: another process
        // never sees a partial copy, and the first one renamed wins.
        let staging = cache.appendingPathComponent(
            "\(name)-\(origin)-\(fingerprint).\(UUID().uuidString).partial", isDirectory: true)
        do {
            try FileManager.default.moveItem(at: temporary, to: staging)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw OmniShotCutError.modelUnavailable(model.path, "\(error)")
        }
        // Without it the copy is still used, only never cleaned up once its
        // package is gone.
        try? Data(sourcePath(of: model).utf8).write(
            to: staging.appendingPathComponent(sourceFileName))
        do {
            try FileManager.default.moveItem(at: staging, to: destination)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            guard FileManager.default.fileExists(atPath: destination.path) else {
                throw OmniShotCutError.modelUnavailable(model.path, "\(error)")
            }
        }
        try? FileManager.default.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: destination.path)
        removeStaleCopies(
            in: cache, name: name, origin: origin, keeping: destination.lastPathComponent)
        return (destination, compiling)
    }

    /// Deletes, among the copies in `cache` of packages named `name`, except
    /// `keeping`, those compiled from earlier versions of the package at
    /// `origin`, those whose package is gone from the path they were compiled
    /// from, and the staging directories compiles left behind — once last
    /// compiled, used or changed more than `age` seconds before `now`. A copy
    /// from another path that does not say where its package was is kept.
    static func removeStaleCopies(
        in cache: URL, name: String, origin: String, keeping: String,
        olderThan age: TimeInterval = staleAge, now: Date = Date()
    ) {
        for entry in (try? FileManager.default.contentsOfDirectory(atPath: cache.path)) ?? []
        where entry != keeping {
            let copy = cache.appendingPathComponent(entry, isDirectory: true)
            let stale =
                // An abandoned compile, or an earlier version of this package.
                isStagingCopy(entry, of: name, origin: nil)
                || isCompiledCopy(entry, of: name, origin: origin)
                || (isCompiledCopy(entry, of: name, origin: nil) && packageIsGone(copy))
            guard stale,
                let modified = try? copy.resourceValues(forKeys: [.contentModificationDateKey])
                    .contentModificationDate,
                now.timeIntervalSince(modified) > age
            else { continue }
            try? FileManager.default.removeItem(at: copy)
        }
    }

    /// Whether the package `copy` was compiled from is gone from the path the
    /// copy records; false for a copy that records none.
    static func packageIsGone(_ copy: URL) -> Bool {
        guard
            let source = try? String(
                contentsOf: copy.appendingPathComponent(sourceFileName), encoding: .utf8)
        else { return false }
        return !FileManager.default.fileExists(atPath: source)
    }

    /// `Caches/OmniShotCutCoreML`, created if missing.
    static func cacheDirectory() throws -> URL {
        let caches = try FileManager.default.url(
            for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let directory = caches.appendingPathComponent("OmniShotCutCoreML", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// 8 hex digits of SHA-256 over the package's path, symbolic links
    /// resolved.
    static func origin(of model: URL) -> String {
        SHA256.hash(data: Data(sourcePath(of: model).utf8)).prefix(4)
            .map { String(format: "%02x", $0) }.joined()
    }

    /// The package's path, symbolic links resolved: what the origin hashes
    /// and a compiled copy records.
    static func sourcePath(of model: URL) -> String {
        model.resolvingSymlinksInPath().standardizedFileURL.path
    }

    /// 16 hex digits of SHA-256 over the package's relative file paths and
    /// sizes, the contents of files up to ``hashedFileSize``, the modification
    /// times of larger ones, and the OS version. coremltools rewrites
    /// `Manifest.json` with the same bytes whenever it opens a package, so
    /// the small files' modification times are left out. A package that is a
    /// symbolic link, or whose files are (the Hugging Face cache's layout),
    /// is described by the files the links name.
    static func fingerprint(of model: URL) throws -> String {
        let keys: Set<URLResourceKey> = [
            .isRegularFileKey, .fileSizeKey, .contentModificationDateKey,
        ]
        var entries: [String] = []
        let rootURL = model.resolvingSymlinksInPath().standardizedFileURL
        let root = rootURL.path
        func describe(_ file: URL, as relative: String) throws {
            let target = file.resolvingSymlinksInPath()
            let values = try target.resourceValues(forKeys: keys)
            guard values.isRegularFile == true else { return }
            let size = values.fileSize ?? 0
            if size <= hashedFileSize {
                let digest = SHA256.hash(data: try Data(contentsOf: target))
                    .map { String(format: "%02x", $0) }.joined()
                entries.append("\(relative)\t\(size)\t\(digest)")
            } else {
                let modified = values.contentModificationDate?.timeIntervalSince1970 ?? 0
                entries.append("\(relative)\t\(size)\t\(modified)")
            }
        }
        if let enumerator = FileManager.default.enumerator(
            at: rootURL, includingPropertiesForKeys: Array(keys))
        {
            for case let file as URL in enumerator {
                // The file's own link is kept: its path inside the package is
                // what the fingerprint records.
                let directory = file.deletingLastPathComponent()
                    .resolvingSymlinksInPath().standardizedFileURL.path
                let path = directory + "/" + file.lastPathComponent
                try describe(file, as: String(path.dropFirst(root.count)))
            }
        }
        // A single-file .mlmodel enumerates nothing.
        try describe(model, as: "")
        var hash = SHA256()
        for entry in entries.sorted() + [ProcessInfo.processInfo.operatingSystemVersionString] {
            hash.update(data: Data((entry + "\n").utf8))
        }
        return hash.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// Whether `entry` is `<name>-<origin>-<16 hex digits>.mlmodelc`, with
    /// any 8-hex-digit origin when `origin` is nil.
    static func isCompiledCopy(_ entry: String, of name: String, origin: String?) -> Bool {
        let prefix = "\(name)-"
        let suffix = ".mlmodelc"
        guard entry.hasPrefix(prefix), entry.hasSuffix(suffix) else { return false }
        let parts = entry.dropFirst(prefix.count).dropLast(suffix.count)
            .split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[0].count == 8, parts[1].count == 16,
            parts.allSatisfy({ $0.allSatisfy(\.isHexDigit) })
        else { return false }
        return origin.map { parts[0] == $0 } ?? true
    }

    /// Whether `entry` is `<name>-<origin>-<16 hex digits>.<UUID>.partial`,
    /// the name a compiled copy is staged under, with any origin when
    /// `origin` is nil.
    static func isStagingCopy(_ entry: String, of name: String, origin: String?) -> Bool {
        let suffix = ".partial"
        guard entry.hasSuffix(suffix) else { return false }
        let staged = entry.dropLast(suffix.count)
        guard let dot = staged.lastIndex(of: "."),
            UUID(uuidString: String(staged[staged.index(after: dot)...])) != nil
        else { return false }
        return isCompiledCopy(String(staged[..<dot]) + ".mlmodelc", of: name, origin: origin)
    }
}
