import XCTest

@testable import OmniShotCutCoreML

/// Finding a model, naming its compiled copy and removing stale copies,
/// without compiling.
final class ModelFilesTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "omnishotcut-models-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// A directory named like a model, holding one weight file.
    @discardableResult
    private func make(_ path: String, bytes: Int = 4) throws -> URL {
        let url = directory.appendingPathComponent(path, isDirectory: true)
        try FileManager.default.createDirectory(
            at: url.appendingPathComponent("Data"), withIntermediateDirectories: true)
        try Data(repeating: 1, count: bytes).write(
            to: url.appendingPathComponent("Data/weight.bin"))
        return url
    }

    func testAModelIsUsedAsItIs() throws {
        let package = try make("a.mlpackage")
        let compiled = try make("b.mlmodelc")
        XCTAssertEqual(try ModelFiles.locate(package), package)
        XCTAssertEqual(try ModelFiles.locate(compiled), compiled)
    }

    func testADirectoryHoldingOneModel() throws {
        let holder = directory.appendingPathComponent("holder", isDirectory: true)
        let package = try make("holder/OmniShotCut.mlpackage")
        XCTAssertEqual(
            try ModelFiles.locate(holder).lastPathComponent, package.lastPathComponent)
        // Its compiled form is preferred.
        try make("holder/OmniShotCut.mlmodelc")
        XCTAssertEqual(try ModelFiles.locate(holder).lastPathComponent, "OmniShotCut.mlmodelc")
    }

    func testAmbiguousOrMissingModelsAreErrors() throws {
        try make("two/fp16.mlpackage")
        try make("two/fp32.mlpackage")
        try make("none/notes")
        for path in ["two", "none", "missing"] {
            XCTAssertThrowsError(
                try ModelFiles.locate(directory.appendingPathComponent(path)), path)
        }
    }

    func testTheFingerprintFollowsThePackagesFiles() throws {
        let package = try make("a.mlpackage")
        let manifest = package.appendingPathComponent("Manifest.json")
        try Data("{}".utf8).write(to: manifest)
        let first = try ModelFiles.fingerprint(of: package)
        XCTAssertEqual(first.count, 16)
        XCTAssertEqual(try ModelFiles.fingerprint(of: package), first)
        // coremltools rewrites Manifest.json with the same bytes on every
        // open: a new modification date alone leaves the fingerprint as it is.
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSinceNow: 3_600)], ofItemAtPath: manifest.path)
        XCTAssertEqual(try ModelFiles.fingerprint(of: package), first)
        try Data("{ }".utf8).write(to: manifest)
        let edited = try ModelFiles.fingerprint(of: package)
        XCTAssertNotEqual(edited, first)
        try Data(repeating: 2, count: 5).write(
            to: package.appendingPathComponent("Data/weight.bin"))
        XCTAssertNotEqual(try ModelFiles.fingerprint(of: package), edited)
    }

    /// A package reached through a link, and one whose files are links into
    /// a blob store, are fingerprinted by the files the links name.
    func testTheFingerprintFollowsSymbolicLinks() throws {
        let package = try make("a.mlpackage")
        let link = directory.appendingPathComponent("link.mlpackage")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: package)
        let direct = try ModelFiles.fingerprint(of: package)
        XCTAssertEqual(try ModelFiles.fingerprint(of: link), direct)
        try Data(repeating: 2, count: 4).write(
            to: package.appendingPathComponent("Data/weight.bin"))
        XCTAssertNotEqual(try ModelFiles.fingerprint(of: link), direct)

        let blobs = directory.appendingPathComponent("blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let blob = blobs.appendingPathComponent("0123")
        try Data(repeating: 1, count: 4).write(to: blob)
        let snapshot = directory.appendingPathComponent("snapshot/a.mlpackage/Data")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: snapshot.appendingPathComponent("weight.bin"), withDestinationURL: blob)
        let linked = snapshot.deletingLastPathComponent()
        XCTAssertEqual(
            try ModelFiles.fingerprint(of: linked),
            try ModelFiles.fingerprint(of: make("b.mlpackage")))
        let before = try ModelFiles.fingerprint(of: linked)
        try Data(repeating: 3, count: 4).write(to: blob)
        XCTAssertNotEqual(try ModelFiles.fingerprint(of: linked), before)
    }

    func testTheOriginFollowsThePackagesPath() throws {
        let package = try make("one/OmniShotCut.mlpackage")
        let other = try make("two/OmniShotCut.mlpackage")
        XCTAssertEqual(ModelFiles.origin(of: package).count, 8)
        XCTAssertEqual(ModelFiles.origin(of: package), ModelFiles.origin(of: package))
        XCTAssertNotEqual(ModelFiles.origin(of: package), ModelFiles.origin(of: other))
        let link = directory.appendingPathComponent("link.mlpackage")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: package)
        XCTAssertEqual(ModelFiles.origin(of: link), ModelFiles.origin(of: package))
    }

    func testCompiledCopiesAreRecognizedByName() {
        let name = "OmniShotCut-v1.5-fp16"
        let origin = "0123abcd"
        XCTAssertTrue(
            ModelFiles.isCompiledCopy(
                "\(name)-\(origin)-0123456789abcdef.mlmodelc", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-89abcdef-0123456789abcdef.mlmodelc", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-0123456789abcdef.mlmodelc", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-\(origin)-0123456789abcdef.mlpackage", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-\(origin)-0123.mlmodelc", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "OmniShotCut-v1.5-fp32-\(origin)-0123456789abcdef.mlmodelc", of: name,
                origin: origin))
        // Any origin.
        XCTAssertTrue(
            ModelFiles.isCompiledCopy(
                "\(name)-89abcdef-0123456789abcdef.mlmodelc", of: name, origin: nil))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-89abcdeg-0123456789abcdef.mlmodelc", of: name, origin: nil))
        XCTAssertFalse(
            ModelFiles.isCompiledCopy(
                "\(name)-\(origin)-0123456789abcdef.mlmodelc", of: "OmniShotCut-v1.5",
                origin: nil))
        let uuid = "6F9619FF-8B86-D011-B42D-00C04FC964FF"
        XCTAssertTrue(
            ModelFiles.isStagingCopy(
                "\(name)-\(origin)-0123456789abcdef.\(uuid).partial", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isStagingCopy(
                "\(name)-89abcdef-0123456789abcdef.\(uuid).partial", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isStagingCopy(
                "\(name)-\(origin)-0123456789abcdef.partial", of: name, origin: origin))
        XCTAssertFalse(
            ModelFiles.isStagingCopy(
                "\(name)-\(origin)-0123456789abcdef.\(uuid).mlmodelc", of: name, origin: origin)
        )
    }

    /// Copies of earlier versions of the package at the same path, copies
    /// whose package is gone from the path they record, and abandoned
    /// compiles are deleted, other than the copy in use, and only once they
    /// are older than the age limit. A copy whose package is still where it
    /// says, or that says nothing, is kept.
    func testStaleCopiesAreRemovedByOriginAndAge() throws {
        let name = "OmniShotCut-v1.5-fp16"
        let now = Date()
        let present = try make("present.mlpackage")
        let gone = directory.appendingPathComponent("gone.mlpackage").path
        func copy(_ entry: String, age: TimeInterval, source: String? = nil) throws -> URL {
            let url = try make(entry)
            if let source {
                try Data(source.utf8).write(
                    to: url.appendingPathComponent(ModelFiles.sourceFileName))
            }
            try FileManager.default.setAttributes(
                [.modificationDate: now.addingTimeInterval(-age)], ofItemAtPath: url.path)
            return url
        }
        let current = try copy("\(name)-0123abcd-0000000000000001.mlmodelc", age: 7_200)
        let stale = try copy("\(name)-0123abcd-0000000000000002.mlmodelc", age: 7_200)
        let recent = try copy("\(name)-0123abcd-0000000000000003.mlmodelc", age: 60)
        let elsewhere = try copy("\(name)-89abcdef-0000000000000004.mlmodelc", age: 7_200)
        let uuid = UUID().uuidString
        let abandoned = try copy(
            "\(name)-0123abcd-0000000000000005.\(uuid).partial", age: 7_200)
        let staging = try copy("\(name)-0123abcd-0000000000000006.\(uuid).partial", age: 60)
        let abandonedElsewhere = try copy(
            "\(name)-89abcdef-0000000000000007.\(uuid).partial", age: 7_200)
        let moved = try copy(
            "\(name)-456789ab-0000000000000008.mlmodelc", age: 7_200, source: gone)
        let movedRecently = try copy(
            "\(name)-456789ab-0000000000000009.mlmodelc", age: 60, source: gone)
        let inUseElsewhere = try copy(
            "\(name)-cdef0123-000000000000000a.mlmodelc", age: 7_200, source: present.path)
        let otherModel = try copy(
            "OmniShotCut-v1.5-fp32-456789ab-000000000000000b.mlmodelc", age: 7_200,
            source: gone)
        ModelFiles.removeStaleCopies(
            in: directory, name: name, origin: "0123abcd", keeping: current.lastPathComponent,
            now: now)
        func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }
        XCTAssertTrue(exists(current))
        XCTAssertFalse(exists(stale))
        XCTAssertTrue(exists(recent))
        XCTAssertTrue(exists(elsewhere))
        XCTAssertFalse(exists(abandoned))
        XCTAssertTrue(exists(staging))
        XCTAssertFalse(exists(abandonedElsewhere))
        XCTAssertFalse(exists(moved))
        XCTAssertTrue(exists(movedRecently))
        XCTAssertTrue(exists(inUseElsewhere))
        // Another model's copies are left to its own loads.
        XCTAssertTrue(exists(otherModel))
    }
}
