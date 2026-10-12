// swift-tools-version: 6.0

import PackageDescription

// No dependencies. The frame scaler's Metal kernels are compiled from source
// at run time, so `swift build` and `swift test` work as well as `xcodebuild`.

let package = Package(
    name: "OmniShotCutCoreML",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "OmniShotCutCoreML", targets: ["OmniShotCutCoreML"]),
        .executable(name: "omnishotcut", targets: ["OmniShotCutCoreMLCLI"]),
    ],
    targets: [
        .target(
            name: "OmniShotCutCoreML",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "OmniShotCutCoreMLCLI",
            dependencies: ["OmniShotCutCoreML"],
            path: "Sources/omnishotcut",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "OmniShotCutCoreMLTests",
            dependencies: ["OmniShotCutCoreML"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
