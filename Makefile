# Built with xcodebuild, as the MLX package is. SwiftPM works too: the frame
# scaler's Metal kernels are compiled from source at run time, so there is no
# metallib to bundle (`make swift-test`).
DERIVED ?= .derivedData
PRODUCTS = $(DERIVED)/Build/Products/Release
TEST_PRODUCTS = $(DERIVED)/Build/Products/Debug
DESTINATION = 'platform=macOS,arch=arm64'

.PHONY: cli build-tests test swift-cli swift-test videos format

# Xcode's package schemes enable code coverage, which instruments the
# Release binary too unless ENABLE_CODE_COVERAGE is off.
cli:
	xcodebuild build -scheme omnishotcut -configuration Release \
		-destination $(DESTINATION) -derivedDataPath "$(DERIVED)" \
		ENABLE_CODE_COVERAGE=NO -quiet
	@echo "$(PRODUCTS)/omnishotcut"

build-tests:
	xcodebuild build-for-testing -scheme OmniShotCutCoreML-Package \
		-destination $(DESTINATION) -derivedDataPath "$(DERIVED)" \
		ENABLE_CODE_COVERAGE=NO -quiet

# Without the videos the unit tests and, with ffmpeg, FrameReaderTests'
# synthetic clips run; the others skip. ReferenceTests also needs
# OMNISHOTCUT_MODEL, and reads OMNISHOTCUT_COMPUTE_UNITS,
# OMNISHOTCUT_VERIFIER and OMNISHOTCUT_THRESHOLD (README.md).
test: build-tests
	xcrun xctest "$(TEST_PRODUCTS)/OmniShotCutCoreMLTests.xctest"

swift-cli:
	swift build -c release --product omnishotcut
	@echo "$$(swift build -c release --show-bin-path)/omnishotcut"

swift-test:
	swift test

# The Blender open movies the reference was made from, 388 MB.
videos:
	scripts/fetch-videos.sh videos

format:
	xcrun swift-format format --in-place --recursive Sources Tests Package.swift
