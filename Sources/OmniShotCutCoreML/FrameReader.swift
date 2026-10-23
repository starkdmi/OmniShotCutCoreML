import AVFoundation
import CoreVideo
import Foundation
import Metal

// OmniShotCut's official code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`,
// and the model notices the difference. So the decoder's YUV is taken as it
// is — H.264 and HEVC decoding are bit-exact — and scaled and converted to
// the same bytes as ffmpeg's for 8-bit 4:2:0 input (FrameScaler.swift,
// docs/frame-scaler-spec.md).
// A 10-bit or 4:2:2 source is decoded too, as the MLX package decodes it:
// AVFoundation converts it to 8-bit 4:2:0 itself, where ffmpeg scales it from
// its own depth and subsampling, so its frames differ from ffmpeg's.

/// Every frame of the first video track as the official pipeline sees it:
/// 128×96 RGB, stretched, rotated and mirrored as ffmpeg's autorotation
/// turns it.
///
/// The decoder's planes are copied a batch at a time into shared Metal
/// buffers and scaled and converted on the GPU while the next batch decodes:
/// resizing full-HD planes on the CPU costs four times the decode itself.
final class FrameReader: FrameSource {
    let url: URL
    /// The end of the video track.
    let duration: TimeInterval
    /// The output frame size.
    let width: Int
    let height: Int
    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    /// Quarter turns clockwise to display the stored picture.
    private let quarterTurns: Int
    /// Whether the display matrix also mirrors the picture.
    private let flipped: Bool
    /// The first sample description's YCbCr matrix, used for every frame;
    /// `nil` when every sample description names a matrix, and each frame's
    /// own is used. A track that mixes tagged and untagged sample
    /// descriptions is converted with the first one's matrix throughout:
    /// untagged frames are BT.601 in ffmpeg but carry the decoder's guess.
    private let trackMatrix: YCbCrMatrix?
    /// Whether the track is H.264, whose ffmpeg decoder keeps a full-range
    /// first frame's range for every later frame.
    private let keepsFullRange: Bool
    /// Whether the first decoded frame is full range.
    private var firstFrameFullRange: Bool?
    private let metal: MetalContext
    private var scaler: FrameScaler?
    private var scalerColour: FrameColour?
    /// One batch converts on the GPU from one slot while the next decodes
    /// into the other; frames are read out of a slot's RGB buffer before it
    /// is filled again.
    private let slots = [Slot(), Slot()]
    private var fillSlot = 0
    private var converting: Batch?
    private var ready: Batch?
    private var readyIndex = 0
    private var carried: CMSampleBuffer?
    private var finished = false

    /// Frames scaled together at most, enough to keep the GPU busy.
    private static let batch = 16
    /// Bytes of decoded planes a batch holds at most, in each of two slots:
    /// 16 frames up to 1080p, 5 of 4K.
    private static let planeBudget = 64 << 20

    private struct Batch {
        let slot: Int
        let times: [TimeInterval]
        let commandBuffer: MTLCommandBuffer
    }

    /// The matrix and range a frame is converted with.
    private struct FrameColour: Hashable {
        var matrix: YCbCrMatrix
        var fullRange: Bool
    }

    init(url: URL, width: Int, height: Int) async throws {
        guard let metal = MetalContext.shared else {
            throw OmniShotCutError.metalUnavailable("no Metal device")
        }
        self.metal = metal
        self.url = url
        let asset = AVURLAsset(url: url)
        do {
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw OmniShotCutError.noVideoTrack(url)
            }
            // The asset's duration also covers an audio track that runs past
            // the last frame.
            let trackEnd = CMTimeRangeGetEnd(try await track.load(.timeRange))
            if trackEnd.isNumeric, trackEnd.seconds > 0 {
                duration = trackEnd.seconds
            } else {
                duration = try await asset.load(.duration).seconds
            }
            let transform = try await track.load(.preferredTransform)
            let angle = atan2(transform.b, transform.a)
            quarterTurns = (Int((angle / (Double.pi / 2)).rounded()) % 4 + 4) % 4
            flipped = transform.a * transform.d - transform.b * transform.c < 0

            let descriptions = try await track.load(.formatDescriptions)
            let matrices = descriptions.map {
                CMFormatDescriptionGetExtension(
                    $0, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String
            }
            trackMatrix =
                !matrices.isEmpty && matrices.allSatisfy { $0 != nil }
                ? nil : Self.matrix(matrices.first ?? nil)
            keepsFullRange =
                !descriptions.isEmpty
                && descriptions.allSatisfy {
                    CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_H264
                }

            reader = try AVAssetReader(asset: asset)
            output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    // Both ranges: the decoder returns each frame in its own
                    // range, unconverted.
                    kCVPixelBufferPixelFormatTypeKey as String: [
                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    ]
                ])
        } catch let error as OmniShotCutError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw OmniShotCutError.decodingFailed(url, Self.detail(error))
        }
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw OmniShotCutError.decodingFailed(
                url, reader.error.map(Self.detail) ?? "unreadable video")
        }
        self.width = width
        self.height = height
    }

    deinit {
        reader.cancelReading()
    }

    /// ffmpeg reads the stream's matrix; untagged video is BT.601 there, not
    /// guessed from the resolution.
    private static func matrix(_ tag: String?) -> YCbCrMatrix {
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 as String { return .bt709 }
        if tag == kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995 as String {
            return .smpte240m
        }
        if tag == kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as String { return .bt2020 }
        return .bt601
    }

    /// AVFoundation's description of an error, with its reason when it gives
    /// one.
    private static func detail(_ error: Error) -> String {
        let error = error as NSError
        guard let reason = error.localizedFailureReason else {
            return error.localizedDescription
        }
        return "\(error.localizedDescription): \(reason)"
    }

    /// The matrix and range of a decoded frame. The range is the decoder's
    /// output format's. ffmpeg's H.264 decoder keeps its full-range output
    /// format (yuvj420p) once the first frame is full range, and ffmpeg
    /// converts that format as full range, so in an H.264 track every frame
    /// after a full-range first frame is converted as full range too. Its
    /// HEVC decoder follows each frame's range.
    private func colour(of image: CVPixelBuffer) -> FrameColour {
        let fullRange =
            CVPixelBufferGetPixelFormatType(image)
            == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        if firstFrameFullRange == nil { firstFrameFullRange = fullRange }
        let matrix =
            trackMatrix
            ?? Self.matrix(
                CVBufferCopyAttachment(image, kCVImageBufferYCbCrMatrixKey, nil) as? String)
        return FrameColour(
            matrix: matrix,
            fullRange: fullRange || (keepsFullRange && firstFrameFullRange == true))
    }

    /// RGB bytes and presentation time of the next frame, `nil` at the end.
    func next() throws -> (pixels: [UInt8], time: TimeInterval)? {
        let frameBytes = width * height * 3
        var time: TimeInterval?
        let pixels = try [UInt8](unsafeUninitializedCapacity: frameBytes) { buffer, count in
            time = try nextFrame(into: UnsafeMutableRawPointer(buffer.baseAddress!))
            count = time == nil ? 0 : frameBytes
        }
        return time.map { (pixels, $0) }
    }

    /// Copies the next frame's `height × width × 3` RGB bytes to
    /// `destination` and returns its presentation time, `nil` at the end.
    func nextFrame(into destination: UnsafeMutableRawPointer) throws -> TimeInterval? {
        while readyIndex >= (ready?.times.count ?? 0), !finished || converting != nil {
            let previous = converting
            converting = nil
            ready = nil
            readyIndex = 0
            if !finished { try fill() }
            if let previous { try collect(previous) }
        }
        guard let ready, readyIndex < ready.times.count, let rgb = slots[ready.slot].rgb
        else { return nil }
        let frameBytes = width * height * 3
        destination.copyMemory(
            from: rgb.contents() + readyIndex * frameBytes, byteCount: frameBytes)
        readyIndex += 1
        return ready.times[readyIndex - 1]
    }

    private func collect(_ batch: Batch) throws {
        batch.commandBuffer.waitUntilCompleted()
        guard batch.commandBuffer.status == .completed else {
            let detail =
                batch.commandBuffer.error.map { "\($0)" } ?? "command buffer not completed"
            throw OmniShotCutError.decodingFailed(url, "GPU conversion failed: \(detail)")
        }
        ready = batch
        readyIndex = 0
    }

    /// Decode up to a batch of frames of one size and colour into the free
    /// slot and start converting them.
    private func fill() throws {
        let slot = slots[fillSlot]
        let device = metal.device
        var layout: PlaneLayout?
        var colour: FrameColour?
        var limit = Self.batch
        var times: [TimeInterval] = []
        times.reserveCapacity(Self.batch)
        while times.count < limit {
            guard let sample = carried ?? output.copyNextSampleBuffer() else {
                finished = true
                break
            }
            carried = nil
            guard let image = CMSampleBufferGetImageBuffer(sample) else { continue }
            CVPixelBufferLockBaseAddress(image, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
            guard CVPixelBufferGetPlaneCount(image) == 2,
                let lumaBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
                let chromaBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
            else {
                throw OmniShotCutError.unsupportedPixelFormat(url)
            }
            let frame = PlaneLayout(
                width: CVPixelBufferGetWidthOfPlane(image, 0),
                height: CVPixelBufferGetHeightOfPlane(image, 0),
                chromaWidth: CVPixelBufferGetWidthOfPlane(image, 1),
                chromaHeight: CVPixelBufferGetHeightOfPlane(image, 1))
            let frameColour = self.colour(of: image)
            if let layout, frame != layout || frameColour != colour {
                // A new frame size or colour starts the next batch.
                carried = sample
                break
            }
            if layout == nil {
                layout = frame
                colour = frameColour
                limit = min(
                    Self.batch,
                    max(1, Self.planeBudget / (frame.lumaBytes + frame.chromaBytes)))
                try slot.reserve(\.luma, bytes: limit * frame.lumaBytes, device: device)
                try slot.reserve(\.chroma, bytes: limit * frame.chromaBytes, device: device)
            }
            let index = times.count
            Self.copy(
                lumaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 0),
                rows: frame.height, bytes: frame.width,
                to: slot.luma!.contents() + index * frame.lumaBytes)
            Self.copy(
                chromaBase, rowBytes: CVPixelBufferGetBytesPerRowOfPlane(image, 1),
                rows: frame.chromaHeight, bytes: frame.chromaWidth * 2,
                to: slot.chroma!.contents() + index * frame.chromaBytes)
            times.append(CMSampleBufferGetPresentationTimeStamp(sample).seconds)
        }
        if reader.status == .failed {
            throw OmniShotCutError.decodingFailed(
                url, reader.error.map(Self.detail) ?? "decoding failed")
        }
        guard let layout, let colour, !times.isEmpty else { return }

        if scaler?.layout != layout || scalerColour != colour {
            scaler = try FrameScaler(
                layout: layout, quarterTurns: quarterTurns, flipped: flipped, width: width,
                height: height, matrix: colour.matrix, fullRange: colour.fullRange,
                metal: metal)
            scalerColour = colour
        }
        let scaler = scaler!
        // The intermediate values never leave the GPU.
        try slot.reserve(
            \.scratch, bytes: scaler.scratchBytes(frames: limit), device: device,
            options: .storageModePrivate)
        try slot.reserve(\.rgb, bytes: limit * width * height * 3, device: device)
        guard let commandBuffer = metal.queue.makeCommandBuffer() else {
            throw OmniShotCutError.metalUnavailable("could not create a command buffer")
        }
        try scaler.encode(
            frames: times.count, luma: slot.luma!, chroma: slot.chroma!, scratch: slot.scratch!,
            rgb: slot.rgb!, into: commandBuffer)
        commandBuffer.commit()
        converting = Batch(slot: fillSlot, times: times, commandBuffer: commandBuffer)
        fillSlot ^= 1
    }

    /// Copies a plane without its row padding.
    private static func copy(
        _ base: UnsafeMutableRawPointer, rowBytes: Int, rows: Int, bytes: Int,
        to destination: UnsafeMutableRawPointer
    ) {
        if rowBytes == bytes {
            destination.copyMemory(from: base, byteCount: rows * bytes)
            return
        }
        for row in 0..<rows {
            (destination + row * bytes).copyMemory(
                from: base + row * rowBytes, byteCount: bytes)
        }
    }
}

/// One batch's buffers. The planes and the RGB output are shared with the
/// CPU; each buffer grows to the largest batch it has held and is reused
/// after that.
private final class Slot {
    var luma: MTLBuffer?
    var chroma: MTLBuffer?
    var scratch: MTLBuffer?
    var rgb: MTLBuffer?

    func reserve(
        _ buffer: ReferenceWritableKeyPath<Slot, MTLBuffer?>, bytes: Int, device: MTLDevice,
        options: MTLResourceOptions = .storageModeShared
    ) throws {
        if let existing = self[keyPath: buffer], existing.length >= bytes { return }
        // Released first, so the old and the new buffer are not held at once.
        self[keyPath: buffer] = nil
        guard let made = device.makeBuffer(length: max(bytes, 1), options: options) else {
            throw OmniShotCutError.metalUnavailable("could not allocate \(bytes) bytes")
        }
        self[keyPath: buffer] = made
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
