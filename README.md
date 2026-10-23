# OmniShotCutCoreML

[OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut) v1.5 shot
boundary detection in Swift on Core ML, the counterpart of
[OmniShotCutMLX](https://github.com/starkdmi/OmniShotCutMLX). It splits a
video into shots and the transitions between them — dissolves, wipes, pushes,
slides, zooms, fades, doorways — and says how each one began: a hard cut, a
transition, or a sudden jump within one camera setup.

Two conversions of the official model: fp16 for the Neural Engine and fp32 for
the GPU. Exact mode combines them and reproduces the official output.

Requires macOS 14 or iOS 17 and Swift 6.0 or later; built and tested with
Xcode 27.0 and Swift 6.4. The models themselves load on macOS 13 and iOS 16.

```swift
.package(url: "https://github.com/starkdmi/OmniShotCutCoreML", from: "0.1.0")
```

## Models

| Model | Built for | Compute units |
|---|---|---|
| [`starkdmi/OmniShotCut-v1.5-coreml-fp16`](https://huggingface.co/starkdmi/OmniShotCut-v1.5-coreml-fp16) | fp16, laid out for the Neural Engine | `.cpuAndNeuralEngine` |
| [`starkdmi/OmniShotCut-v1.5-coreml-fp32`](https://huggingface.co/starkdmi/OmniShotCut-v1.5-coreml-fp32) | fp32 | `.cpuAndGPU` |

```bash
hf download starkdmi/OmniShotCut-v1.5-coreml-fp16 --local-dir OmniShotCut-v1.5-coreml-fp16
hf download starkdmi/OmniShotCut-v1.5-coreml-fp32 --local-dir OmniShotCut-v1.5-coreml-fp32
```

`scripts/convert.py` converts the official checkpoint,
[`uva-cv-lab/OmniShotCut_v1.5`](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5),
into both:

```bash
pip install -r scripts/requirements.txt
curl -LO https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/fd050da5732d1fc5d5baabff6c64abd9de1f94aa/OmniShotCut_ckpt.pth
python scripts/convert.py OmniShotCut_ckpt.pth models
```

coremltools 9.0 supports torch up to 2.7.0; the published models were
converted with torch 2.7.0, numpy 2.5.4 and Python 3.12, the versions
`scripts/requirements.txt` pins.

## Use

```swift
import OmniShotCutCoreML

// The fp16 conversion on the Neural Engine.
let detector = try await OmniShotCut(model: fp16URL)  // .mlpackage, .mlmodelc or a directory
for segment in try await detector.segments(in: videoURL) {
    print(segment.start, segment.end, segment.frames, segment.kind, segment.boundary)
}

// Exact mode: the fp32 conversion on the GPU runs again the near-tie windows.
let exact = try await OmniShotCut(model: fp16URL, verifier: fp32URL)

// The fp32 conversion alone.
let gpu = try await OmniShotCut(model: fp32URL, computeUnits: .cpuAndGPU)
```

Segments cover the whole video in order, without gaps. `start` and `end` are
presentation times, `frames` the decoded frame indices as the official code
numbers them; the last segment ends at the end of the video track. Take
keyframes from `kind == .shot`: a dissolve's middle frame is two pictures at
once, a fade's is half black. The model's padding labels, `.padding`, can
reach a real segment, as they do in the official output.

`segments(in:)` decodes and predicts on the detector's own dispatch queue, one
video at a time per detector, so a long video does not hold a thread of Swift's
cooperative pool. Cancelling the calling task stops it at the next frame with
`CancellationError`; a call still waiting for another video throws when its
turn comes, before decoding a frame.

Every frame is converted on the GPU, and iOS does not let an app use the GPU
in the background: on iOS, segment videos while the app is in the foreground.
In the background `segments(in:)` fails with `OmniShotCutError.decodingFailed`.

`fastPrediction` (on by default) loads with
`MLOptimizationHints.specializationStrategy` `.fastPrediction` on macOS 15 and
iOS 18 and later: faster predictions on the Neural Engine, 20.3 ms per window
against 25.4 on an M4, after a slower first load, 7.7 s against 3.9.

A model's first load on the Neural Engine compiles it for the ANE; Core ML
caches the result per process name and compiled model location, so later loads
take milliseconds until the system purges that cache. Load the model at launch,
not per video. An `.mlpackage` is compiled once into
`Caches/OmniShotCutCoreML` and the compiled copy reused while the package is
unchanged; an `.mlmodelc` is used in place.

The command line tool prints the same:

```bash
make cli
.derivedData/Build/Products/Release/omnishotcut --model OmniShotCut-v1.5-coreml-fp16 video.mp4
.derivedData/Build/Products/Release/omnishotcut --model OmniShotCut-v1.5-coreml-fp16 \
    --verifier OmniShotCut-v1.5-coreml-fp32 video.mp4
```

| Option | |
|---|---|
| `--model <path>` | the model run on every window (required) |
| `--compute-units ane\|gpu\|cpu\|all` | its compute units: `.cpuAndNeuralEngine` (default), `.cpuAndGPU`, `.cpuOnly`, `.all` |
| `--verifier <path>` | exact mode, with this model on `.cpuAndGPU` as the verifier |
| `--threshold <t>` | exact mode's near-tie margin in logits, default 1.0 |
| `--no-fast-prediction` | load without the `.fastPrediction` specialization |
| `--json` | JSON instead of text |

## Results

Mac mini M4, 16 GB, macOS 26.6.2, on the three Blender open movies of the
tests, against OmniShotCutMLX 0.1.0 on the same machine:

| | Cuts identical to PyTorch | ms / window | Tears of Steel, 12 min | Memory |
|---|---|---|---|---|
| fp16, Neural Engine | 218 of 219 | 20.3 | 4.5 s | 167 MB |
| Exact mode | 219 of 219 | 26.6 | 5.9 s | 620–640 MB |
| fp32, GPU | 219 of 219 | 55.3–55.4 | 12.1–13.2 s | 481–498 MB |
| fp16, GPU | 217 of 219 | 49.4 | | 260 MB |
| OmniShotCutMLX, fp16 weights (its default) | 218 of 219 | 72.4 | 16.9–17.1 s | 1338 MB |
| OmniShotCutMLX, fp32 weights | 219 of 219 | 72.1–72.7 | 16.9–17.0 s | 1246 MB |

Per window is the median of 20 windows of Tears of Steel (for exact mode, the
mean over its 195 full windows), from the window's frames to its decisions,
decoding excluded; Tears of Steel is the command line
tool's wall time for the whole film, decoding included. Memory is the peak
footprint of the process, plus the Neural Engine program's 128 MB where it
runs, in MB of 2^20 bytes.

On the Neural Engine, the fp16 model's differences from PyTorch are near-ties:
decisions PyTorch in fp64 takes by less than 0.05. fp16 arithmetic is not the same on every chip,
so on another Mac it can decide a near-tie or two the other way; exact mode
runs those windows again in fp32.

- **fp16 on the Neural Engine**, the default, for speed and memory.
- **Exact mode** for the official output, at 31% more time per window, with
  both models loaded.
- **fp32 on the GPU** where no Neural Engine is available.

## Exact mode

In exact mode the fp16 model runs every window, and a window is run again on
the verifier, whose logits are then used, when a decision its segments depend
on has a margin below `threshold`. The decisions are those the official
`decode_window_segments` reads: every query's end, up to the query that
reaches the window's last frame, and the intra and inter labels of the
queries that make a segment. A decision comes out the same in both models when
their logits differ by less than half its margin, so as long as the fp16
model's logits stay within `threshold / 2` of the fp32 model's, every window's
segments are the verifier's.

The default `threshold` is 1.0. On the test videos the fp16 model's
decision-relevant logits differ from the fp32 model's by at most 0.21. At 1.0
all 219 cuts match PyTorch's, and the verifier runs on one window in six.

## How it matches the official code

- **Every frame, at the file's own rate**, as the official code reads them.
  Frames stream through one 100-frame window at a time, so an hour of video
  costs one window of memory.
- **Windowing and stitching are `engine.py`'s**, step for step: windows of 100
  frames overlapping by 10, each trusted for its half of the overlap, every
  segment labelled by the cut that begins it. The model runs on one window
  while the next window's frames decode.
- **Decoding is ffmpeg's, bit for bit, for 8-bit 4:2:0 video.** The official
  code decodes with `ffmpeg -s 128x96 -pix_fmt rgb24`. `FrameReader` takes the
  decoder's YUV from AVFoundation (H.264 and HEVC decoding are bit-exact) and
  scales and converts it in two Metal compute kernels to the same bytes as
  ffmpeg's scaler — fixed-point bicubic filters, horizontal then vertical,
  half-width chroma and integer YUV→RGB, in each frame's matrix and range, as
  [docs/frame-scaler-spec.md](docs/frame-scaler-spec.md) specifies — and turns
  and mirrors video with a display matrix as ffmpeg's autorotation does.
- **Other sources are close, not exact.** AVFoundation converts a 10-bit or
  4:2:2 source — iPhone HDR video, ProRes — to 8-bit 4:2:0 itself, where ffmpeg
  scales it from its own depth and subsampling, so those frames differ from
  ffmpeg's: on the test clips, 8–16% of values, by up to 72 levels. Exact mode's equivalence to the official result
  holds for 8-bit 4:2:0 H.264 and HEVC.
- **The model contract.** Each model reads `frames`, Float16
  `[100, 3, 96, 128]`, RGB 0–255, black past the end of the video, and returns
  the `intra` (10 classes), `inter` (7) and `end` (102) logits of 24 queries as
  `[24, C]` or `[1, C, 1, 24]`. Each query's prediction is the argmax over
  every class but the last, as `engine.py` takes it.

## Build

```bash
make cli          # xcodebuild, Release: .derivedData/Build/Products/Release/omnishotcut
make swift-cli    # swift build -c release
```

The Metal kernels are compiled from source when the first frame reader starts,
so SwiftPM builds work as well as Xcode's: `swift build`, `swift run` and
`swift test`.

## Testing

```bash
make videos       # the test videos, 388 MB, into videos/
make test         # or swift test

OMNISHOTCUT_MODEL=OmniShotCut-v1.5-coreml-fp16 make test
OMNISHOTCUT_MODEL=OmniShotCut-v1.5-coreml-fp32 OMNISHOTCUT_COMPUTE_UNITS=gpu make test
OMNISHOTCUT_MODEL=OmniShotCut-v1.5-coreml-fp16 \
OMNISHOTCUT_VERIFIER=OmniShotCut-v1.5-coreml-fp32 make test
```

| Test | Needs | Checks |
|---|---|---|
| `StitchingTests` | nothing | windowing and stitching against `engine.py` |
| `PipelineTests` | nothing; one case the videos | window starts, lengths and padding, stitched segments and cancellation, with a stub model |
| `WindowScoresTests` | nothing | the near-tie gate, the input array, and both output layouts in Float16 and Float32 |
| `ModelFilesTests` | nothing | locating, fingerprinting and caching compiled models |
| `FrameReaderTests` | ffmpeg, and the videos for three cases | every decoded frame byte for byte against ffmpeg's: the videos, rotated and mirrored copies, HEVC, 4K, and tracks whose range or matrix changes |
| `ReferenceTests` | the videos, a model | segments against the official PyTorch output |

`ReferenceTests` reads the model from `OMNISHOTCUT_MODEL`, and optionally
`OMNISHOTCUT_COMPUTE_UNITS` (`ane`, `gpu`, `cpu`, `all`), `OMNISHOTCUT_VERIFIER`,
`OMNISHOTCUT_THRESHOLD` and `OMNISHOTCUT_VIDEOS`; without a model it skips.
`Fixtures/omnishotcut-reference.json` is the official code's output, fp32 on
the CPU. The fp32 model and exact mode have to reproduce all 219 cuts with
their labels. The fp16 model may differ only in windows exact mode would run
again.

The videos are Blender open movies — the Sintel trailer, the Big Buck Bunny
trailer and Tears of Steel — © Blender Foundation,
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/),
[blender.org](https://www.blender.org). `make videos`
(`scripts/fetch-videos.sh`) downloads them and checks their SHA-256.

## License

MIT, as are OmniShotCut's code and weights. The model script adapts code from
DETR and VisTR, under the Apache License 2.0. See [LICENSE](LICENSE) and
[LICENSES](LICENSES).

```bibtex
@article{wang2026omnishotcut,
  title={OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer},
  author={Wang, Boyang and Xu, Guangyi and Zhang, Jiahui and Tang, Zhipeng and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2604.24762},
  year={2026}
}
```
