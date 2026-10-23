# Frame scaler specification

What `FrameScaler` computes: 8-bit 4:2:0 video frames, decoded by
AVFoundation, turned into the 128×96 RGB frames that
`ffmpeg -s 128x96 -pix_fmt rgb24` produces from the same video, byte for byte.

This document describes behaviour: arithmetic on integers, written as
mathematics. It contains no FFmpeg code and follows none of its structure, and
the implementation is written from it alone. Acceptance is `FrameReaderTests`,
which runs the `ffmpeg` binary as an outside reference and compares every
output byte.

## Notation

- All arithmetic is on signed 64-bit integers unless stated otherwise.
- `⌊x⌋` is floor. `trunc(x)` rounds toward zero. The two differ for negative
  quotients, and several steps below depend on which is used.
- `a ≫ k` is `⌊a / 2^k⌋`, an arithmetic shift, so it floors negative values.
- `clamp(x, lo, hi)` is `min(max(x, lo), hi)`.

## 1. Input, output and orientation

A batch holds `F` frames of the same size and colour. Each frame has:

- a **luma plane**: `SW × SH` bytes, row-major, no padding (`PlaneLayout.width`
  and `PlaneLayout.height`);
- a **chroma plane**: `CW × CH` samples of two bytes each, Cb then Cr,
  row-major, no padding (`chromaWidth` and `chromaHeight`).

The output is `F × H × W × 3` bytes: R, G and B for every pixel, rows top to
bottom. `W × H` is the requested size, 128 × 96 in practice, but any size must
work.

A frame is scaled as it is **displayed**, not as it is stored. `turns ∈ {0, 1,
2, 3}` is the number of clockwise quarter turns that display the stored
picture, and `flipped` says whether the display is also mirrored. For a plane
stored as `w × h` (luma: `SW × SH`; chroma: `CW × CH`), the displayed plane is
`w × h` for even `turns` and `h × w` for odd ones. Displayed pixel `(row, col)`
is stored pixel `(srow, scol)`, at byte offset `srow · w + scol` (times two
for chroma):

| turns | not flipped | flipped |
|---|---|---|
| 0 | `(row, col)` | `(h − 1 − row, col)` |
| 1 | `(h − 1 − col, row)` | `(col, row)` |
| 2 | `(h − 1 − row, w − 1 − col)` | `(row, w − 1 − col)` |
| 3 | `(col, w − 1 − row)` | `(h − 1 − col, w − 1 − row)` |

Below, `S_lw × S_lh` is the displayed luma size and `S_cw × S_ch` the
displayed chroma size.

## 2. Resampling weights for one axis

The picture is resampled along one axis at a time, with a set of integer
weights per output position. Four sets are built per scaler:

| Set | Source length `S` | Output length `D` | Tap multiple `A` | Unity `U` |
|---|---|---|---|---|
| luma, horizontal | `S_lw` | `W` | 4 | 2^14 |
| chroma, horizontal | `S_cw` | `⌈W / 2⌉` | 4 | 2^14 |
| luma, vertical | `S_lh` | `H` | 2 | 2^12 |
| chroma, vertical | `S_ch` | `H` | 2 | 2^12 |

Chroma is sited at the centre of each 2×2 luma block, so it uses the same
geometry as luma. A set is `M` taps per output position, `M` the same for
every position, plus, for every output position `i`, the source index `s_i` of
its first tap and `M` integer weights `q_{i,0} … q_{i,M−1}` that sum to `U`.
Output `i` is `Σ_j q_{i,j} · source[s_i + j]`, and a tap past the last source
sample always has weight 0.

### 2.1 Step and centres

The step is the ratio `S / D` in 16.16 fixed point, rounded:

    inc = ⌊(S · 2^16 + ⌊D / 2⌋) / D⌋

Output sample `i` is centred at source coordinate `c_i = ((i + ½) · S/D) − ½`.
In units of 2^−17 of a source sample, with the rounded step:

    P_i = (2i + 1) · inc − 2^16

**Identity.** When `|inc − 2^16| < 10`, the set has `N = 1` tap per output,
starting at `s_i = i` with weight `2^54` (see 2.3 for that scale), and skips
2.2–2.3; trimming, widening, edges and quantization (2.4–2.7) still apply.

### 2.2 Support

The number of candidate taps per output, `N`:

- when `inc ≤ 2^16` (enlarging or equal), `N = 5`;
- otherwise `N = 1 + ⌊(4S + D − 1) / D⌋`, which is `1 + ⌈4S/D⌉`.

Then `N = max(min(N, S − 2), 1)`.

The first candidate tap of output `i` is

    s_i = trunc((P_i − (N − 2) · 2^16) / 2^17)

and its taps are the source indices `s_i, s_i + 1, …, s_i + N − 1`. Here
`trunc` matters: near the left edge the quotient is negative.

### 2.3 Kernel

The kernel is the Mitchell–Netravali cubic (Mitchell and Netravali, 1988) with
`B = 0` and `C = 0.6`:

    k(t) = 1/6 · ((12 − 9B − 6C)|t|³ + (−18 + 12B + 6C)|t|² + (6 − 2B))          |t| < 1
    k(t) = 1/6 · ((−B − 6C)|t|³ + (6B + 30C)|t|² + (−12B − 48C)|t| + (8B + 24C))  1 ≤ |t| < 2
    k(t) = 0                                                                         |t| ≥ 2

It is evaluated in fixed point. Let `b = 0` and `c = trunc(0.6 · 2^24) =
10066329`, so B and C are in units of 2^−24.

For the tap at source index `x`, the distance from the centre in units of
2^−30 of a source sample is

    d = |x · 2^17 − P_i| · 2^13

When shrinking (`inc > 2^16`) the kernel is stretched by the ratio: `d =
⌊d · D / S⌋`, using the exact `D` and `S`, not `inc`.

Then, with

    dd  = (d · d) ≫ 30
    ddd = (dd · d) ≫ 30

the raw weight is

    d ≥ 2^31:          w = 0
    d < 2^30:          w = (12·2^24 − 9b − 6c) · ddd + (−18·2^24 + 12b + 6c) · dd + (6·2^24 − 2b) · 2^30
    otherwise:         w = (−b − 6c) · ddd + (6b + 30c) · dd + (−12b − 48c) · d + (8b + 24c) · 2^30

which is `6 · 2^54 · k(d / 2^30)`, up to the flooring in `dd` and `ddd`. Every
product fits in 64 bits.

Finally the weight is scaled down for large reductions. Let `r = ⌊S / D⌋`, `g
= ⌊log₂ r⌋` for `r ≥ 1` and `g = 0` for `r = 0`, and `k = min(g, 8)`. Then

    w = trunc(w / 2^k)

and the **reference magnitude** of the set is `R = 2^(54 − k)`. The identity
case's weight in 2.1 is `R` with `k = 0`.

### 2.4 Trimming

Weights near zero at either end of a position's taps are dropped. The
threshold is `τ = 0.002 · R`, compared in double precision.

Positions are trimmed from the **last to the first**, `i = D − 1` down to `0`,
because each one looks at the already-trimmed start of the next.

**From the left.** Keep a running total `σ = 0`. Up to `N` times: add `|w|` of
the current first tap to `σ`. Stop if `σ > τ`. Also stop if `i < D − 1` and
`s_i ≥ s_{i+1}`, so that a position never starts at or after the next one.
Otherwise drop the first tap: the remaining taps move one place towards the
front, a zero weight fills the last place, and `s_i` grows by 1.

**From the right**, on the result. Keep a new `σ = 0` and a length `n_i = N`.
For `j = N − 1` down to `1`: add `|w_j|` to `σ`, stop if `σ > τ`, otherwise
reduce `n_i` by 1. The first tap is never dropped here.

The set's width before rounding is `n = max_i n_i`.

### 2.5 Width

The width is rounded up to a multiple of `A`. The one exception is a vertical
set whose `n` is 1, which stays 1:

    A' = 1 if A = 2 and n = 1, otherwise A
    M  = ⌈n / A'⌉ · A'

Every position's taps are then cut or extended to `M`: taps `0 … M − 1`, zero
where `j ≥ N`. Weights past `M` are discarded, not moved anywhere.

### 2.6 Edges

A tap outside `0 … S − 1` gives its weight to the nearest edge sample. Per
position, with the `M` taps from 2.5:

1. If `s_i < 0`, every tap's weight moves to source index `max(s_i + j, 0)`,
   where weights that land on the same index add up, and the position then
   starts at `s_i = 0`.
2. Then, if `s_i + M > S`, the position moves to start at `s_i = max(S − M,
   0)`. Every tap's weight moves to source index `min(original index, S − 1)`,
   adding up again, and is stored in the slot for that index in the moved
   position.

After this every nonzero weight is at an index within `0 … S − 1` and within
the position's `M` taps. A tap past `S − 1` (possible only when `M > S`) has
weight 0.

### 2.7 Quantization

Each position's `M` weights are rescaled so they sum to `U`, with each tap's
rounding error carried to the next:

    T = Σ_j w_j
    m = trunc((T + U/2) / U),   then m = 1 if m = 0
    e = 0
    for j = 0 … M − 1:
        v = w_j + e
        q_j = sign(v) · ⌊(|v| + ⌊m/2⌋) / m⌋        (rounding half away from zero)
        e = v − q_j · m

The `q_j` are the set's weights. They fit in 32 bits.

## 3. Horizontal pass

For every frame, every displayed row `y` of a plane, and every output column
`x` of its horizontal set:

    h = ⌊(Σ_j p(y, s_x + j) · q_{x,j}) / 2^7⌋        (≫ 7, flooring negatives)
    h = min(h, 32767)

`p(y, col)` is the displayed sample (section 1). The sum fits in 32 bits. No
lower bound applies: `h` can be slightly negative.

The luma plane gives `S_lh` rows of `W` values. The chroma plane gives `S_ch`
rows of `⌈W / 2⌉` value pairs, filtered with the same weights for Cb and Cr.

## 4. Vertical pass and colour

### 4.1 Vertical filtering

For every output pixel `(x, y)`, with `ρ_y` from 4.2 and the vertical sets'
weights:

    Y = ρ_y + Σ_j h_luma(s_y + j, x)          · q_{y,j}^luma
    U = ρ_y + Σ_j h_Cb(s'_y + j, ⌊x / 2⌋)     · q_{y,j}^chroma
    V = ρ_y + Σ_j h_Cr(s'_y + j, ⌊x / 2⌋)     · q_{y,j}^chroma

Here `s_y` and `s'_y` are the luma and chroma vertical sets' starts. The sums
fit in 32 bits. Then:

    Y' = clamp(Y ≫ 19, −512, 767)
    U' = clamp(U ≫ 19, 0, 255)
    V' = clamp(V ≫ 19, 0, 255)

### 4.2 Rounding term

`ρ_y = 2^18`, except that `ρ_y = 0` for an output row `y` when all of these
hold:

- both vertical sets have `M = 2`;
- the luma set's two weights `(a, b)` for row `y` satisfy `a + b = 4096` and
  `0 ≤ b ≤ 4096`;
- the chroma set's two weights for row `y` satisfy the same.

### 4.3 YCbCr to RGB

A frame's colour is a **matrix** (BT.601, BT.709, SMPTE 240M or BT.2020) and a
**range** (full or limited/video). Each matrix has four positive integer
coefficients `(r, b, gu, gv)`, from its luma weights `Kr` and `Kb`, with `Kg =
1 − Kr − Kb`:

    r  = round(2^16 · 255/224 · 2(1 − Kr))
    b  = round(2^16 · 255/224 · 2(1 − Kb))
    gu = round(2^16 · 255/224 · 2Kb(1 − Kb) / Kg)
    gv = round(2^16 · 255/224 · 2Kr(1 − Kr) / Kg)

| Matrix | Kr | Kb | (r, b, gu, gv) |
|---|---|---|---|
| BT.601 | 0.299 | 0.114 | 104597, 132201, 25675, 53279 |
| BT.709 | 0.2126 | 0.0722 | 117489, 138438, 13975, 34925 |
| SMPTE 240M | 0.212 | 0.087 | 117579, 136230, 16907, 35559 |
| BT.2020 | 0.2627 | 0.0593 | 110013, 140363, 12277, 42626 |

The formula gives every entry except BT.2020's `r`, which the reference uses
as 110013, one below the formula's 110014. Use the table.

Then, per scaler:

    gu = −gu,  gv = −gv
    full range:    each of r, b, gu, gv = trunc(X · 224 / 255);  cy = 2^16;               oy = 0;        base = 896
    limited range: cy = trunc(2^16 · 255 / 219) = 76309;         oy = 16 · 2^16;          base = 838
    each of r, b, gu, gv = trunc((X · 2^16 + 2^15) / cy)

(trunc toward zero for the negative `gu` and `gv` too). With

    o(X, n) = −(X ≫ 9) + ((n · X) ≫ 16)
    L(i)    = clamp((−384 · 2^16 − 512 · cy − oy + i · cy + 2^15) ≫ 16, 0, 255)

the output pixel is

    R = L(base + o(r, V')                + Y')
    G = L(base + o(gu, U') + o(gv, V')   + Y')
    B = L(base + o(b, U')                + Y')

For every input the index of `L` is within `0 … 2047`. Computing `L` and `o`
from lookup tables or directly gives the same bytes.

## 5. The interface

`FrameScaler.swift` in `Sources/OmniShotCutCoreML` provides the following,
used by `FrameReader.swift`:

```swift
/// A frame's YCbCr→RGB matrix.
enum YCbCrMatrix: Sendable, Hashable {
    case bt601, bt709, smpte240m, bt2020
}

/// Scales and converts batches of one frame size and colour on the GPU.
final class FrameScaler {
    /// The stored plane sizes this scaler was built for.
    let layout: PlaneLayout

    init(
        layout: PlaneLayout, quarterTurns: Int, flipped: Bool, width: Int, height: Int,
        matrix: YCbCrMatrix, fullRange: Bool, metal: MetalContext
    ) throws

    /// Bytes of GPU-private scratch `encode` needs for `frames` frames.
    func scratchBytes(frames: Int) -> Int

    /// Encodes the conversion of `frames` frames (section 1's layouts) from
    /// `luma` and `chroma` into `rgb`, using `scratch` for intermediate
    /// values, on `commandBuffer`. It does not commit the command buffer.
    func encode(
        frames: Int, luma: MTLBuffer, chroma: MTLBuffer, scratch: MTLBuffer,
        rgb: MTLBuffer, into commandBuffer: MTLCommandBuffer
    ) throws
}
```

`PlaneLayout` and `MetalContext` already exist. `MetalContext.pipeline(_:source:constants:)`
compiles a kernel from Metal source at run time, with integer preprocessor
macros, and caches it. Metal source is compiled at run time so that SwiftPM
builds need no metallib. Errors are `OmniShotCutError.metalUnavailable`.

Every per-pixel step runs on the GPU. A batch is up to 16 frames up to 1080p, or
5 frames of 4K, and decoding a 12-minute 720p film must stay well under the
time decoding takes. Building the weights and coefficients on the CPU once per
scaler is fine.
