"""Core ML export formulations of OmniShotCutWindow, for scripts/convert.py.

Both modules compute OmniShotCutWindow's function (scripts/omnishotcut_torch.py)
under the Core ML model contract:

    input   frames  [100, 3, 96, 128] RGB 0..255, black (0) past the end of the video
    outputs intra, inter, end logits of the 24 queries, [24, C] or [1, C, 1, 24]

and are built from a loaded OmniShotCutWindow, so they depend only on that
module and the official checkpoint.

GpuWindow, for FLOAT32 on the GPU: OmniShotCutWindow's sequence-major graph
(tokens [1200, 576], nn.Linear, attention with the heads reshaped to a batch
axis) with every FrozenBatchNorm2d folded into the convolution before it.
The input normalization stays a separate step.

AneWindow, for FLOAT16 on the Neural Engine, in the layout of Apple's
ml-ane-transformers: activations [1, C, 1, S] with the channels on axis 1 and
the sequence last; every linear a 1x1 convolution; attention split per head
along the channels with `split` and computed with two einsums per head
("bchq,bkhc->bkhq", softmax over the key axis 1, "bkhq,bchk->bchq"), which
coremltools lowers to MIL's native `einsum` op; the 1/sqrt(72) query scale
folded into the query projection; LayerNorm over the channel axis written
out as mean / variance / rsqrt followed by the affine multiply and add,
which coremltools converts to `layer_norm` (axes [1], no gamma or beta)
followed by a `batch_norm` with mean 0, variance 1 and epsilon 1e-5 that
carries the LayerNorm weight and bias (its fuse_elementwise_to_batchnorm
pass); the resulting 1 / sqrt(1 + 1e-5) scale is below float16 resolution;
every FrozenBatchNorm2d folded into the convolution before it; the ImageNet
normalization folded into conv1.
Its logits come out as [1, C, 1, 24] (the heads' convolution output) or,
with output_layout="queries", transposed to [24, C].

Folding the input normalization into conv1. The official input to conv1 is
n = (x / 255 - m) / s per channel, zero-padded by 3 pixels: conv1 sees
pad0(n), where pad0 pads with zeros. With a = 1 / (255 s) and c = -m / s,
n = a x + c, and since scaling an input channel commutes with zero padding,

    conv_W(pad0(n)) + b = conv_{W a}(pad0(x)) + [b + conv_W(pad0(c 1))]

where 1 is an all-ones 96 x 128 image and conv_{W a} scales input channel i of
W by a_i. The bracket is a constant [64, 48, 64] map K: one value per output
channel in the interior, b + sum over (i, ky, kx) of W c_i, and different
values where the 7 x 7, stride 2, padding 3 window covers padding (output
rows 0, 1 and 47, columns 0, 1 and 63), because there the padded zeros of n
are not the zeros of x. W and b already include the folded BatchNorm. K and
W a are computed in float64 and cast to the module dtype. The normalized
input is never materialized, so FLOAT16 execution does not round it.
Rounding n to float16, with everything else computed in float32, moves the
logits by up to 0.09.
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from omnishotcut_torch import (  # noqa: E402
    HEIGHT, HIDDEN, IMAGENET_MEAN, IMAGENET_STD, QUERIES, WIDTH,
    BasicBlock, OmniShotCutWindow,
)

OUTPUT_LAYOUTS = ("channels", "queries")


# MARK: - Backbone folding


def fold_batch_norm(conv, norm):
    """`norm(conv(x))` as one convolution with a bias, folded in float64 and
    cast to conv's dtype. FrozenBatchNorm2d computes x * scale + shift with
    scale = weight / sqrt(running_var + eps), shift = bias - running_mean * scale."""
    dtype = conv.weight.dtype
    scale = norm.weight.double() * (norm.running_var.double() + norm.eps).rsqrt()
    shift = norm.bias.double() - norm.running_mean.double() * scale
    if conv.bias is not None:
        shift = shift + conv.bias.double() * scale
    folded = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                       conv.padding, conv.dilation, conv.groups, bias=True)
    folded.weight = nn.Parameter((conv.weight.double() * scale.reshape(-1, 1, 1, 1)).to(dtype),
                                 requires_grad=False)
    folded.bias = nn.Parameter(shift.to(dtype), requires_grad=False)
    return folded


def fold_input_normalization(conv, height=HEIGHT, width=WIDTH):
    """(conv', offset) with conv'(pad0(x)) + offset == conv(pad0((x / 255 - m) / s))
    for raw 0..255 input x; derivation in the module docstring. conv' has no
    bias; offset is [1, out_channels, out_height, out_width]."""
    dtype = conv.weight.dtype
    weight = conv.weight.double()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float64)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float64)
    a = (1.0 / (255.0 * std)).reshape(1, 3, 1, 1)
    c = (-mean / std).reshape(1, 3, 1, 1)
    scaled = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                       conv.padding, conv.dilation, conv.groups, bias=False)
    scaled.weight = nn.Parameter((weight * a).to(dtype), requires_grad=False)
    constant = c.expand(1, 3, height, width)
    offset = F.conv2d(constant, weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    if conv.bias is not None:
        offset = offset + conv.bias.double().reshape(1, -1, 1, 1)
    return scaled, offset.to(dtype)


class FoldedBasicBlock(nn.Module):
    """BasicBlock with its batch norms folded into its convolutions."""

    def __init__(self, block: BasicBlock):
        super().__init__()
        self.conv1 = fold_batch_norm(block.conv1, block.bn1)
        self.conv2 = fold_batch_norm(block.conv2, block.bn2)
        self.downsample = (None if block.downsample is None
                           else fold_batch_norm(block.downsample[0], block.downsample[1]))

    def forward(self, x):
        # torchvision's order: the main branch, then the downsample branch.
        y = self.conv2(F.relu(self.conv1(x)))
        identity = x if self.downsample is None else self.downsample(x)
        return F.relu(y + identity)


class FoldedResNet18(nn.Module):
    """omnishotcut_torch.ResNet18 with every FrozenBatchNorm2d folded. With
    fold_input, the input is raw RGB 0..255 and the ImageNet normalization is
    folded into conv1 and the constant `conv1_offset`; otherwise the input is
    the normalized image, as for ResNet18."""

    def __init__(self, resnet, fold_input):
        super().__init__()
        self.fold_input = fold_input
        conv1 = fold_batch_norm(resnet.conv1, resnet.bn1)
        if fold_input:
            conv1, offset = fold_input_normalization(conv1)
            self.register_buffer("conv1_offset", offset)
        self.conv1 = conv1
        self.maxpool = copy.deepcopy(resnet.maxpool)
        self.layers = nn.Sequential(*[
            FoldedBasicBlock(block)
            for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4) for block in layer])

    def forward(self, x):
        x = self.conv1(x)
        if self.fold_input:
            x = x + self.conv1_offset
        return self.layers(self.maxpool(F.relu(x)))


# MARK: - GPU formulation (sequence-major)


class GpuWindow(nn.Module):
    """OmniShotCutWindow with the backbone's batch norms folded:
    frames [100, 3, 96, 128] RGB 0..255 -> intra [24, 10], inter [24, 7],
    end [24, 102]. The transformer and heads are OmniShotCutWindow's own
    modules (copied)."""

    def __init__(self, window: OmniShotCutWindow, fold_input=False):
        super().__init__()
        self.normalize = None if fold_input else copy.deepcopy(window.normalize)
        self.backbone = FoldedResNet18(window.backbone, fold_input)
        self.input_proj = copy.deepcopy(window.input_proj)
        self.encoder = copy.deepcopy(window.encoder)
        self.decoder = copy.deepcopy(window.decoder)
        self.heads = copy.deepcopy(window.heads)
        self.register_buffer("pos", window.pos.clone(), persistent=False)

    def forward(self, frames):
        x = frames if self.normalize is None else self.normalize(frames)
        features = self.input_proj(self.backbone(x))                      # [100, 576, 3, 4]
        tokens = features.flatten(2).transpose(1, 2).reshape(-1, HIDDEN)  # [1200, 576]
        memory = self.encoder(tokens, self.pos)
        return self.heads(self.decoder(memory, self.pos))


# MARK: - ANE formulation ([1, C, 1, S])


def conv1x1(linear, scale=1.0):
    """nn.Linear(in, out) as nn.Conv2d(in, out, 1), weight and bias times `scale`."""
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1)
    weight = linear.weight.double() * scale
    bias = linear.bias.double() * scale
    conv.weight = nn.Parameter(weight.reshape(*weight.shape, 1, 1).to(linear.weight.dtype), requires_grad=False)
    conv.bias = nn.Parameter(bias.to(linear.bias.dtype), requires_grad=False)
    return conv


class ChannelLayerNorm(nn.Module):
    """nn.LayerNorm over axis 1 of [B, C, 1, S]: biased variance, eps inside
    the square root, then the affine weight and bias."""

    def __init__(self, norm: nn.LayerNorm):
        super().__init__()
        self.eps = norm.eps
        self.register_buffer("weight", norm.weight.detach().clone().reshape(1, -1, 1, 1))
        self.register_buffer("bias", norm.bias.detach().clone().reshape(1, -1, 1, 1))

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        variance = (centered * centered).mean(dim=1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class AneAttention(nn.Module):
    """omnishotcut_torch.Attention on [1, D, 1, L] activations. The heads are
    `split` along the channels; per head, scores [1, Lk, 1, Lq] with the keys
    on axis 1, softmax over axis 1, and the weighted sum of the values. The
    query scale sqrt(1 / head_dim) is folded into the query projection
    ((W x + b) * s = (s W) x + s b)."""

    def __init__(self, attention):
        super().__init__()
        self.head_dim = attention.head_dim
        self.q_proj = conv1x1(attention.q_proj, attention.scale)
        self.k_proj = conv1x1(attention.k_proj)
        self.v_proj = conv1x1(attention.v_proj)
        self.out_proj = conv1x1(attention.out_proj)

    def forward(self, query, key, value):
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        q = q.split(self.head_dim, dim=1)                                    # [1, 72, 1, Lq] each
        k = k.transpose(1, 3).split(self.head_dim, dim=3)                    # [1, Lk, 1, 72] each
        v = v.split(self.head_dim, dim=1)                                    # [1, 72, 1, Lk] each
        weights = [torch.einsum("bchq,bkhc->bkhq", qh, kh).softmax(dim=1)   # [1, Lk, 1, Lq] each
                   for qh, kh in zip(q, k)]
        heads = [torch.einsum("bkhq,bchk->bchq", wh, vh)                     # [1, 72, 1, Lq] each
                 for wh, vh in zip(weights, v)]
        return self.out_proj(torch.cat(heads, dim=1))


class AneEncoderLayer(nn.Module):
    """omnishotcut_torch.EncoderLayer on [1, D, 1, S]."""

    def __init__(self, layer):
        super().__init__()
        self.self_attn = AneAttention(layer.self_attn)
        self.linear1 = conv1x1(layer.linear1)
        self.linear2 = conv1x1(layer.linear2)
        self.norm1 = ChannelLayerNorm(layer.norm1)
        self.norm2 = ChannelLayerNorm(layer.norm2)

    def forward(self, src, pos):
        q = k = src + pos
        src = self.norm1(src + self.self_attn(q, k, src))
        return self.norm2(src + self.linear2(F.relu(self.linear1(src))))


class AneDecoderLayer(nn.Module):
    """omnishotcut_torch.DecoderLayer on [1, D, 1, L]; the cross-attention
    keys `memory + pos` are computed once by the caller."""

    def __init__(self, layer):
        super().__init__()
        self.self_attn = AneAttention(layer.self_attn)
        self.cross_attn = AneAttention(layer.cross_attn)
        self.linear1 = conv1x1(layer.linear1)
        self.linear2 = conv1x1(layer.linear2)
        self.norm1 = ChannelLayerNorm(layer.norm1)
        self.norm2 = ChannelLayerNorm(layer.norm2)
        self.norm3 = ChannelLayerNorm(layer.norm3)

    def forward(self, tgt, memory, memory_keys, query_pos):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, k, tgt))
        tgt = self.norm2(tgt + self.cross_attn(tgt + query_pos, memory_keys, memory))
        return self.norm3(tgt + self.linear2(F.relu(self.linear1(tgt))))


class AneWindow(nn.Module):
    """frames [100, 3, 96, 128] RGB 0..255 -> intra, inter, end logits as
    [1, C, 1, 24] (output_layout "channels") or [24, C] ("queries")."""

    def __init__(self, window: OmniShotCutWindow, output_layout="channels"):
        super().__init__()
        assert output_layout in OUTPUT_LAYOUTS, output_layout
        self.output_layout = output_layout
        self.backbone = FoldedResNet18(window.backbone, fold_input=True)
        self.input_proj = copy.deepcopy(window.input_proj)
        self.encoder = nn.ModuleList(AneEncoderLayer(layer) for layer in window.encoder.layers)
        self.decoder = nn.ModuleList(AneDecoderLayer(layer) for layer in window.decoder.layers)
        self.norm = ChannelLayerNorm(window.decoder.norm)
        self.intra = conv1x1(window.heads.intra)
        self.inter = conv1x1(window.heads.inter)
        self.end = conv1x1(window.heads.end)
        # [S, D] -> [1, D, 1, S]
        self.register_buffer("pos", window.pos.t().reshape(1, HIDDEN, 1, -1).clone(), persistent=False)
        self.register_buffer("query_pos", window.decoder.query_embed.detach().t().reshape(1, HIDDEN, 1, QUERIES)
                             .clone(), persistent=False)

    def forward(self, frames):
        features = self.input_proj(self.backbone(frames))                  # [100, 576, 3, 4]
        # [100, 576, 12] -> [576, 100, 12] -> [1, 576, 1, 1200]: frame, then row, then column
        x = features.flatten(2).permute(1, 0, 2).reshape(1, HIDDEN, 1, -1)
        for layer in self.encoder:
            x = layer(x, self.pos)
        memory_keys = x + self.pos
        tgt = torch.zeros_like(self.query_pos)
        for layer in self.decoder:
            tgt = layer(tgt, x, memory_keys, self.query_pos)
        hs = self.norm(tgt)
        logits = self.intra(hs), self.inter(hs), self.end(hs)              # [1, C, 1, 24]
        if self.output_layout == "queries":
            return tuple(value[0, :, 0].t() for value in logits)
        return logits


def build(window, formulation, output_layout="channels"):
    """Export module of a loaded OmniShotCutWindow: "gpu" or "ane"."""
    if formulation == "gpu":
        module = GpuWindow(window)
    elif formulation == "ane":
        module = AneWindow(window, output_layout)
    else:
        raise ValueError(f"unknown formulation {formulation!r}")
    module = module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module

