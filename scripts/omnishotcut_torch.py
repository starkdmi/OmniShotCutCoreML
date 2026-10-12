"""OmniShotCut v1.5 for one 100-frame window, as a static-shape PyTorch module.

The official model (omnishotcut/architecture in the official repository) takes a
NestedTensor, builds a padding mask, computes the position embedding per call
and runs `nn.MultiheadAttention`. For one full window every one of those is a
constant: frames all have the same size, so the mask is all False and the
position embedding depends only on the feature-map shape. This module
computes the same function with those constants folded in, so that
`torch.jit.trace`, `torch.export` and coremltools see a single static graph:

    frames [100, 3, 96, 128] RGB 0..255, black (0) past the end of the video
      -> Normalize               x / 255, ImageNet mean and std (val transform)
      -> ResNet18                DETR FrozenBatchNorm2d, layer4 output [100, 512, 3, 4]
      -> input_proj              1x1 conv 512 -> 576
      -> tokens [1200, 576]      frame, then row, then column
      -> Encoder x3              post-norm, position added to queries and keys
      -> Decoder x6 + norm       24 queries
      -> Heads                   intra [24, 10], inter [24, 7], end [24, 102]

Only the last decoder layer feeds the heads; the official auxiliary outputs
are training-only. Submodules are separate classes so that later stages can
replace one of them (for example an ANE-friendly attention) or split the
model at a module boundary.
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

WINDOW = 100
HEIGHT, WIDTH = 96, 128
HIDDEN = 576
HEADS = 8
FEED_FORWARD = 2048
ENCODER_LAYERS = 3
DECODER_LAYERS = 6
QUERIES = 24
INTRA_CLASSES = 10
INTER_CLASSES = 7
END_CLASSES = WINDOW + 2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OmniShotCut_ckpt.pth of uva-cv-lab/OmniShotCut_v1.5, revision
# fd050da5732d1fc5d5baabff6c64abd9de1f94aa.
CHECKPOINT_SHA256 = "5bafe33d8d0dde16c6e24fc63dfc580da9d5731804d6e51e1fc8de21d4600999"


# MARK: - Input


class Normalize(nn.Module):
    """`Video_Augmentation_Transform(set_type="val")`: `uint8 -> float / 255`,
    then `torchvision.transforms.functional.normalize` with the ImageNet mean
    and std. The val transform has no resize, crop or other augmentation.

    The constants are non-persistent buffers, rebuilt from the Python floats
    by `set_constants_dtype` so that a float64 module uses float64 constants,
    as `F.normalize` does for a float64 input.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.ones(1, 3, 1, 1), persistent=False)
        self.set_constants_dtype(torch.float32)

    def set_constants_dtype(self, dtype):
        device = self.mean.device
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).reshape(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=dtype, device=device).reshape(1, 3, 1, 1)

    def forward(self, frames):
        return (frames / 255.0 - self.mean) / self.std


# MARK: - Backbone


class FrozenBatchNorm2d(nn.Module):
    """DETR's FrozenBatchNorm2d (architecture/backbone.py): fixed statistics,
    eps = 1e-5 added to the variance before the reciprocal square root."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BasicBlock(nn.Module):
    """torchvision's ResNet BasicBlock with FrozenBatchNorm2d."""

    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, stride=1, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(outputs)
        if stride != 1 or inputs != outputs:
            self.downsample = nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False),
                FrozenBatchNorm2d(outputs),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + identity)


class ResNet18(nn.Module):
    """torchvision resnet18 up to layer4 (DETR's IntermediateLayerGetter with
    return_layers {"layer4": "0"}; no avgpool, no fc). NCHW in, NCHW out at
    1/32 of the input size: [100, 3, 96, 128] -> [100, 512, 3, 4]."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1), BasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512, 1))

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


# MARK: - Position embedding


def sine_position_embedding(frames, height, width, hidden=HIDDEN, temperature=10000):
    """VisTR's 3-D PositionEmbeddingSine (architecture/position_encoding.py)
    with normalize=True, scale 2*pi, num_pos_feats = hidden // 3 and an
    all-False mask, evaluated with the official operations in float32 (the
    official code computes it in float32 whatever the model dtype and casts
    the result). Returns [frames * height * width, hidden] in token order:
    frame, then row, then column; channels are (time, y, x), each
    hidden // 3 wide, sine on even and cosine on odd channels."""
    features = hidden // 3
    scale = 2 * math.pi
    not_mask = torch.ones(1, frames, height, width, dtype=torch.bool)
    z_embed = not_mask.cumsum(1, dtype=torch.float32)
    y_embed = not_mask.cumsum(2, dtype=torch.float32)
    x_embed = not_mask.cumsum(3, dtype=torch.float32)
    eps = 1e-6
    z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * scale
    y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * scale

    dim_t = torch.arange(features, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / features)

    pos_x = x_embed[:, :, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, :, None] / dim_t
    pos_z = z_embed[:, :, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos = torch.cat((pos_z, pos_y, pos_x), dim=4)          # [1, F, H, W, hidden]
    return pos.reshape(frames * height * width, hidden)


# MARK: - Transformer


class Attention(nn.Module):
    """`nn.MultiheadAttention` in eval mode without masks, with the packed
    `in_proj` split into three linears. Sequence-first without a batch axis:
    query [Lq, D], key and value [Lk, D] -> [Lq, D].

    The order of operations is `F.multi_head_attention_forward`'s: queries
    scaled by sqrt(1 / head_dim) before the product with the keys, softmax
    over the keys, product with the values, heads concatenated, out_proj.
    """

    def __init__(self, dim=HIDDEN, heads=HEADS):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = math.sqrt(1.0 / float(self.head_dim))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        q_len, dim = query.shape
        k_len = key.shape[0]
        q = self.q_proj(query).reshape(q_len, self.heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).reshape(k_len, self.heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).reshape(k_len, self.heads, self.head_dim).transpose(0, 1)
        weights = torch.softmax(torch.matmul(q * self.scale, k.transpose(1, 2)), dim=-1)
        attended = torch.matmul(weights, v)                 # [heads, Lq, head_dim]
        attended = attended.transpose(0, 1).reshape(q_len, dim)
        return self.out_proj(attended)


class EncoderLayer(nn.Module):
    """DETR TransformerEncoderLayer.forward_post (normalize_before=False,
    dropout inactive): the position is added to queries and keys only."""

    def __init__(self, dim=HIDDEN, heads=HEADS, feed_forward=FEED_FORWARD):
        super().__init__()
        self.self_attn = Attention(dim, heads)
        self.linear1 = nn.Linear(dim, feed_forward)
        self.linear2 = nn.Linear(feed_forward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, src, pos):
        q = k = src + pos
        src = self.norm1(src + self.self_attn(q, k, src))
        return self.norm2(src + self.linear2(F.relu(self.linear1(src))))


class DecoderLayer(nn.Module):
    """DETR TransformerDecoderLayer.forward_post: self-attention with
    q = k = tgt + query_pos and v = tgt, cross-attention with
    q = tgt + query_pos, k = memory + pos, v = memory, then the FFN."""

    def __init__(self, dim=HIDDEN, heads=HEADS, feed_forward=FEED_FORWARD):
        super().__init__()
        self.self_attn = Attention(dim, heads)
        self.cross_attn = Attention(dim, heads)
        self.linear1 = nn.Linear(dim, feed_forward)
        self.linear2 = nn.Linear(feed_forward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt, memory, pos, query_pos):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, k, tgt))
        tgt = self.norm2(tgt + self.cross_attn(tgt + query_pos, memory + pos, memory))
        return self.norm3(tgt + self.linear2(F.relu(self.linear1(tgt))))


class Encoder(nn.Module):
    def __init__(self, layers=ENCODER_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(layers))

    def forward(self, src, pos):
        for layer in self.layers:
            src = layer(src, pos)
        return src


class Decoder(nn.Module):
    """DETR TransformerDecoder with return_intermediate=True, reduced to the
    last layer: the heads read only `norm(output of the last layer)`."""

    def __init__(self, layers=DECODER_LAYERS, queries=QUERIES, dim=HIDDEN):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.query_embed = nn.Parameter(torch.zeros(queries, dim))

    def forward(self, memory, pos):
        query_pos = self.query_embed
        tgt = torch.zeros_like(query_pos)
        for layer in self.layers:
            tgt = layer(tgt, memory, pos, query_pos)
        return self.norm(tgt)


class Heads(nn.Module):
    """intra_relation_class_embed, inter_relation_class_embed and
    range_class_embed. `end` has window + 2 classes: the end frame 0..100,
    and the padding class last."""

    def __init__(self, dim=HIDDEN):
        super().__init__()
        self.intra = nn.Linear(dim, INTRA_CLASSES)
        self.inter = nn.Linear(dim, INTER_CLASSES)
        self.end = nn.Linear(dim, END_CLASSES)

    def forward(self, hs):
        return self.intra(hs), self.inter(hs), self.end(hs)


# MARK: - Model


class OmniShotCutWindow(nn.Module):
    """One window: frames [100, 3, 96, 128] RGB 0..255 (black past the end of
    the video) -> (intra [24, 10], inter [24, 7], end [24, 102]) logits of the
    last decoder layer, batch element 0 of the official outputs
    `intra_clip_logits`, `inter_clip_logits` and `pred_shot_logits`."""

    def __init__(self, frames=WINDOW, height=HEIGHT, width=WIDTH):
        super().__init__()
        self.frames = frames
        self.normalize = Normalize()
        self.backbone = ResNet18()
        self.input_proj = nn.Conv2d(512, HIDDEN, kernel_size=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.heads = Heads()
        feature_height, feature_width = math.ceil(height / 32), math.ceil(width / 32)
        self.register_buffer(
            "pos", sine_position_embedding(frames, feature_height, feature_width), persistent=False)

    def set_constants_dtype(self, dtype):
        """Constants that are not weights: the normalization's mean and std
        in `dtype`, and the float32 position embedding cast to `dtype`."""
        self.normalize.set_constants_dtype(dtype)
        self.pos = self.pos.to(dtype)

    def tokens(self, features):
        """[F, C, H, W] -> [F * H * W, C]. Official order: src_proj
        [B*F, C, H, W] -> reshape [B, F, C, H, W] -> permute [B, C, F, H, W]
        -> flatten [B, C, F, H*W] -> transformer flatten(2).permute(2, 0, 1)
        -> [F*H*W, B, C]: frame, then row, then column."""
        count, channels = features.shape[0], features.shape[1]
        return features.flatten(2).transpose(1, 2).reshape(-1, channels)

    def forward(self, frames):
        x = self.normalize(frames)
        features = self.input_proj(self.backbone(x))
        memory = self.encoder(self.tokens(features), self.pos)
        hs = self.decoder(memory, self.pos)
        return self.heads(hs)


# MARK: - Checkpoint


def checkpoint_key_map(state_keys):
    """Explicit mapping from module state_dict keys to checkpoint keys.

    Returns (mapping, unused) where mapping[module_key] = (checkpoint_key,
    slice index or None) and `unused` lists checkpoint keys deliberately not
    loaded, each with the reason. Each attention's packed in_proj_weight
    [3 * 576, 576] and in_proj_bias [3 * 576] are split in q, k, v order
    (nn.MultiheadAttention's `w.chunk(3)`)."""
    mapping = {}

    def same(module_key, checkpoint_key):
        mapping[module_key] = (checkpoint_key, None)

    def attention(module_prefix, checkpoint_prefix):
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            mapping[f"{module_prefix}.{name}.weight"] = (f"{checkpoint_prefix}.in_proj_weight", index)
            mapping[f"{module_prefix}.{name}.bias"] = (f"{checkpoint_prefix}.in_proj_bias", index)
        for kind in ("weight", "bias"):
            same(f"{module_prefix}.out_proj.{kind}", f"{checkpoint_prefix}.out_proj.{kind}")

    # ResNet18: `backbone.0` is the Joiner's Backbone, `.body` its IntermediateLayerGetter.
    for key in state_keys:
        if key.startswith("backbone."):
            same(key, "backbone.0.body." + key[len("backbone."):])
    for kind in ("weight", "bias"):
        same(f"input_proj.{kind}", f"input_proj.{kind}")
    for i in range(ENCODER_LAYERS):
        prefix, source = f"encoder.layers.{i}", f"transformer.encoder.layers.{i}"
        attention(f"{prefix}.self_attn", f"{source}.self_attn")
        for name in ("linear1", "linear2", "norm1", "norm2"):
            for kind in ("weight", "bias"):
                same(f"{prefix}.{name}.{kind}", f"{source}.{name}.{kind}")
    for i in range(DECODER_LAYERS):
        prefix, source = f"decoder.layers.{i}", f"transformer.decoder.layers.{i}"
        attention(f"{prefix}.self_attn", f"{source}.self_attn")
        attention(f"{prefix}.cross_attn", f"{source}.multihead_attn")
        for name in ("linear1", "linear2", "norm1", "norm2", "norm3"):
            for kind in ("weight", "bias"):
                same(f"{prefix}.{name}.{kind}", f"{source}.{name}.{kind}")
    for kind in ("weight", "bias"):
        same(f"decoder.norm.{kind}", f"transformer.decoder.norm.{kind}")
        same(f"heads.intra.{kind}", f"intra_relation_class_embed.{kind}")
        same(f"heads.inter.{kind}", f"inter_relation_class_embed.{kind}")
        same(f"heads.end.{kind}", f"range_class_embed.{kind}")
    same("decoder.query_embed", "query_embed.weight")

    # FrozenBatchNorm2d._load_from_state_dict drops num_batches_tracked; the
    # v1.5 checkpoint has none, but a torchvision-style export would.
    unused = {
        key: "BatchNorm counter, not used by FrozenBatchNorm2d"
        for key in state_keys if key.endswith("num_batches_tracked")
    }
    return mapping, unused


def load_state(module, checkpoint_state):
    """Copy the official state dict into `module` through `checkpoint_key_map`.
    Raises ValueError unless every checkpoint key is consumed or listed as
    unused and every persistent parameter and buffer of the module is
    loaded."""
    targets = module.state_dict()
    mapping, _ = checkpoint_key_map(list(targets.keys()))
    _, unused = checkpoint_key_map(list(checkpoint_state.keys()))

    missing = sorted(set(targets) - set(mapping))
    if missing:
        raise ValueError(f"module keys without a checkpoint source: {missing}")
    extra = sorted(set(mapping) - set(targets))
    if extra:
        raise ValueError(f"mapped keys the module does not have: {extra}")

    consumed = set()
    loaded = {}
    for module_key, (checkpoint_key, index) in mapping.items():
        if checkpoint_key not in checkpoint_state:
            raise ValueError(f"checkpoint lacks {checkpoint_key} (for {module_key})")
        value = checkpoint_state[checkpoint_key]
        if index is not None:
            value = value.chunk(3, dim=0)[index]
        target = targets[module_key]
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"{checkpoint_key}[{index}] {tuple(value.shape)} -> {module_key} {tuple(target.shape)}")
        loaded[module_key] = value.detach().clone().to(target.dtype)
        consumed.add(checkpoint_key)

    leftover = sorted(set(checkpoint_state) - consumed - set(unused))
    if leftover:
        raise ValueError(f"checkpoint keys neither loaded nor listed as unused: {leftover}")
    result = module.load_state_dict(loaded, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"load_state_dict: {result}")
    return sorted(consumed), unused


def load_official(checkpoint_path, dtype=torch.float32):
    """Build OmniShotCutWindow from the official checkpoint
    ({'args', 'model'}) in eval mode and `dtype`. Returns (module, args).

    The checkpoint is unpickled with weights_only=True: tensors, containers
    and the argparse.Namespace of its training arguments, no other code.
    Raises ValueError when its arguments describe another architecture."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(os.path.expanduser(checkpoint_path), map_location="cpu", weights_only=True)
    args = state["args"]
    expected = {
        "backbone": "resnet18", "hidden_dim": HIDDEN, "enc_layers": ENCODER_LAYERS,
        "dec_layers": DECODER_LAYERS, "nheads": HEADS, "dim_feedforward": FEED_FORWARD,
        "num_queries": QUERIES, "num_intra_relation_classes": INTRA_CLASSES,
        "num_inter_relation_classes": INTER_CLASSES, "max_process_window_length": WINDOW,
        "process_height": HEIGHT, "process_width": WIDTH, "pre_norm": False,
        "position_embedding": "sine", "dilation": False, "masks": False,
    }
    for name, value in expected.items():
        if not hasattr(args, name):
            raise ValueError(f"checkpoint args lacks {name}; this module implements {name} = {value!r}")
        actual = getattr(args, name)
        if actual != value:
            raise ValueError(f"checkpoint args.{name} = {actual!r}, this module implements {value!r}")
    module = OmniShotCutWindow()
    load_state(module, state["model"])
    module = module.to(dtype).eval()
    module.set_constants_dtype(dtype)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module, args

