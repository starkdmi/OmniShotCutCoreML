#!/usr/bin/env python3
"""Convert the official OmniShotCut checkpoint to the Core ML models the package loads.

    pip install -r scripts/requirements.txt
    curl -LO https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/fd050da5732d1fc5d5baabff6c64abd9de1f94aa/OmniShotCut_ckpt.pth
    python scripts/convert.py OmniShotCut_ckpt.pth models                  # both models
    python scripts/convert.py OmniShotCut_ckpt.pth models --precision fp16

coremltools 9.0 supports torch up to 2.7.0. The shipped models were converted
with torch 2.7.0, numpy 2.5.4 and Python 3.12; the traced graph and its MIL
lowering depend on the torch version, so the same digests are expected only
with the same versions. With them, weights/weight.bin (the same file in the
.mlpackage and the .mlmodelc) has SHA-256
4d821f8b0b2f195567220ec72bff18a218cda402fbb93f4aa27f1d036d24ca8d for fp16 and
3fc2f1fd103e7b2771d2c69b6f7788ccf0857b7ce572209d9edea3aecfda7a77 for fp32.
Manifest.json and model.mlmodel differ between conversions (random UUIDs,
protobuf map order). The checkpoint is checked against the v1.5 release's
SHA-256 before it is loaded; another checkpoint is an error unless
--any-checkpoint is given.

Writes <destination>/OmniShotCut-v1.5-fp16.mlpackage and
OmniShotCut-v1.5-fp32.mlpackage, each compiled to a sibling .mlmodelc. Loading
an .mlpackage compiles it to a temporary directory on every load, which also
misses the Neural Engine compile cache; load the .mlmodelc.

fp16 is AneWindow (scripts/coreml_modules.py) with FLOAT16 compute precision,
for the Neural Engine: 664 ops, all placed on the Neural Engine under
cpuAndNeuralEngine. Its logits are [1, C, 1, 24], the output layout of the
heads' 1x1 convolutions; --output-layout queries gives [24, C] at the cost of
three transposes on the CPU. fp32 is GpuWindow with FLOAT32 compute
precision: 365 ops, all on the GPU under cpuAndGPU, logits [24, C]. Both
read "frames", Float16 [100, 3, 96, 128], RGB 0..255. fp16 returns Float16
logits, fp32 Float32.

The folds (in both models the batch norms into the convolutions; in the fp16
model also the input normalization into conv1 and the attention scale into
the query projections) are computed on a float64 copy of the checkpoint, and
every tensor is then rounded once: to float16 for the fp16 model, to float32
for the fp32 model. The deployment target is macOS 13 / iOS 16, the first
that accepts Float16 inputs. The model's user-defined metadata holds the
configuration the Swift loader reads: window_length, width, height, queries,
intra_classes, inter_classes, end_classes, precision, input_layout,
output_layout, compute_units, formulation, checkpoint_sha256.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from coreml_modules import OUTPUT_LAYOUTS, build  # noqa: E402
from omnishotcut_torch import (  # noqa: E402
    CHECKPOINT_SHA256, END_CLASSES, HEIGHT, INTER_CLASSES, INTRA_CLASSES, QUERIES, WIDTH, WINDOW,
    load_official,
)

NAME = "OmniShotCut-v1.5"
VERSION = "1.5"
AUTHOR = "Converted by Dmitry Starkov from UVA Computer Vision Lab's OmniShotCut v1.5"
LICENSE = "MIT"
TARGETS = ("macOS13", "macOS14", "macOS15")

# precision -> (formulation, Core ML compute precision, output dtype, compute units the model is built for)
PRECISIONS = {
    "fp16": ("ane", "FLOAT16", np.float16, "cpuAndNeuralEngine"),
    "fp32": ("gpu", "FLOAT32", np.float32, "cpuAndGPU"),
}
CLASSES = {"intra": INTRA_CLASSES, "inter": INTER_CLASSES, "end": END_CLASSES}
DESCRIPTIONS = {
    "frames": f"{WINDOW} frames of {WIDTH}x{HEIGHT} RGB (width x height) as [frame, channel, row, column], "
              "values 0..255; frames past the end of the video are 0 (black)",
    "intra": "per query, logits of the segment's kind, a shot or a transition type ({n} classes; "
             "the last is padding)",
    "inter": "per query, logits of how the segment begins relative to the previous one ({n} classes; "
             "the last is padding)",
    "end": "per query, logits of the segment's end frame 0..{w} ({n} classes; the last is padding)",
}


def output_shape(classes, layout):
    return [1, classes, 1, QUERIES] if layout == "channels" else [QUERIES, classes]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_module(window64, precision, output_layout="channels"):
    """The export module of a float64 OmniShotCutWindow, as float32 PyTorch
    holding the model's weights: the folds (batch norms; for the fp16 model
    also input normalization and query scale) are computed in float64 and
    every tensor is rounded once, to float16 for the fp16 model and to float32
    for the fp32 model."""
    formulation = PRECISIONS[precision][0]
    module = build(window64, formulation, output_layout)
    storage = torch.float16 if precision == "fp16" else torch.float32
    return module.to(storage).to(torch.float32).eval()


def convert(module, precision, target, output_layout):
    """ML program of one export module, not compiled or loaded."""
    import coremltools as ct
    _, compute_precision, output_dtype, _ = PRECISIONS[precision]
    example = torch.zeros(WINDOW, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        traced = torch.jit.trace(module, (example,), check_trace=False)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=tuple(example.shape), dtype=np.float16)],
        outputs=[ct.TensorType(name=name, dtype=output_dtype) for name in CLASSES],
        convert_to="mlprogram",
        compute_precision=getattr(ct.precision, compute_precision),
        minimum_deployment_target=getattr(ct.target, target),
        skip_model_load=True,
    )


def describe(mlmodel, precision, output_layout, checkpoint_sha256):
    """Model description and the configuration the Swift loader reads."""
    formulation, _, _, units = PRECISIONS[precision]
    mlmodel.short_description = (
        f"OmniShotCut v1.5 shot boundary detection on one {WINDOW}-frame window: per-query logits of the "
        f"segment kind, the boundary relation and the end frame ({precision}, "
        f"{'Neural Engine' if formulation == 'ane' else 'GPU'} formulation). Decode with the argmax over "
        "every class but the last.")
    mlmodel.author = AUTHOR
    mlmodel.license = LICENSE
    mlmodel.version = VERSION
    mlmodel.input_description["frames"] = DESCRIPTIONS["frames"]
    shape = output_shape("C", output_layout)
    for name, classes in CLASSES.items():
        mlmodel.output_description[name] = (
            DESCRIPTIONS[name].format(n=classes, w=WINDOW) + f", shape {output_shape(classes, output_layout)}")
    mlmodel.user_defined_metadata.update({
        "window_length": str(WINDOW),
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "queries": str(QUERIES),
        "intra_classes": str(INTRA_CLASSES),
        "inter_classes": str(INTER_CLASSES),
        "end_classes": str(END_CLASSES),
        "precision": precision,
        "input_layout": "NCHW RGB 0..255",
        "output_layout": "[" + ",".join(map(str, shape)) + "]",
        "compute_units": units,
        "formulation": "neural-engine" if formulation == "ane" else "gpu",
        "checkpoint_sha256": checkpoint_sha256,
    })


def compile_to(package, destination):
    """`package` compiled to `destination` (.mlmodelc), replacing what is there."""
    from coremltools.models.utils import compile_model
    directory, name = os.path.split(os.path.abspath(destination))
    temporary = os.path.join(directory, f".{os.getpid()}.{name}")  # compile_model wants the .mlmodelc suffix
    shutil.rmtree(temporary, ignore_errors=True)
    compile_model(package, temporary)
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(temporary, destination)
    return destination


def size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return total / 1e6


def main(checkpoint, destination, precisions, target="macOS13", output_layout="channels", compile=True,
         any_checkpoint=False):
    checkpoint_sha256 = sha256(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        message = (f"{checkpoint} has SHA-256 {checkpoint_sha256}, not the v1.5 release's "
                   f"{CHECKPOINT_SHA256}")
        if not any_checkpoint:
            sys.exit(f"convert.py: {message}; --any-checkpoint converts it anyway")
        print(f"warning: {message}", file=sys.stderr, flush=True)
    window64, _ = load_official(checkpoint, torch.float64)
    os.makedirs(destination, exist_ok=True)
    for precision in precisions:
        formulation = PRECISIONS[precision][0]
        layout = output_layout if formulation == "ane" else "queries"
        started = time.perf_counter()
        mlmodel = convert(export_module(window64, precision, layout), precision, target, layout)
        describe(mlmodel, precision, layout, checkpoint_sha256)
        package = os.path.join(destination, f"{NAME}-{precision}.mlpackage")
        shutil.rmtree(package, ignore_errors=True)
        mlmodel.save(package)
        del mlmodel
        line = f"{package}: {size_mb(package):.0f} MB, {time.perf_counter() - started:.0f} s"
        if compile:
            compiled = compile_to(package, os.path.splitext(package)[0] + ".mlmodelc")
            line += f"; {compiled}: {size_mb(compiled):.0f} MB"
        print(line, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("destination")
    parser.add_argument("--precision", default="both", choices=["fp16", "fp32", "both"],
                        help="fp16: Neural Engine model; fp32: GPU model (default both)")
    parser.add_argument("--target", default="macOS13", choices=TARGETS,
                        help="minimum deployment target (default macOS13, the first with float16 inputs)")
    parser.add_argument("--output-layout", default="channels", choices=OUTPUT_LAYOUTS,
                        help="fp16 model's logits: channels [1, C, 1, 24] (default) or queries [24, C]")
    parser.add_argument("--no-compile", action="store_true", help="do not write the sibling .mlmodelc")
    parser.add_argument("--any-checkpoint", action="store_true",
                        help="convert a checkpoint whose SHA-256 is not the v1.5 release's")
    options = parser.parse_args()
    precisions = ["fp16", "fp32"] if options.precision == "both" else [options.precision]
    main(options.checkpoint, options.destination, precisions, options.target, options.output_layout,
         not options.no_compile, options.any_checkpoint)
