#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnx"]
# ///
"""Make an upscaler's ONNX output dims independent of its input dims.

PyTorch exports of these models name the output's spatial dims with the same
dim_params as the input (`height`, `width`). ONNX Runtime treats equal
dim_params as equal sizes, so for a 2× model its memory planner allocates the
output at input size and the session crashes on the first run. Renaming every
output dim to `out_<axis>_<name>` tells ORT they are unknown and independent.

    uv run tools/patch-dynamic-shapes.py model.onnx [out.onnx]

Writes in place when no output path is given. Already-patched models are left
byte-identical.
"""
import sys

import onnx


def patch(model: onnx.ModelProto) -> int:
    renamed = 0
    for output in model.graph.output:
        for axis, dim in enumerate(output.type.tensor_type.shape.dim):
            if dim.dim_param and not dim.dim_param.startswith('out_'):
                dim.dim_param = f'out_{axis}_{dim.dim_param}'
                renamed += 1
    return renamed


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else src
    model = onnx.load(src)
    renamed = patch(model)
    if renamed or dst != src:
        onnx.save(model, dst)
    print(f'{src}: renamed {renamed} output dim(s)')
