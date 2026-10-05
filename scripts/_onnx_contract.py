#!/usr/bin/env python3
"""Check an exported ONNX against the det_ros input/output contract.

Run by 05_export_rknn.sh BEFORE rknn-toolkit2 sees the model, so a wrong export
(stock ultralytics' single ``[1, 4+nc, N]`` head, a square 640x640 input) stops
at the cheap step with a clear message instead of producing a .rknn that
det_ros cannot read.

Contract (ros2_stack/src/det_ros, mower_docs/03-yolo-perception.md):

* input  ``images`` NCHW ``[1, 3, 640, 480]`` (640 tall, 480 wide)
* 9 outputs, per stride 8/16/32: ``box [1,64,H/s,W/s]`` (DFL, reg_max 16),
  ``cls [1,nc,H/s,W/s]`` (sigmoid inside the graph), ``score_sum [1,1,H/s,W/s]``
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

STRIDES = (8, 16, 32)
REG_MAX = 16


def expected_outputs(nc: int, height: int = 640, width: int = 480) -> list[list[int]]:
    out: list[list[int]] = []
    for s in STRIDES:
        gh, gw = height // s, width // s
        out += [[1, 4 * REG_MAX, gh, gw], [1, nc, gh, gw], [1, 1, gh, gw]]
    return out


def check_shapes(input_shape: Sequence[int], output_shapes: Sequence[Sequence[int]],
                 nc: int = 22, height: int = 640, width: int = 480) -> list[str]:
    """Return a list of contract violations (empty = OK)."""
    errors: list[str] = []
    want_in = [1, 3, height, width]
    if list(input_shape) != want_in:
        errors.append(f"input shape {list(input_shape)} != {want_in} "
                      f"(NCHW, {height} tall x {width} wide)")
    want_out = expected_outputs(nc, height, width)
    got = [list(s) for s in output_shapes]
    if len(got) != len(want_out):
        hint = ""
        if len(got) == 1:
            hint = (" — a single output is the stock ultralytics head; export with "
                    "the airockchip fork (format='rknn')")
        errors.append(f"{len(got)} output(s), expected 9 (box/cls/score_sum x 3 "
                      f"strides){hint}")
        return errors
    for i, (g, w) in enumerate(zip(got, want_out)):
        if g != w:
            errors.append(f"output {i} is {g}, expected {w}")
    return errors


def _dims(value_info) -> list[int]:
    return [d.dim_value if d.HasField("dim_value") else -1
            for d in value_info.type.tensor_type.shape.dim]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("onnx")
    ap.add_argument("--nc", type=int, default=22)
    ap.add_argument("--imgsz", default="640,480", help="H,W")
    args = ap.parse_args(argv)
    try:
        import onnx
    except ImportError:
        print("  SKIP: the 'onnx' package is not installed, cannot inspect the graph "
              "(pip install onnx).")
        return 3
    h, w = (int(v) for v in args.imgsz.split(","))
    model = onnx.load(args.onnx)
    init = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in init]
    if len(inputs) != 1:
        print(f"  FAIL: {len(inputs)} graph inputs, expected 1")
        return 1
    in_shape = _dims(inputs[0])
    out_shapes = [_dims(o) for o in model.graph.output]
    print(f"  input  {inputs[0].name}: {in_shape}")
    for o, shp in zip(model.graph.output, out_shapes):
        print(f"  output {o.name}: {shp}")
    errors = check_shapes(in_shape, out_shapes, args.nc, h, w)
    for e in errors:
        print(f"  FAIL: {e}")
    if errors:
        return 1
    print(f"  OK: input [1,3,{h},{w}], 9 outputs, {args.nc} class channels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
