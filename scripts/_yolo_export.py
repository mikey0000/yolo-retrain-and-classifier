#!/usr/bin/env python3
"""Library-API fallback for `yolo export format=onnx`, used by 05_export_rknn.sh.

Same role as `_yolo_train.py`: the `yolo` console script is the normal path,
this exists for environments where only the library is importable. Takes the
same ``key=value`` arguments as the CLI and forwards them to
``YOLO.export()``.
"""

from __future__ import annotations

import os
import sys


def cast(value: str):
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except ValueError:
            try:
                return [float(p) for p in parts]
            except ValueError:
                return parts
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main(argv: list[str]) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.  pip install ultralytics",
              file=sys.stderr)
        return 1

    args: dict[str, object] = {}
    for item in argv:
        if "=" not in item:
            print(f"ERROR: expected key=value, got {item!r}", file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        args[key.strip()] = cast(value.strip())

    weights = args.pop("model", None)
    if not weights:
        print("ERROR: model=<checkpoint> is required", file=sys.stderr)
        return 2

    # ultralytics writes the .onnx next to the checkpoint unless the cwd is
    # already the right place; 05_export_rknn.sh moves it if needed.
    os.chdir(os.path.dirname(os.path.abspath(str(weights))) or ".")
    YOLO(str(weights)).export(**args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
