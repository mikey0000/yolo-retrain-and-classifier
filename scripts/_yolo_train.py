#!/usr/bin/env python3
"""Library-API fallback for `yolo detect train`, used by 04_train.sh.

The `yolo` console script is the normal path. This exists only so training
still works when ultralytics is installed as a library in an environment where
the console script is not on PATH (a venv without the bin dir, a container, a
user install with --user).

It takes the same ``key=value`` arguments as the CLI and forwards them to
``YOLO.train()`` with the string values cast to the types ultralytics expects.
"""

from __future__ import annotations

import sys


def cast(value: str):
    """Turn a CLI string into the type ultralytics wants."""
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

    resume = args.pop("resume", None)
    weights = args.pop("model", None)
    if not weights:
        print("ERROR: model=<weights> is required", file=sys.stderr)
        return 2
    if resume:
        # resume=True model=<run>/weights/last.pt: the run dir holds every
        # other setting, so nothing else is forwarded.
        YOLO(str(weights)).train(resume=True)
        return 0
    YOLO(str(weights)).train(**args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
