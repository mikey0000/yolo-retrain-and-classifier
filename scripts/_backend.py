#!/usr/bin/env python3
"""Torch backend detection for 04_train.sh.

Prints one human line (``torch=... hip=... cuda_available=...``) and then
``backend=<name>`` on the last line, which the shell script reads. Exit 1 when
torch is not importable.

Backends:

* ``rocm``             HIP build of torch with a usable device (RX 7900 XTX).
* ``cuda``             CUDA build with a usable device (NVIDIA). ``hip`` is None
                       here, which is why "hip=None" must not mean "cpu".
* ``rocm-unavailable`` HIP build, but no device is visible (groups, HSA override).
* ``cpu``              no usable GPU at all.
"""

from __future__ import annotations

import sys


def classify_backend(cuda_available: bool, hip: str | None) -> str:
    """Decide the backend from ``torch.cuda.is_available()`` and ``torch.version.hip``.

    Device availability is checked FIRST: torch exposes ROCm devices through
    the ``torch.cuda`` API, so ``cuda_available`` is True for both CUDA and
    ROCm, and only then does ``hip`` tell the two apart.
    """
    if cuda_available:
        return "rocm" if hip else "cuda"
    if hip:
        return "rocm-unavailable"
    return "cpu"


def main() -> int:
    try:
        import torch
    except Exception:
        print("torch is not importable")
        print("backend=missing")
        return 1
    hip = getattr(torch.version, "hip", None)
    avail = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if avail else "-"
    print(f"torch={torch.__version__} hip={hip} cuda={getattr(torch.version, 'cuda', None)} "
          f"cuda_available={avail} devices={torch.cuda.device_count()} name={name}")
    print(f"backend={classify_backend(avail, hip)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
