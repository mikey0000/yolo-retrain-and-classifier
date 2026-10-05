#!/usr/bin/env python3
"""Build the int8 calibration set for 05_export_rknn.sh.

Two things matter and both are easy to get wrong:

1. **Which frames.** det_ros only ever sees the two OA cameras, so the
   calibration set is stratified by the *camera* recorded in the capture
   manifest (``manifest.jsonl``), weighted by ``export.rknn.calib_camera_weights``
   — not by parent directory, which in ``dataset/images/{train,val}`` is the
   split, not the camera.

2. **What they look like.** rknn-toolkit2 resizes calibration images straight
   to the model input, stretching a 1920x1080 frame into 480x640. det_ros never
   does that: ``mower_rknn.preprocess.letterbox`` scales by
   ``min(480/w, 640/h)``, centres the result and pads with black (0). So every
   calibration image is pre-letterboxed exactly like that into a temp dir, and
   the toolkit is handed images that are already 480x640.

Pure functions (``letterbox_geometry``, ``allocate_quotas``,
``select_calibration``, ``camera_for``) are unit-tested; ``main`` does the I/O.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# letterbox — mirrors ros2_stack/src/mower_rknn/mower_rknn/preprocess.py
# ---------------------------------------------------------------------------

def letterbox_geometry(iw: int, ih: int, w: int, h: int) -> tuple[int, int, int, int, float]:
    """``(new_w, new_h, pad_x, pad_y, scale)`` for fitting ``iw x ih`` into ``w x h``.

    Same arithmetic as ``mower_rknn.preprocess.letterbox``: scale by
    ``min(w/iw, h/ih)``, round the resized size, centre with floor-divided
    padding.
    """
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    return nw, nh, (w - nw) // 2, (h - nh) // 2, scale


def letterbox_image(img, w: int, h: int, pad: int = 0):
    """PIL image -> ``w x h`` RGB PIL image, letterboxed with ``pad`` (black)."""
    from PIL import Image

    img = img.convert("RGB")
    nw, nh, px, py, _scale = letterbox_geometry(img.width, img.height, w, h)
    canvas = Image.new("RGB", (w, h), (pad, pad, pad))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas


# ---------------------------------------------------------------------------
# camera attribution + stratified selection
# ---------------------------------------------------------------------------

def load_camera_map(frames_root: Path) -> dict[str, str]:
    """Image filename -> manifest ``camera`` for every session under frames_root."""
    out: dict[str, str] = {}
    if not frames_root.is_dir():
        return out
    for manifest in sorted(frames_root.rglob("manifest.jsonl")):
        with manifest.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = Path(str(rec.get("path", ""))).name
                if name and rec.get("camera"):
                    out[name] = str(rec["camera"])
    return out


def camera_for(path: Path, camera_map: Mapping[str, str]) -> str:
    """The manifest camera for ``path``, else ``unknown``.

    The parent directory is deliberately NOT used as a fallback for dataset
    images (it is ``train``/``val``). Frames captured by 01 sit in
    ``<session>/<camera>/``, and their manifest covers them.
    """
    return camera_map.get(path.name, "unknown")


def camera_shares(cameras: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    """Share per camera: named weights as given, ``'*'`` split evenly over the
    cameras not named. With no weights at all every camera counts equally."""
    if not weights:
        return {c: 1.0 for c in cameras}
    others = [c for c in cameras if c not in weights]
    star = float(weights.get("*", 0.0))
    return {c: (float(weights[c]) if c in weights else star / len(others))
            for c in cameras}


def allocate_quotas(available: Mapping[str, int], count: int,
                    weights: Mapping[str, float]) -> dict[str, int]:
    """How many calibration images to take from each camera.

    Only cameras that have frames take part, so an absent camera's share is
    redistributed over the others. A camera that runs out hands its unmet
    quota on to the cameras that still have frames, again by weight. A camera
    with weight 0 is never used. Total = ``min(count, frames available in
    cameras with weight > 0)``.
    """
    quota = {c: 0 for c in available}
    cams = sorted(c for c, n in available.items() if n > 0)
    shares = camera_shares(cams, weights)
    active = [c for c in cams if shares[c] > 0]
    remaining = min(max(count, 0), sum(available[c] for c in active))
    while remaining > 0 and active:
        wsum = sum(shares[c] for c in active)
        exact = {c: remaining * shares[c] / wsum for c in active}
        give = {c: int(exact[c]) for c in active}
        # largest remainder, ties broken by name so the result is deterministic
        for c in sorted(active, key=lambda c: (give[c] - exact[c], c))[:remaining - sum(give.values())]:
            give[c] += 1
        for c in active:
            take = min(give[c], available[c] - quota[c])
            quota[c] += take
            remaining -= take
        active = [c for c in active if quota[c] < available[c]]
    return quota


def select_calibration(by_camera: Mapping[str, Sequence[Path]], count: int,
                       weights: Mapping[str, float], seed: int = 1234) -> list[Path]:
    """Pick ``count`` images stratified by camera according to ``weights``."""
    quotas = allocate_quotas({c: len(v) for c, v in by_camera.items()}, count, weights)
    rng = random.Random(seed)
    picked: list[Path] = []
    for cam in sorted(by_camera):
        pool = sorted(by_camera[cam])
        rng.shuffle(pool)
        picked.extend(pool[:quotas.get(cam, 0)])
    return picked


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["dataset", "frames"], default="dataset")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--weights", default="{}", help="JSON camera -> share")
    ap.add_argument("--size", default="480,640", help="model input W,H")
    ap.add_argument("--out-dir", required=True, help="letterboxed copies go here")
    ap.add_argument("--list", required=True, help="calibration list file to write")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required to letterbox the calibration set. "
              "pip install pillow", file=sys.stderr)
        return 1

    frames_root = Path(args.frames_dir)
    root = Path(args.dataset_dir) / "images" if args.source == "dataset" else frames_root
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if root.is_dir() else []
    if not files:
        print(f"ERROR: no calibration images under {root}. Run 01_capture.py (and 03) "
              "first — int8 calibration has to come from these cameras, not COCO.",
              file=sys.stderr)
        return 1

    cam_map = load_camera_map(frames_root)
    by_camera: dict[str, list[Path]] = {}
    for p in files:
        by_camera.setdefault(camera_for(p, cam_map), []).append(p)
    weights = {str(k): float(v) for k, v in json.loads(args.weights or "{}").items()}
    picked = select_calibration(by_camera, args.count, weights, args.seed)

    w, h = (int(v) for v in args.size.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for i, src in enumerate(picked):
        try:
            with Image.open(src) as im:
                dst = out_dir / f"{i:05d}_{src.stem}.png"   # lossless: no extra JPEG loss
                letterbox_image(im, w, h).save(dst)
        except Exception as exc:  # unreadable frame: skip, report
            print(f"  skip {src}: {exc}", file=sys.stderr)
            continue
        written.append(str(dst.resolve()))

    list_path = Path(args.list)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(written) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    picked_set = set(picked)
    for cam, items in by_camera.items():
        counts[cam] = sum(1 for p in items if p in picked_set)
    print(f"calibration set: {len(written)} images, letterboxed to {w}x{h} (pad 0)")
    print(f"  list : {list_path}")
    print("  per camera: " + ", ".join(f"{c}={n}/{len(by_camera[c])}"
                                       for c, n in sorted(counts.items())))
    if "unknown" in by_camera:
        print(f"  NOTE: {len(by_camera['unknown'])} image(s) had no manifest camera "
              "(not captured by 01, or frames dir moved).")
    if len(written) < args.count:
        print(f"  NOTE: only {len(written)} of {args.count} requested images were available.")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
