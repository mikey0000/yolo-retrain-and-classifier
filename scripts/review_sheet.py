#!/usr/bin/env python3
"""review_sheet.py — one HTML page to eyeball the VLM's boxes against the onboard detector.

Run after 02, before 03. For every labelled frame it shows:

* the raw frame with the VLM's suggested boxes drawn on it (coordinates
  resolved exactly as 03 will: per-frame space, session consensus for
  ambiguous replies, pixel replies rescaled by original/sent size);
* next to it, the onboard detector's annotated frame captured at the same
  moment (01 ``reference:``), when there is one.

Where they disagree, one of them is wrong. The VLM missing something the
detector found, or boxing grass, is a label to fix by hand (03 ``review/``);
the detector missing what the VLM found correctly is exactly what retraining
is for.

Output: ``work/review_sheets/<session>.html``. Images are linked by relative
path, not embedded, so the page stays small; open it from Windows via
``\\\\wsl$\\<distro>\\...`` or ``explorer.exe work/review_sheets``.

    python3 scripts/review_sheet.py --images work/frames/<session>
    python3 scripts/review_sheet.py --images work/frames/<session> --only boxes
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LOG, add_common_args, cfg_path, get_dotted, image_size_bytes, load_config,
    read_jsonl, resolve, setup_logging,
)


def _load_converter():
    """03_to_yolo.py is not importable by name; load it for its coord helpers."""
    path = Path(__file__).resolve().parent / "03_to_yolo.py"
    spec = importlib.util.spec_from_file_location("to_yolo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["to_yolo"] = mod
    spec.loader.exec_module(mod)
    return mod


# Distinct, readable on grass and on dark frames.
_PALETTE = ["#ff3b30", "#ffcc00", "#34c759", "#0a84ff", "#ff2d92", "#64d2ff",
            "#bf5af2", "#ff9f0a", "#30d158", "#ffffff"]


def _rel(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()


def build_rows(conv, raw_records: list[dict], manifest: dict[str, dict],
               names: list[str], override: str | None, max_edge: int) -> list[dict]:
    """Per frame: image path, size, pixel boxes, reference path, flags."""
    sized = []
    for rec in raw_records:
        image = Path(rec.get("image_path") or "")
        if not image.is_absolute():
            image = resolve(image)
        w, h = int(rec.get("orig_width") or 0), int(rec.get("orig_height") or 0)
        if not (w and h) and image.is_file():
            w, h = image_size_bytes(image.read_bytes()[:65536])
        sw, sh, _ = conv.sent_size_of(rec, w, h, max_edge)
        sized.append((rec, image, w, h, sw, sh))

    session_space = conv.resolve_space_name(
        [conv.CoordSpace.classify(rec.get("boxes") or [], sw, sh)
         for rec, _i, _w, _h, sw, sh in sized if rec.get("boxes")])

    rows = []
    for rec, image, w, h, sw, sh in sized:
        boxes = rec.get("boxes") or []
        space_name, decided = conv.frame_space_name(boxes, sw, sh, override, session_space)
        drawn = []
        if boxes and w and h:
            space = conv.CoordSpace.for_frame(space_name, w, h, sw, sh)
            for b in boxes:
                coords = conv._as_list(b.get("box"))
                if len(coords) != 4:
                    continue
                x1, y1, x2, y2 = conv.to_pixel_box(coords, space)
                cls = str(b.get("class", "?"))
                drawn.append({
                    "class": cls, "conf": b.get("confidence"),
                    "xyxy": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                    "color": _PALETTE[names.index(cls) % len(_PALETTE)]
                    if cls in names else "#ffffff",
                })
        mrec = manifest.get(image.name) or {}
        ref = mrec.get("reference_path")
        rows.append({
            "stem": rec.get("stem") or image.stem, "image": image, "w": w, "h": h,
            "boxes": drawn, "ok": bool(rec.get("ok")), "error": rec.get("error") or "",
            "oov": rec.get("out_of_vocabulary") or [],
            "space": f"{space_name} ({decided})",
            "reference": resolve(ref) if ref else None,
            "reference_age_ms": mrec.get("reference_age_ms"),
            "reference_used": bool(rec.get("reference_used")),
            "camera": mrec.get("camera", ""),
        })
    return rows


def render(rows: list[dict], out: Path, title: str) -> str:
    base = out.parent
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font:14px system-ui,sans-serif;background:#111;color:#ddd;margin:16px}}
h1{{font-size:18px}} .row{{display:flex;gap:12px;margin:0 0 20px;align-items:flex-start}}
.cell{{flex:1;min-width:0}} .cell img,.cell svg{{width:100%;height:auto;display:block}}
.frame{{position:relative}} .frame svg{{position:absolute;inset:0;height:100%}}
.meta{{font-size:12px;color:#aaa;margin:4px 0}} .bad{{color:#ff6b6b}}
.none{{display:flex;align-items:center;justify-content:center;aspect-ratio:4/3;
background:#222;color:#777}} .tag{{display:inline-block;padding:0 6px;margin:2px;
border-radius:3px;color:#000;font-size:12px}}
</style></head><body><h1>{html.escape(title)}</h1>
<p class="meta">Left: raw frame + VLM suggestions (as 03 will read them). Right:
onboard detector output at the same moment. {len(rows)} frame(s).</p>"""]
    for r in rows:
        img = _rel(r["image"], base)
        svg = []
        for b in r["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            conf = f" {b['conf']:.2f}" if isinstance(b["conf"], (int, float)) else ""
            label = html.escape(b["class"] + conf)
            svg.append(
                f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}" '
                f'fill="none" stroke="{b["color"]}" stroke-width="{max(r["w"], 1) / 320:.1f}"/>'
                f'<text x="{x1 + 2:.1f}" y="{max(y1 - 4, 14):.1f}" fill="{b["color"]}" '
                f'font-size="{max(r["w"], 1) / 40:.0f}" stroke="#000" stroke-width="0.6" '
                f'paint-order="stroke">{label}</text>')
        tags = "".join(f'<span class="tag" style="background:{b["color"]}">'
                       f'{html.escape(b["class"])}</span>' for b in r["boxes"])
        status = ("" if r["ok"] else
                  f'<span class="bad">FAILED: {html.escape(r["error"][:120])}</span> ')
        if r["oov"]:
            status += f'<span class="bad">out-of-vocabulary: {html.escape(", ".join(r["oov"]))}</span> '
        if r["ok"] and not r["boxes"]:
            status += "negative (no boxes) "
        if r["reference"] and r["reference"].is_file():
            age = r["reference_age_ms"]
            ref_html = (f'<img loading="lazy" src="{html.escape(_rel(r["reference"], base))}">'
                        f'<div class="meta">onboard detector, {age} ms older'
                        f'{"; sent to VLM" if r["reference_used"] else "; not sent to VLM"}</div>')
        else:
            ref_html = '<div class="none">no reference frame</div>'
        parts.append(f"""<div class="row" id="{html.escape(r['stem'])}">
<div class="cell"><div class="frame"><img loading="lazy" src="{html.escape(img)}">
<svg viewBox="0 0 {r['w']} {r['h']}" preserveAspectRatio="none">{''.join(svg)}</svg></div>
<div class="meta">{html.escape(r['stem'])} · {html.escape(r['camera'])} · space {html.escape(r['space'])}</div>
<div class="meta">{status}{tags}</div></div>
<div class="cell">{ref_html}</div></div>""")
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--images", required=True, help="frame session dir from 01")
    parser.add_argument("--labels", help="label dir from 02 (default: labels_dir/<session>)")
    parser.add_argument("--out", help="HTML file (default: work/review_sheets/<session>.html)")
    parser.add_argument("--only", choices=["all", "boxes", "problems"], default="all",
                        help="boxes = frames with suggestions; problems = failed or OOV")
    parser.add_argument("--coord-space", help="override, as in 03 (0-1000 | 0-1 | pixels)")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config, args.overrides)
    conv = _load_converter()
    names = list(conv.class_names(cfg))

    images_dir = Path(args.images).expanduser()
    session = images_dir.name
    labels_dir = Path(args.labels).expanduser() if args.labels else \
        cfg_path(cfg, "paths.labels_dir") / session
    raw_dir = labels_dir / "raw"
    if not raw_dir.is_dir():
        LOG.error("no %s — run 02_llm_label.py first", raw_dir)
        return 2
    raw = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            raw.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("skipping %s: %s", path, exc)
    manifest = {Path(r.get("path", "")).name: r
                for r in read_jsonl(images_dir / "manifest.jsonl")}

    override = conv.named_space(args.coord_space or get_dotted(cfg, "convert.coord_space"))
    rows = build_rows(conv, raw, manifest, names, override,
                      int(get_dotted(cfg, "llm.max_edge", 1024)))
    if args.only == "boxes":
        rows = [r for r in rows if r["boxes"]]
    elif args.only == "problems":
        rows = [r for r in rows if not r["ok"] or r["oov"]]

    out = Path(args.out).expanduser() if args.out else \
        cfg_path(cfg, "paths.work_dir") / "review_sheets" / f"{session}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, out, f"review {session}"), encoding="utf-8")
    print(f"{len(rows)} frame(s) -> {out}")
    distro = os.environ.get("WSL_DISTRO_NAME")
    if distro:
        win = "\\\\wsl.localhost\\" + distro + str(out.resolve()).replace("/", "\\")
        print(f"open in Windows: {win}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
