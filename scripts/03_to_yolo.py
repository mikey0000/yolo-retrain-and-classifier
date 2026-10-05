#!/usr/bin/env python3
"""03_to_yolo.py — validate the LLM's suggestions and write a real YOLO dataset.

Input:  ``work/labels/<session>/raw/*.json`` from 02_llm_label.py
Output: a standard Ultralytics dataset directory plus ``data.yaml`` and an
        audit trail.

Why this step exists at all, given the LLM already returned JSON: because the
model's output is a *suggestion*, and a suggestion that is wrong in a small way
is worse than no label at all — it trains confidently in the wrong direction.
This step is the gate. It

* **closes the vocabulary.** A frame whose reply used a class outside the 22
  (``"robot"``, ``"bird"``; after 02's alias table) goes to ``review/``: the
  unnamed object is still in the picture and must not train as background.
* **normalises coordinates.** Models report boxes in 0-1000 normalised space,
  0-1 floats, or raw pixels, and sometimes xywh instead of xyxy. YOLO wants
  ``cx cy w h`` normalised to [0,1]. ``coord_space: auto`` classifies each
  response against the image size the LLM was *sent*; a frame that settles it
  on its own keeps its verdict, ambiguous frames take their session's majority.
  Scale factors are always per frame, and pixel replies are rescaled by
  original/sent size. Every decision is recorded in ``provenance.jsonl``.
* **rejects degenerate boxes.** A box covering 85 % of the frame, or one smaller
  than 0.05 % of it, is a hallucination, not an object.
* **routes uncertain frames to ``review/``** instead of the labels. A kept box
  in the low-confidence band, too many boxes, an unparseable reply — a human
  labels those. Under ``bad_box_policy: lenient`` a frame where some boxes were
  invalid keeps its valid boxes and is ACCEPTED (a "partial accept", noted in
  provenance); ``strict`` sends it to review instead.
* **keeps negatives.** ``{"boxes": []}`` produces an *empty* label file, which
  is a real and necessary training signal. A detector trained only on positives
  invents obstacles in clean grass.

Layout produced::

    work/dataset/
    ├── images/train/*.jpg      hardlinks (or symlinks/copies) — no duplicate bytes
    ├── images/val/*.jpg
    ├── labels/train/*.txt      "<class_id> cx cy w h", normalised
    ├── labels/val/*.txt
    ├── review/                 frames a human must label, with the reason;
    │                           hand labels go here too (<stem>.txt)
    ├── data.yaml
    └── provenance.jsonl        frame -> manifest, model, thresholds, coord space

Splits are by **session (and pose bucket, when the manifest has odom — 01 does
not write it), never by frame**: at 1 Hz the same rock appears in consecutive
frames, so a per-frame split puts near-identical images in both train and val
and the resulting mAP is fiction. ``images/`` and ``labels/`` are rebuilt from
scratch every run so a frame cannot linger in a split it no longer belongs to.

Examples
--------
    python3 scripts/03_to_yolo.py --labels work/labels/20261005-142530 \
                                  --images work/frames/20261005-142530

    # strict: any single bad box rejects the whole frame
    python3 scripts/03_to_yolo.py --labels ... --bad-box-policy strict

    # report only, write nothing
    python3 scripts/03_to_yolo.py --labels ... --dry-run

    # two sessions into one dataset (train/val split then spans both)
    python3 scripts/03_to_yolo.py --labels work/labels/A --labels work/labels/B \
        --images work/frames/A --images work/frames/B

    # once the review/ images are hand-labelled, fold them back in
    python3 scripts/03_to_yolo.py --labels ... --images ... --include-review
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("ERROR: pyyaml is required.  pip install pyyaml")

from common import (  # noqa: E402
    LOG, add_common_args, cfg_path, class_index, class_names, downscaled_size,
    file_sha256, get_dotted, list_images, load_config, now_iso, read_jsonl,
    setup_logging, write_json_atomic,
)


# ---------------------------------------------------------------------------
# coordinate normalisation
# ---------------------------------------------------------------------------

#: The prompt asks for 0-1000, so that is the fallback when the evidence is
#: genuinely balanced. Only the space NAME is resolved per session; the scale
#: factors are always built per frame from that frame's own sizes.
AMBIGUOUS = "ambiguous"
NORMALISED = "0-1000"
UNIT = "0-1"
PIXELS = "pixels"


@dataclasses.dataclass(frozen=True)
class CoordSpace:
    """How to turn one response's coordinates into ORIGINAL-frame pixels."""
    name: str
    scale_x: float
    scale_y: float

    @staticmethod
    def for_frame(name: str, width: int, height: int,
                  sent_width: int = 0, sent_height: int = 0) -> "CoordSpace":
        """Scale factors for one frame.

        ``width``/``height`` are the original frame; ``sent_*`` the image the
        LLM actually saw (02 downscales to ``llm.max_edge``). 0-1000 and 0-1
        are resolution-independent; pixel replies are in the SENT image's
        pixels and are rescaled by original/sent.
        """
        if name == NORMALISED:
            return CoordSpace(name, width / 1000.0, height / 1000.0)
        if name == UNIT:
            return CoordSpace(name, float(width), float(height))
        if name == PIXELS:
            sw, sh = sent_width or width, sent_height or height
            return CoordSpace(name, width / float(sw) if sw else 1.0,
                              height / float(sh) if sh else 1.0)
        raise ValueError(f"unknown coordinate space {name!r}")

    @staticmethod
    def classify(boxes: list[dict], width: int, height: int) -> str:
        """Decide what space ONE response is in, or return ``AMBIGUOUS``.

        ``width``/``height`` must be the size of the image the LLM was SENT,
        since that is the only frame a pixel reply can refer to.

        The prompt asks for 0-1000, and on a ~1000 px image that space and raw
        pixel space overlap numerically, so a single response often cannot
        settle it. Ambiguous frames take the session-level consensus
        (``resolve_space_name``); unambiguous frames keep their own verdict.

        Per-response rules, in order:

        * any value < 0                  -> pixels (signed/centred convention)
        * max <= 1                       -> 0-1 (pixels are never all <= 1 on a
                                            real frame)
        * any value > 1000               -> pixels (impossible in 0-1000)
        * any x > width or any y > height-> 0-1000, because a pixel coordinate
                                            cannot exceed its own frame
        * otherwise                      -> AMBIGUOUS (in range for both)
        """
        pairs = [(_as_list(b.get("box"))) for b in boxes]
        pairs = [p for p in pairs if len(p) == 4]
        if not pairs:
            return AMBIGUOUS
        xs = [v for p in pairs for v in (p[0], p[2])]
        ys = [v for p in pairs for v in (p[1], p[3])]

        if min(min(xs), min(ys)) < 0:
            return PIXELS
        if max(max(xs), max(ys)) <= 1.0 and width > 1:
            return UNIT
        if max(max(xs), max(ys)) > 1000.0:
            return PIXELS
        # Axis-aware bound: the strongest single-frame evidence available.
        if (width and max(xs) > width) or (height and max(ys) > height):
            return NORMALISED
        return AMBIGUOUS


def resolve_space_name(verdicts: Sequence[str]) -> str:
    """Turn a session's per-response verdicts into the space for its
    AMBIGUOUS frames. Only a name: every frame still gets its own scale.

    Frames that classify unambiguously outvote nothing but each other. Ties and
    an all-ambiguous session fall back to 0-1000, the space the prompt asked
    for — right unless the model ignored the instruction, and the report says so.
    """
    counts = collections.Counter(v for v in verdicts if v and v != AMBIGUOUS)
    if not counts:
        return NORMALISED
    ranked = counts.most_common()
    winner = ranked[0][0]
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        winner = NORMALISED if NORMALISED in {r[0] for r in ranked[:2]} else winner
    if len(counts) > 1:
        LOG.warning("coordinate space is split in this session: %s. Unambiguous "
                    "frames keep their own space; the %d ambiguous frame(s) are read "
                    "as %s. Re-check one frame by eye and pass --coord-space if this "
                    "is wrong.", dict(counts),
                    sum(1 for v in verdicts if v == AMBIGUOUS), winner)
    return winner


def _as_list(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    return []


def named_space(name: str | None) -> str | None:
    """The config/CLI override as a space name. None for ``auto`` = infer."""
    key = (name or "auto").lower()
    if key in ("auto", ""):
        return None
    if key in ("0-1000", "1000"):
        return NORMALISED
    if key in ("0-1", "1", "norm"):
        return UNIT
    if key in ("pixels", "px"):
        return PIXELS
    sys.exit(f"ERROR: unknown convert.coord_space: {name!r}")


def sent_size_of(record: dict, width: int, height: int,
                 max_edge: int) -> tuple[int, int, bool]:
    """``(sent_w, sent_h, inferred)`` for one raw reply.

    02 records ``sent_width``/``sent_height``. Raw files written before it did
    are inferred with the same downscale arithmetic 02 used (``llm.max_edge``),
    and flagged as inferred in provenance.
    """
    try:
        sw, sh = int(record.get("sent_width") or 0), int(record.get("sent_height") or 0)
    except (TypeError, ValueError):
        sw = sh = 0
    if sw > 0 and sh > 0:
        return sw, sh, False
    sw, sh = downscaled_size(width, height, max_edge)
    return sw, sh, True


def frame_space_name(boxes: list[dict], sent_w: int, sent_h: int,
                     override: str | None, session_name: str | None) -> tuple[str, str]:
    """``(space name, how it was decided)`` for one frame.

    Precedence: explicit override > this frame's own unambiguous verdict >
    the session consensus for ambiguous frames.
    """
    if override:
        return override, "override"
    verdict = CoordSpace.classify(boxes, sent_w, sent_h)
    if verdict != AMBIGUOUS:
        return verdict, "frame"
    if session_name:
        return session_name, "session"
    return AMBIGUOUS, "none"


def to_pixel_box(box: list[float], space: CoordSpace) -> list[float]:
    return [box[0] * space.scale_x, box[1] * space.scale_y,
            box[2] * space.scale_x, box[3] * space.scale_y]


# ---------------------------------------------------------------------------
# per-frame validation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameOutcome:
    stem: str
    image: Path | None
    decision: str                      # accept | review | skip
    reason: str = ""
    labels: list[str] = dataclasses.field(default_factory=list)
    boxes: list[tuple[int, float, float, float, float]] = dataclasses.field(
        default_factory=list)
    class_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    coord_space: str = ""
    coord_decided_by: str = ""         # override | frame | session
    sent_size: tuple[int, int] = (0, 0)
    sent_size_inferred: bool = False
    n_raw_boxes: int = 0
    dropped: list[str] = dataclasses.field(default_factory=list)
    negative: bool = False
    note: str = ""                     # e.g. "partial: ..." on a lenient accept
    source: str = "llm_suggestion"     # llm_suggestion | human (| pseudo, reserved)


def validate_frame(stem: str, image: Path | None, record: dict,
                   width: int, height: int, names: list[str],
                   idx: dict[str, int], cfg: dict,
                   session_space: str | None = None) -> FrameOutcome:
    """Validate one response against the thresholds in ``convert:``.

    ``width``/``height`` are the ORIGINAL frame size. ``session_space`` is the
    space name the session resolved for its ambiguous frames; this frame's own
    unambiguous verdict (judged against the size the LLM was sent) wins over
    it, and ``convert.coord_space`` wins over both.
    """
    min_conf = float(get_dotted(cfg, "convert.min_conf", 0.35))
    review_hi = float(get_dotted(cfg, "convert.review_conf_max", 0.55))
    max_boxes = int(get_dotted(cfg, "convert.max_boxes_per_frame", 12))
    max_area = float(get_dotted(cfg, "convert.max_box_area_frac", 0.85))
    min_area = float(get_dotted(cfg, "convert.min_box_area_frac", 0.0005))
    clamp = bool(get_dotted(cfg, "convert.clamp", True))
    policy = str(get_dotted(cfg, "convert.bad_box_policy", "lenient")).lower()
    keep_negatives = bool(get_dotted(cfg, "convert.keep_negatives", True))
    max_edge = int(get_dotted(cfg, "llm.max_edge", 1024) or 0)

    if not record.get("ok"):
        return FrameOutcome(stem, image, "review",
                            f"request failed: {record.get('error', 'unknown')}")

    boxes = record.get("boxes") or []
    oov = [str(n) for n in (record.get("out_of_vocabulary") or [])]
    if oov:
        # 02 strips out-of-vocabulary boxes but the object is still in the
        # picture. Accepting the rest (or an empty reply) as-is would train
        # that object as background, so a human decides.
        return FrameOutcome(stem, image, "review",
                            f"out-of-vocabulary class(es) in reply: "
                            f"{', '.join(sorted(set(oov)))}",
                            n_raw_boxes=len(boxes))

    outcome = FrameOutcome(stem, image, "accept", n_raw_boxes=len(boxes))

    if not boxes:
        if not keep_negatives:
            return FrameOutcome(stem, image, "skip", "empty reply, negatives disabled")
        outcome.negative = True
        outcome.reason = "negative (no objects from the vocabulary)"
        return outcome

    if len(boxes) > max_boxes:
        return FrameOutcome(stem, image, "review",
                            f"{len(boxes)} boxes in one frame (limit {max_boxes}) "
                            f"— likely a bad response, not a scene",
                            n_raw_boxes=len(boxes))

    if width == 0 or height == 0:
        return FrameOutcome(stem, image, "review",
                            "image size unknown (install Pillow) — cannot normalise",
                            n_raw_boxes=len(boxes))

    sent_w, sent_h, inferred = sent_size_of(record, width, height, max_edge)
    override = named_space(str(get_dotted(cfg, "convert.coord_space", "auto")))
    name, how = frame_space_name(boxes, sent_w, sent_h, override, session_space)
    if name == AMBIGUOUS:
        return FrameOutcome(
            stem, image, "review",
            "coordinate space could not be determined from this frame alone "
            "— pass --coord-space 0-1000 or 0-1 to settle it",
            n_raw_boxes=len(boxes))
    space = CoordSpace.for_frame(name, width, height, sent_w, sent_h)
    outcome.coord_space = space.name
    outcome.coord_decided_by = how
    outcome.sent_size = (sent_w, sent_h)
    outcome.sent_size_inferred = inferred

    any_bad = False
    low_conf = False
    counts: collections.Counter[str] = collections.Counter()

    for item in boxes:
        name = str(item.get("class", "")).strip()
        if name not in idx:
            outcome.dropped.append(f"class '{name}' not in vocabulary")
            any_bad = True
            continue
        coords = _as_list(item.get("box"))
        if len(coords) != 4:
            outcome.dropped.append(f"{name}: box was not 4 numbers")
            any_bad = True
            continue

        x1, y1, x2, y2 = to_pixel_box(coords, space)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        if clamp:
            x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
            y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
        elif x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            outcome.dropped.append(f"{name}: box outside the frame")
            any_bad = True
            continue

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            outcome.dropped.append(f"{name}: zero-area box")
            any_bad = True
            continue
        area_frac = (bw * bh) / float(width * height)
        if area_frac > max_area:
            outcome.dropped.append(f"{name}: box covers {area_frac:.0%} of the frame")
            any_bad = True
            continue
        if area_frac < min_area:
            outcome.dropped.append(f"{name}: box covers {area_frac:.3%} of the frame")
            any_bad = True
            continue

        conf = item.get("confidence")
        try:
            conf_val = float(conf) if conf is not None else 1.0
        except (TypeError, ValueError):
            conf_val = 1.0
        if conf_val < min_conf:
            outcome.dropped.append(f"{name}: confidence {conf_val:.2f} < {min_conf}")
            any_bad = True
            continue
        if conf_val < review_hi:
            low_conf = True

        cx = (x1 + bw / 2.0) / width
        cy = (y1 + bh / 2.0) / height
        outcome.boxes.append((idx[name], cx, cy, bw / width, bh / height))
        counts[name] += 1

    def to_review(reason: str) -> FrameOutcome:
        return FrameOutcome(stem, image, "review", reason,
                            n_raw_boxes=outcome.n_raw_boxes, coord_space=space.name,
                            coord_decided_by=how, sent_size=(sent_w, sent_h),
                            sent_size_inferred=inferred, dropped=outcome.dropped)

    if any_bad and policy == "strict":
        return to_review("strict policy: " + "; ".join(outcome.dropped[:4]))
    if any_bad and not outcome.boxes:
        return to_review("no usable box survived: " + "; ".join(outcome.dropped[:4]))
    if low_conf:
        # A KEPT box in the uncertain band: the label file would assert it.
        return to_review(f"at least one kept box in the {min_conf}-{review_hi} "
                         f"confidence band")
    if any_bad:
        # lenient: keep the frame with its valid boxes; the drop is a note,
        # not a reason to withhold the frame.
        outcome.note = "partial: " + "; ".join(outcome.dropped[:3])

    outcome.class_counts = dict(counts)
    outcome.labels = [
        f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in outcome.boxes]
    return outcome


# ---------------------------------------------------------------------------
# hand labels (the review/ queue, folded back in with --include-review)
# ---------------------------------------------------------------------------

#: Checked in this order for each stem. ``<stem>.txt`` is what CVAT's and
#: labelme's YOLO exporters write; ``<stem>.label.txt`` is this pipeline's own
#: convention; ``<stem>.json`` uses the LLM prompt's schema.
HAND_LABEL_SUFFIXES = (".label.txt", ".txt", ".json")
#: Files a YOLO export drops next to the labels that are not labels.
_NOT_LABELS = {"classes.txt", "obj.names", "obj.data", "train.txt", "valid.txt",
               "val.txt", "test.txt"}
#: Class-name files a YOLO export writes; when present their order defines the
#: ids in the .txt files and is remapped onto the vocabulary by NAME.
_CLASS_FILES = ("classes.txt", "obj.names")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def hand_label_stem(path: Path) -> str | None:
    """The frame stem a review-dir file labels, or None if it is not a label."""
    name = path.name
    if name in _NOT_LABELS or name.endswith(".reason.json"):
        return None
    for suffix in HAND_LABEL_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] or None
    return None


def review_class_remap(review_dir: Path, names: Sequence[str]) -> tuple[dict[int, int] | None, str]:
    """``{exported id: vocabulary id}`` from a classes.txt/obj.names, if present.

    ``(None, "")`` when there is no class file (ids are taken as vocabulary
    ids). ``(None, error)`` when the file names a class outside the vocabulary.
    """
    idx = class_index(names)
    for fname in _CLASS_FILES:
        path = review_dir / fname
        if not path.is_file():
            continue
        exported = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
        unknown = [n for n in exported if n not in idx]
        if unknown:
            return None, f"{path.name} names classes outside the vocabulary: {unknown}"
        return {i: idx[n] for i, n in enumerate(exported)}, ""
    return None, ""


def parse_yolo_hand_label(text: str, nc: int,
                          remap: dict[int, int] | None = None) -> tuple[list[str], list[str]]:
    """``(rows, errors)`` for a YOLO ``.txt`` hand label.

    Each row is ``<class_id> cx cy w h`` normalised. Ids are vocabulary ids
    unless ``remap`` (from the export's classes.txt) says otherwise. Any bad
    row is an error: a hand label is ground truth, so it is never repaired.
    """
    rows: list[str] = []
    errors: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {n}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"line {n}: class id must be an integer and the box 4 "
                          f"numbers (got {parts[0]!r} — class NAMES are only "
                          f"accepted in the .json form)")
            continue
        if remap is not None:
            if cid not in remap:
                errors.append(f"line {n}: class id {cid} not in the export's class file")
                continue
            cid = remap[cid]
        if not 0 <= cid < nc:
            errors.append(f"line {n}: class id {cid} out of range 0..{nc - 1}")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            errors.append(f"line {n}: box {cx} {cy} {bw} {bh} is not normalised to [0,1]")
            continue
        rows.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return rows, errors


def parse_json_hand_label(data: object, idx: dict[str, int]) -> tuple[list[str], list[str]]:
    """``(rows, errors)`` for a hand label in the prompt's JSON schema:
    ``{"boxes": [{"class": "<name>", "box": [x1,y1,x2,y2]}]}`` in 0-1000.

    ``class`` is mapped name -> id through the vocabulary; ``class_id`` may be
    given instead. Unknown names and out-of-range ids are errors, never 0.
    """
    nc = len(idx)
    if not isinstance(data, dict) or not isinstance(data.get("boxes"), list):
        return [], ["expected an object with a 'boxes' list"]
    rows: list[str] = []
    errors: list[str] = []
    for n, item in enumerate(data["boxes"]):
        if not isinstance(item, dict):
            errors.append(f"box {n}: not an object")
            continue
        if "class_id" in item:
            try:
                cid = int(item["class_id"])
            except (TypeError, ValueError):
                errors.append(f"box {n}: class_id {item['class_id']!r} is not an integer")
                continue
        else:
            name = str(item.get("class", "")).strip()
            if name not in idx:
                errors.append(f"box {n}: class {name!r} is not in the vocabulary")
                continue
            cid = idx[name]
        if not 0 <= cid < nc:
            errors.append(f"box {n}: class id {cid} out of range 0..{nc - 1}")
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in item.get("box", [])]
        except (TypeError, ValueError):
            errors.append(f"box {n}: 'box' must be 4 numbers")
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 < 0 or y1 < 0 or x2 > 1000 or y2 > 1000 or x2 == x1 or y2 == y1:
            errors.append(f"box {n}: {item.get('box')} is not a 0-1000 xyxy box")
            continue
        cx, cy = (x1 + x2) / 2000.0, (y1 + y2) / 2000.0
        bw, bh = (x2 - x1) / 1000.0, (y2 - y1) / 1000.0
        rows.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return rows, errors


def find_hand_label(review_dirs: Sequence[Path], stem: str) -> Path | None:
    for base in review_dirs:
        for suffix in HAND_LABEL_SUFFIXES:
            path = base / f"{stem}{suffix}"
            if path.is_file():
                return path
    return None


def review_image(review_dirs: Sequence[Path], stem: str) -> Path | None:
    """The frame copy 03 put in review/, used when the frame dir is gone."""
    for base in review_dirs:
        for ext in IMAGE_EXTS:
            path = base / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def load_hand_label(path: Path, stem: str, image: Path | None,
                    names: Sequence[str]) -> FrameOutcome:
    """A hand label -> an accepted, human-sourced outcome, or a review row
    explaining why it was rejected. An empty file is a human-confirmed
    negative (CVAT writes one for a frame with no annotations)."""
    idx = class_index(names)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return FrameOutcome(stem, image, "review", f"hand label unreadable: {exc}",
                            source="human")
    if path.name.endswith(".json"):
        try:
            rows, errors = parse_json_hand_label(json.loads(text), idx)
        except json.JSONDecodeError as exc:
            rows, errors = [], [f"invalid JSON: {exc}"]
    else:
        remap, err = review_class_remap(path.parent, names)
        rows, errors = ([], [err]) if err else parse_yolo_hand_label(text, len(names), remap)
    if errors:
        return FrameOutcome(stem, image, "review",
                            f"hand label rejected ({path.name}): " + "; ".join(errors[:3]),
                            source="human")
    if image is None:
        return FrameOutcome(stem, None, "review",
                            f"hand label {path.name} has no matching image", source="human")
    counts = collections.Counter(names[int(r.split()[0])] for r in rows)
    return FrameOutcome(stem, image, "accept", "human label", labels=rows,
                        class_counts=dict(counts), negative=not rows,
                        coord_space="yolo", source="human")


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------

def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Hardlink when possible: the frames already occupy disk, copying doubles it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            LOG.debug("hardlink failed for %s (%s); falling back to symlink", src, exc)
    if mode in ("hardlink", "symlink"):
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError as exc:
            LOG.debug("symlink failed for %s (%s); falling back to copy", src, exc)
    shutil.copy2(src, dst)


def pose_bucket(manifest_rec: dict | None, cell_m: float) -> str:
    """Bucket key for the split: session + coarse pose, so neighbouring frames
    of the same scene cannot straddle train and val."""
    if not manifest_rec:
        return "session-only"
    odom = manifest_rec.get("odom")
    if isinstance(odom, (list, tuple)) and len(odom) >= 2:
        try:
            return f"{int(float(odom[0]) / cell_m)}_{int(float(odom[1]) / cell_m)}"
        except (TypeError, ValueError):
            pass
    return "session-only"


def write_data_yaml(out_dir: Path, names: list[str], cfg: dict) -> Path:
    """Ultralytics data.yaml. `path` is the dataset root, so the folder moves.

    `names` is emitted as a YAML *list*, not a mapping. A mapping with trailing
    commas and inline comments parses as a list of single-key dicts, which
    ultralytics rejects — and it fails at the first `yolo train`, long after you
    think the dataset is fine.
    """
    path = out_dir / "data.yaml"
    critical = set(get_dotted(cfg, "classes.recall_critical", []) or [])
    lines = [
        "# Generated by 03_to_yolo.py — do not hand-edit names or their order.",
        "# The 22 names and their order are the vendor detector's ABI; see README.",
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        # Quote anything with a space so YAML cannot reinterpret it.
        value = f"'{name}'" if (" " in name or "-" in name) else name
        comment = "  # recall-critical" if name in critical else ""
        lines.append(f"  - {value}{comment}  # {i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify what we just wrote: a data.yaml that does not parse is the single
    # most annoying failure in this pipeline, so never trust the emitter.
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    got = written.get("names")
    if not isinstance(got, list) or len(got) != len(names) or list(got) != list(names):
        raise SystemExit(
            f"ERROR: generated data.yaml does not parse correctly.\n"
            f"  expected a {len(names)}-name list, got: {got!r}\n"
            f"  file: {path}")
    return path


def is_hand_label_file(path: Path) -> bool:
    return hand_label_stem(path) is not None or path.name in _CLASS_FILES


def prepare_dataset_dir(out_dir: Path, full_clean: bool) -> None:
    """Empty everything 03 generates, never a human's work.

    ``images/`` and ``labels/`` are ALWAYS emptied: a frame that moved from
    train to val (or to review) between runs must not survive in its old split,
    which is exactly the leakage the bucket split exists to prevent. In
    ``review/`` the generated ``*.reason.json`` files and frame copies are
    removed, but hand labels (``*.txt``, ``*.label.txt``, hand ``*.json``, the
    export's classes file) and the frame copies they label are kept.
    ``full_clean`` (--clean) additionally removes every other file in the
    dataset root.
    """
    for sub in ("images", "labels"):
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
    review = out_dir / "review"
    if review.is_dir():
        labelled = {hand_label_stem(p) for p in review.iterdir() if p.is_file()} - {None}
        for path in review.iterdir():
            if not path.is_file() and not path.is_symlink():
                continue
            if is_hand_label_file(path):
                continue
            if path.suffix.lower() in IMAGE_EXTS and path.stem in labelled:
                continue
            path.unlink()
    if full_clean and out_dir.is_dir():
        for path in out_dir.iterdir():
            if path.name == "review":
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def assemble(args, cfg: dict, names: list[str], outcomes: list[FrameOutcome],
             manifests: dict[str, dict],
             models_by_stem: dict[str, str] | None = None) -> dict:
    models_by_stem = models_by_stem or {}
    out_dir = Path(args.dataset).expanduser() if args.dataset else \
        cfg_path(cfg, "convert.dataset_dir")
    prepare_dataset_dir(out_dir, full_clean=bool(args.clean))
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    accepted = [o for o in outcomes if o.decision == "accept" and o.image]
    reviews = [o for o in outcomes if o.decision == "review"]

    # --- split -------------------------------------------------------------
    val_fraction = float(get_dotted(cfg, "convert.val_fraction", 0.2))
    split_by = str(get_dotted(cfg, "convert.split_by", "session_pose")).lower()
    cell_m = float(get_dotted(cfg, "convert.pose_bucket_m", 0.5))
    seed = int(get_dotted(cfg, "convert.seed", 1234))

    def bucket_of(o: FrameOutcome) -> str:
        rec = manifests.get(o.image.name if o.image else "")
        base = (rec or {}).get("session", "unknown-session")
        if split_by.startswith("frame"):
            return o.stem
        if split_by == "session":
            return base
        return f"{base}#{pose_bucket(rec, cell_m)}"

    buckets = sorted({bucket_of(o) for o in accepted})
    split_note = ""
    if len(buckets) < 2 and accepted:
        # Every frame is in one bucket (one session, no pose data). Splitting by
        # bucket would put 100 % of the data in val and train on nothing, which
        # is worse than the leakage we are trying to avoid: fall back to a
        # per-frame split and say plainly what that costs.
        split_note = (
            f"only {len(buckets)} bucket(s) available for {len(accepted)} frames, "
            f"so the split fell back to per-frame. That can put near-identical "
            f"consecutive frames in both train and val, which inflates mAP. "
            f"Capture more sessions, or add odom to the manifest to enable real "
            f"pose buckets.")
        LOG.warning(split_note)
        split_by = "frame (fallback: too few buckets)"
        buckets = sorted({o.stem for o in accepted})

    rng = random.Random(seed)
    rng.shuffle(buckets)
    n_val = max(1, int(round(len(buckets) * val_fraction))) if buckets else 0
    # Never leave train empty: with few buckets, cap val so at least one bucket
    # (and therefore at least one frame) stays on the training side.
    if buckets and n_val >= len(buckets):
        n_val = len(buckets) - 1
    val_buckets = set(buckets[:n_val])

    mode = str(get_dotted(cfg, "convert.link_mode", "hardlink"))
    counts = collections.Counter()
    written = 0
    provenance: list[dict] = []

    for outcome in accepted:
        split = "val" if bucket_of(outcome) in val_buckets else "train"
        img_dst = out_dir / "images" / split / outcome.image.name
        lbl_dst = out_dir / "labels" / split / f"{outcome.stem}.txt"
        link_or_copy(outcome.image, img_dst, mode)
        # An empty label file is meaningful: it marks a negative sample.
        lbl_dst.write_text(
            ("\n".join(outcome.labels) + "\n") if outcome.labels else "", encoding="utf-8")
        counts[f"{split}_images"] += 1
        if outcome.labels:
            counts[f"{split}_labels"] += 1
        else:
            counts["negatives"] += 1
        for name, n in outcome.class_counts.items():
            counts[f"cls::{name}"] += n
        written += 1

        rec = manifests.get(outcome.image.name, {})
        provenance.append({
            "image": outcome.image.name,
            "split": split,
            "bucket": bucket_of(outcome),
            "stem": outcome.stem,
            "labels": len(outcome.labels),
            "classes": outcome.class_counts,
            "coord_space": outcome.coord_space,
            "coord_decided_by": outcome.coord_decided_by,
            "sent_size": list(outcome.sent_size),
            "sent_size_inferred": outcome.sent_size_inferred,
            "n_raw_boxes": outcome.n_raw_boxes,
            "dropped": outcome.dropped,
            "note": outcome.note,
            "negative": outcome.negative,
            "decision": outcome.decision,
            # llm_suggestion | human. "pseudo" is reserved for teacher-model
            # labels, which this pipeline does not produce (see README).
            "source": outcome.source,
            # The labelling model lives in the raw reply, not the capture manifest.
            "source_model": (rec.get("model") or rec.get("labelled_model")
                             or models_by_stem.get(outcome.stem)),
            "camera": rec.get("camera"),
            "topic": rec.get("topic"),
            "session": rec.get("session"),
            "stamp_ns": rec.get("stamp_ns"),
            "image_sha256_64k": file_sha256(outcome.image, limit=65536),
            "built_at": now_iso(),
        })

    # --- review queue ------------------------------------------------------
    review_rows = []
    for outcome in reviews:
        row = {
            "stem": outcome.stem,
            "reason": outcome.reason,
            "image": str(outcome.image) if outcome.image else None,
            "n_raw_boxes": outcome.n_raw_boxes,
            "coord_space": outcome.coord_space,
            "dropped": outcome.dropped,
            "source": outcome.source,
        }
        (out_dir / "review" / f"{outcome.stem}.reason.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8")
        dst = out_dir / "review" / outcome.image.name if outcome.image else None
        if outcome.image and dst is not None and outcome.image.resolve() != dst.resolve():
            link_or_copy(outcome.image, dst, mode)
        review_rows.append(row)

    data_yaml = write_data_yaml(out_dir, names, cfg)
    prov_path = out_dir / "provenance.jsonl"
    with prov_path.open("w", encoding="utf-8") as fh:
        for row in provenance:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "built_at": now_iso(),
        "dataset_dir": str(out_dir.resolve()),
        "data_yaml": str(data_yaml),
        "n_classes": len(names),
        "class_names": names,
        "split_by": split_by,
        "val_fraction": val_fraction,
        "seed": seed,
        "buckets_total": len(buckets),
        "buckets_val": len(val_buckets),
        "split_note": split_note,
        "link_mode": mode,
        "counts": dict(counts),
        "review": len(reviews),
        "thresholds": {
            "min_conf": get_dotted(cfg, "convert.min_conf", 0.35),
            "review_conf_max": get_dotted(cfg, "convert.review_conf_max", 0.55),
            "max_boxes_per_frame": get_dotted(cfg, "convert.max_boxes_per_frame", 12),
            "max_box_area_frac": get_dotted(cfg, "convert.max_box_area_frac", 0.85),
            "coord_space": get_dotted(cfg, "convert.coord_space", "auto"),
            "bad_box_policy": get_dotted(cfg, "convert.bad_box_policy", "lenient"),
        },
        "n_images_written": written,
    }
    write_json_atomic(out_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(outcomes: list[FrameOutcome], summary: dict | None, names: list[str],
           cfg: dict) -> int:
    """Print the human summary. Returns a shell-friendly exit status."""
    accept = [o for o in outcomes if o.decision == "accept"]
    review = [o for o in outcomes if o.decision == "review"]
    skip = [o for o in outcomes if o.decision == "skip"]
    negatives = [o for o in accept if o.negative]
    partial = [o for o in accept if o.dropped]

    print()
    print("=== validation summary ===")
    print(f"frames seen      : {len(outcomes)}")
    print(f"  accepted       : {len(accept)}  ({len(negatives)} negatives)")
    print(f"  review queue   : {len(review)}")
    print(f"  skipped        : {len(skip)}")
    print(f"  partial accepts: {len(partial)}  (some boxes dropped, rest kept)")
    human = [o for o in accept if o.source == "human"]
    if human:
        print(f"  hand labels    : {len(human)}  (from review/, used instead of the LLM)")

    reasons = collections.Counter()
    for o in review:
        head = o.reason.split(":")[0].split("(")[0].strip()
        reasons[head or "unknown"] += 1
    if reasons:
        print("\nreview reasons:")
        for reason, n in reasons.most_common():
            print(f"  {n:>5}  {reason}")

    spaces = collections.Counter(
        (o.coord_space, o.coord_decided_by) for o in accept
        if o.coord_space and o.source != "human")
    if spaces:
        print("\ncoordinate space used (space, decided by):")
        for (space, how), n in spaces.most_common():
            print(f"  {n:>5}  {space:<8} {how}")
        print("         Verify one frame by eye. A wrong space scales every box "
              "and the model\n         still trains 'fine' — pass --coord-space to "
              "override.")

    # per-class counts
    cls_counts = collections.Counter()
    for o in accept:
        cls_counts.update(o.class_counts)
    print("\nper-class box counts (dataset is NOT balanced by design —"
          " recall-critical\nclasses need more examples than exotic ones):")
    critical = set(get_dotted(cfg, "classes.recall_critical", []) or [])
    for i, name in enumerate(names):
        n = cls_counts.get(name, 0)
        mark = "  <-- recall-critical, 0 is a problem" if name in critical and n == 0 else ""
        bar = "#" * min(n, 40)
        print(f"  {i:>2} {name:<14} {n:>6}  {bar}{mark}")

    missing = [n for i, n in enumerate(names) if cls_counts.get(n, 0) == 0]
    if missing:
        print(f"\nWARNING: {len(missing)} class(es) have zero examples: "
              f"{', '.join(missing)}")
        print("         Either the LLM never suggested them or they are genuinely "
              "absent from\n         these frames. A detector with a class it never "
              "saw will not detect it.\n         Add frames containing them, or "
              "confirm the class is not needed.")

    if summary:
        print()
        print("=== dataset written ===")
        print(f"dir      : {summary['dataset_dir']}")
        print(f"data.yaml: {summary['data_yaml']}")
        c = summary["counts"]
        print(f"train    : {c.get('train_images', 0)} images")
        print(f"val      : {c.get('val_images', 0)} images "
              f"({summary['buckets_val']}/{summary['buckets_total']} buckets, "
              f"split by {summary['split_by']})")
        if summary.get("split_note"):
            print(f"          NOTE: {summary['split_note']}")
        print(f"negatives: {c.get('negatives', 0)}")
        print(f"review   : {c.get('review', summary['review'])} frames in review/")
        try:
            import subprocess  # noqa: WPS433
            du = subprocess.run(["du", "-sh", str(summary["dataset_dir"])],
                                capture_output=True, text=True, check=False)
            if du.returncode == 0:
                print(f"size     : {du.stdout.split()[0]}")
        except (OSError, IndexError):
            pass

    print()
    if review:
        print(f"Before training: open {summary['dataset_dir'] + '/review' if summary else 'review/'} "
              f"({len(review)} frames)")
        print("  Label them by hand (labelme / CVAT), or drop them. Hand-labelled")
        print("  frames are ground truth; LLM suggestions are not.")
        print("  Re-run 03 with --include-review to fold them back in.")
    if not summary:
        print("dry run — nothing written. Drop --dry-run to build the dataset.")
        return 0
    if summary["counts"].get("train_images", 0) == 0:
        print("ERROR: no training images. There is nothing to train on — check the "
              "review\n       queue and the labels/frames dirs before running 04.")
        return 1
    if summary["counts"].get("val_images", 0) == 0:
        print("WARNING: no validation images. mAP cannot be computed, so you have "
              "no\n         signal on whether the model is worth deploying. Capture "
              "another session.")
    print(f"next: bash scripts/04_train.sh --data {summary['data_yaml']}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate LLM box suggestions and write a YOLO dataset "
                    "with data.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.")
    add_common_args(parser)

    # Either per-session (--labels/--images, repeatable) or the roots of both
    # trees (--labels-root/--images-root), which is how you build one dataset
    # from several capture sessions.
    parser.add_argument("--labels", action="append", default=[],
                        metavar="DIR",
                        help="labels dir from 02 (raw/ inside it); repeatable")
    parser.add_argument("--images", action="append", default=[],
                        metavar="DIR",
                        help="image dir from 01; repeatable, matched to --labels "
                             "by session name")
    parser.add_argument("--dataset", help="output dataset dir "
                                          "(default: convert.dataset_dir)")
    parser.add_argument("--labels-root",
                        help="the labels root: every session dir under it is used")
    parser.add_argument("--images-root",
                        help="the frames root: every session dir under it is used")
    parser.add_argument("--bad-box-policy", choices=["strict", "lenient"],
                        help="override convert.bad_box_policy")
    parser.add_argument("--val-fraction", type=float, help="override convert.val_fraction")
    parser.add_argument("--coord-space", choices=["auto", "0-1000", "0-1", "pixels"],
                        help="override convert.coord_space")
    parser.add_argument("--include-review", action="store_true",
                        help="use hand labels from <dataset>/review/ (<stem>.txt as "
                             "exported by CVAT/labelme in YOLO format, "
                             "<stem>.label.txt, or <stem>.json in the prompt schema) "
                             "in place of the LLM suggestion")
    parser.add_argument("--clean", action="store_true",
                        help="also remove every other generated file in the dataset "
                             "dir (images/ and labels/ are always rebuilt; hand "
                             "labels in review/ are never deleted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report, write nothing")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config, args.overrides)
    if args.bad_box_policy:
        cfg.setdefault("convert", {})["bad_box_policy"] = args.bad_box_policy
    if args.val_fraction is not None:
        cfg.setdefault("convert", {})["val_fraction"] = args.val_fraction
    if args.coord_space:
        cfg.setdefault("convert", {})["coord_space"] = args.coord_space

    names = class_names(cfg)
    idx = class_index(names)

    label_dirs = [Path(p).expanduser() for p in args.labels]
    image_dirs = [Path(p).expanduser() for p in args.images]
    if args.labels_root:
        root = Path(args.labels_root).expanduser()
        label_dirs = sorted(p for p in root.iterdir()
                            if p.is_dir() and (p / "raw").is_dir())
    if args.images_root:
        root = Path(args.images_root).expanduser()
        image_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    if not label_dirs or not image_dirs:
        LOG.error("need at least one of --labels/--labels-root and "
                  "--images/--images-root")
        return 2

    # Match label session -> image session by name; fall back to a global pool.
    images_by_session: dict[str, list[Path]] = {}
    all_images: list[Path] = []
    for d in image_dirs:
        if not d.is_dir():
            LOG.error("image dir not found: %s", d)
            return 2
        images_by_session.setdefault(d.name, []).extend(list_images(d))
        all_images.extend(list_images(d))

    manifests: dict[str, dict] = {}
    for d in image_dirs:
        manifests.update(_read_manifest(d))

    # A filename -> path map across every image dir, so a raw reply can be tied
    # to its frame regardless of which session dir it landed in.
    path_by_name = {p.name: p for p in all_images}
    by_stem = collections.defaultdict(list)
    for p in all_images:
        by_stem[p.stem].append(p)

    # image dimensions, read once per file
    size_cache: dict[Path, tuple[int, int]] = {}

    def dims_for(path: Path) -> tuple[int, int]:
        if path not in size_cache:
            size_cache[path] = _jpeg_size(path)
        return size_cache[path]

    dataset_dir = Path(args.dataset).expanduser() if args.dataset else \
        cfg_path(cfg, "convert.dataset_dir")
    # Where hand labels are looked for: the review queue 03 itself writes
    # (dataset/review), plus <labels>/review for anyone who labelled there.
    review_dirs = [dataset_dir / "review"] + [d / "review" for d in label_dirs]
    review_dirs = [d for d in dict.fromkeys(review_dirs) if d.is_dir()]

    def image_for(stem: str, session: str | None = None) -> Path | None:
        candidates = by_stem.get(stem) or []
        # Prefer the image in the same session, else any matching stem, else
        # the copy in review/ (the frames dir may have been pruned).
        same = next((p for p in candidates if session and p.parent.parent.name == session),
                    None)
        return same or (candidates[0] if candidates else None) or \
            review_image(review_dirs, stem)

    outcomes: list[FrameOutcome] = []
    seen_stems: set[str] = set()
    # label session -> [(stem, image, record, width, height)] for every reply
    # that has a frame, held until that session's ambiguous-frame space is known.
    pending: dict[str, list[tuple[str, Path, dict, int, int]]] = {}

    models_by_stem: dict[str, str] = {}

    for label_dir in label_dirs:
        raw_dir = label_dir / "raw"
        if not raw_dir.is_dir():
            LOG.warning("no raw/ under %s — skipping", label_dir)
            continue
        session_images = images_by_session.get(label_dir.name, [])
        LOG.info("session %s: %d raw replies, %d frames available",
                 label_dir.name, len(list(raw_dir.glob('*.json'))), len(session_images))

        for raw_path in sorted(raw_dir.glob("*.json")):
            stem = raw_path.stem
            if stem in seen_stems:
                continue
            try:
                record = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOG.warning("unreadable %s: %s", raw_path, exc)
                continue
            seen_stems.add(stem)
            if record.get("model"):
                models_by_stem[stem] = str(record["model"])

            image = image_for(stem, label_dir.name)

            if args.include_review:
                hand_path = find_hand_label(review_dirs, stem)
                if hand_path is not None:
                    outcomes.append(load_hand_label(hand_path, stem, image, names))
                    continue

            if image is None:
                outcomes.append(FrameOutcome(
                    stem, None, "review",
                    f"no image found for stem '{stem}' — frame and label dirs out "
                    f"of sync"))
                continue

            width, height = dims_for(image)
            pending.setdefault(label_dir.name, []).append(
                (stem, image, record, width, height))

    # --- coordinate space: a NAME per session for ambiguous frames ------------
    override = named_space(str(get_dotted(cfg, "convert.coord_space", "auto")))
    max_edge = int(get_dotted(cfg, "llm.max_edge", 1024) or 0)
    for session, items in pending.items():
        session_space: str | None = None
        if override is None:
            # Each frame is judged against the size the LLM was SENT; frames of
            # different cameras/resolutions never share a scale factor.
            verdicts = []
            for _stem, _img, rec, w, h in items:
                sw, sh, _inf = sent_size_of(rec, w, h, max_edge)
                verdicts.append(CoordSpace.classify(rec.get("boxes") or [], sw, sh))
            session_space = resolve_space_name(verdicts)
            tally = collections.Counter(verdicts)
            LOG.info("session %s coordinate space for ambiguous frames: %s  "
                     "(per-frame verdicts: %s)", session, session_space, dict(tally))
        for stem, image, record, width, height in items:
            outcomes.append(validate_frame(stem, image, record, width, height,
                                           names, idx, cfg, session_space))

    # Hand-labelled frames that have no raw reply at all (a reviewer may label
    # a frame the LLM failed on, or one dropped before labelling).
    if args.include_review:
        for review_dir in review_dirs:
            for path in sorted(review_dir.iterdir()):
                stem = hand_label_stem(path)
                if stem is None or stem in seen_stems:
                    continue
                image = image_for(stem)
                if image is None:
                    LOG.debug("ignoring %s: no frame with stem %s", path, stem)
                    continue
                seen_stems.add(stem)
                outcomes.append(load_hand_label(path, stem, image, names))

    summary = None
    if not args.dry_run:
        summary = assemble(args, cfg, names, outcomes, manifests, models_by_stem)

    return report(outcomes, summary, names, cfg)


def _read_manifest(images_dir: Path) -> dict[str, dict]:
    manifest = images_dir / "manifest.jsonl"
    out: dict[str, dict] = {}
    for rec in read_jsonl(manifest):
        name = Path(rec.get("path", "")).name or rec.get("file", "")
        if name:
            out[name] = rec
    return out


def _jpeg_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image  # noqa: WPS433
    except ImportError:
        LOG.warning("Pillow not installed: image sizes unknown, every frame will go "
                    "to review. pip install pillow")
        return (0, 0)
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


if __name__ == "__main__":
    sys.exit(main())
