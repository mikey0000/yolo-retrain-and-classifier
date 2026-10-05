#!/usr/bin/env python3
"""02_llm_label.py — ask a local vision LLM (LM Studio) for boxes, one frame at a time.

Runs on a LAN host with an OpenAI-compatible server. LM Studio is the default:
Developer tab -> Start server, then the endpoint is
``http://<host>:1234/v1``. Anything else works too (vLLM, llama.cpp's server,
Ollama's ``/v1``) as long as it speaks ``POST /v1/chat/completions`` with an
``image_url`` data-URL part.

Output::

    work/labels/<session>/raw/<stem>.json    the request and the full raw reply
    work/labels/<session>/index.jsonl        one row per frame, for step 03

Design decisions worth knowing before you tune anything:

* **The class list is closed.** The vendor detector has a 22-class ABI; the
  prompt enumerates the names and the client rejects anything outside them
  later. A vision model left to invent class names produces a dataset that
  trains fine and deploys as garbage.

* **temperature 0 and one image per request.** A vision model's boxes drift
  between runs otherwise, so re-running the batch silently changes the dataset
  and the training result is not reproducible. No multi-image or multi-turn
  context: it makes small VLMs worse, not better.

* **Every response is kept, including failures.** The raw JSON is the audit
  trail. If a batch came out wrong you can re-run or hand-fix without
  re-capturing, which is the expensive step.

* **Empty responses are data.** ``{"boxes": []}`` means "nothing from the
  vocabulary here". Those frames are negatives, and a detector trained only on
  positives invents obstacles in clean grass.

* **This is a suggestion, not ground truth.** Read `labels/*/index.jsonl` before
  training. The `03_to_yolo.py` validation thresholds assume a human has looked
  at the low-confidence and out-of-vocabulary buckets.

Examples
--------
    # point at LM Studio, auto-discover the loaded model
    python3 scripts/02_llm_label.py --images work/frames/20261005-142530

    # explicit model, 4-way concurrency, 200 frames
    python3 scripts/02_llm_label.py --images work/frames/20261005-142530 \
        --model qwen2.5-vl-7b-instruct --concurrency 4 --limit 200

    # from another machine: bind LM Studio to the LAN, then
    python3 scripts/02_llm_label.py --base-url http://192.168.1.50:1234/v1 --images ...

    # single frame, print the raw reply (sanity-check coordinates before a batch)
    python3 scripts/02_llm_label.py --images work/frames/<session> --limit 1 --print-response

    # re-run only what failed last time
    python3 scripts/02_llm_label.py --images ... --retry-errors
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import io
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LOG, add_common_args, any_match, cfg_path, class_names, get_dotted,
    image_size_bytes, list_images, load_config, maybe_reencode, now_iso,
    setup_logging, write_json_atomic,
)

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("ERROR: requests is required.  pip install requests")


# ---------------------------------------------------------------------------
# vocabulary aliases
# ---------------------------------------------------------------------------

#: Synonyms a VLM reaches for, mapped onto the closed 22-class vocabulary.
#: Applied before the out-of-vocabulary check, so "rock" becomes "stone"
#: instead of being stripped (and the object silently turned into background).
#: Merged with ``llm.class_aliases`` from the config. Keys are lower-case.
DEFAULT_ALIASES: dict[str, str] = {
    "trash can": "trashbin", "trashcan": "trashbin", "trash bin": "trashbin",
    "bin": "trashbin", "garbage can": "trashbin", "wheelie bin": "trashbin",
    "toy car": "toycars", "toy cars": "toycars", "toy vehicle": "toycars",
    "can": "cans", "soda can": "cans", "tin can": "cans",
    "ball": "sports ball", "football": "sports ball", "soccer ball": "sports ball",
    "tennis ball": "sports ball",
    "rock": "stone", "rocks": "stone", "stones": "stone", "pebble": "stone",
    "log": "wood", "plank": "wood", "stick": "wood", "branch": "wood",
    "tree trunk": "trunk", "stump": "trunk", "tree stump": "trunk",
    "charging station": "dock", "charging dock": "dock", "docking station": "dock",
    "plant pot": "potted plant", "flower pot": "potted plant",
    "flowerpot": "potted plant", "pot plant": "potted plant",
}


def build_aliases(names: list[str], extra: dict | None = None) -> dict[str, str]:
    """Lower-case alias -> canonical vocabulary name.

    Entries whose target is not in ``names`` are dropped with a warning (a
    custom vocabulary must not be poisoned by the built-in table), and an alias
    that is itself a vocabulary name is ignored.
    """
    canon = {n.lower(): n for n in names}
    user = {" ".join(str(k).lower().split()): str(v).strip()
            for k, v in (extra or {}).items()}
    out: dict[str, str] = {}
    for alias, target in {**DEFAULT_ALIASES, **user}.items():
        if alias in canon:
            continue
        name = canon.get(target.lower())
        if name is None:
            if alias in user:
                LOG.warning("llm.class_aliases %r -> %r: target is not in the "
                            "vocabulary; ignored", alias, target)
            continue
        out[alias] = name
    return out


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LabelerConfig:
    base_url: str
    model: str
    api_key: str
    timeout_s: float
    max_retries: int
    retry_backoff_s: float
    temperature: float
    max_tokens: int
    top_p: float
    response_format: str
    max_edge: int
    jpeg_quality: int
    system_prompt: str
    user_prompt: str
    condition: str
    view_hint: str


@dataclasses.dataclass
class Result:
    """Outcome of labelling one frame."""
    stem: str
    path: Path
    ok: bool
    boxes: list[dict]
    raw: dict
    error: str = ""
    elapsed_s: float = 0.0
    attempts: int = 1
    out_of_vocabulary: list[str] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    # The image size actually sent to the model (after llm.max_edge) and the
    # original frame size. A model answering in pixels answers in the SENT
    # space; 03 needs both to map those boxes back onto the frame.
    sent_size: tuple[int, int] = (0, 0)
    orig_size: tuple[int, int] = (0, 0)


class Labeler:
    """One OpenAI-compatible vision client. Thread-safe (requests.Session is)."""

    def __init__(self, cfg: LabelerConfig, names: list[str],
                 aliases: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.names = names
        self.allowed = {n.lower() for n in names}
        self.aliases = aliases if aliases is not None else build_aliases(names)
        # Set once the server has rejected response_format with HTTP 400, so
        # every later frame skips the doomed first request.
        self._no_response_format = False
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        })
        self._fmt_lock = threading.Lock()

    # -- server discovery ---------------------------------------------------

    def list_models(self) -> list[str]:
        url = f"{self.cfg.base_url.rstrip('/')}/models"
        resp = self._session.get(url, timeout=min(self.cfg.timeout_s, 30.0))
        resp.raise_for_status()
        return [str(m.get("id")) for m in resp.json().get("data", []) if m.get("id")]

    def pick_model(self) -> str:
        if self.cfg.model:
            return self.cfg.model
        models = self.list_models()
        if not models:
            raise RuntimeError(
                f"{self.cfg.base_url}/models returned nothing. Is the LM Studio "
                "server started with a model loaded? (Developer tab)")
        # Prefer a vision-capable-looking name; fall back to whatever is loaded.
        hints = ("vl", "vision", "qwen2.5", "qwen3", "gemma-3", "minicpm", "internvl",
                 "llava", "moondream", "smolvlm")
        for name in models:
            low = name.lower()
            if any(h in low for h in hints):
                LOG.info("auto-selected model %s (of %s)", name, models)
                return name
        LOG.info("auto-selected model %s (no vision hint matched; %s)", models[0], models)
        return models[0]

    def health(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except Exception as exc:
            return False, f"{self.cfg.base_url} unreachable: {exc}"
        if not models:
            return False, f"{self.cfg.base_url} reachable but no model is loaded"
        return True, f"{self.cfg.base_url} serving {len(models)} model(s): {models}"

    # -- payload ------------------------------------------------------------

    def encode_image(self, path: Path) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
        """``(jpeg data URL, sent (w, h), original (w, h))``, downscaled to ``max_edge``.

        Vision models degrade on very large images and the wire cost is real;
        1024 px on the long edge is the practical sweet spot for a 7B VLM at
        ~120-180 KB per request. The sizes are returned because a model that
        replies in pixels replies in the *sent* image's pixels.
        """
        try:
            original = path.read_bytes()
        except OSError as exc:
            LOG.debug("cannot read %s: %s", path, exc)
            return None
        payload = maybe_reencode(original, self.cfg.max_edge, self.cfg.jpeg_quality)
        if not payload.startswith(b"\xff\xd8"):
            # PNG/BMP that needed no downscale: convert to JPEG explicitly.
            try:
                from PIL import Image  # noqa: WPS433
                with Image.open(io.BytesIO(payload)) as im:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="JPEG",
                                           quality=self.cfg.jpeg_quality)
                    payload = buf.getvalue()
            except Exception as exc:
                LOG.warning("cannot JPEG-encode %s: %s", path, exc)
                return None
        orig_size = image_size_bytes(original)
        sent_size = image_size_bytes(payload)
        data_url = "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")
        return data_url, sent_size, orig_size

    def build_body(self, data_url: str, view: str) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": self.cfg.user_prompt.format(
                        view=view or self.cfg.view_hint,
                        condition=self.cfg.condition)},
                    {"type": "image_url",
                     "image_url": {"url": data_url}},
                ]},
            ],
        }
        if self._no_response_format:
            return body
        if self.cfg.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.cfg.response_format == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "labels",
                "schema": {
                    "type": "object",
                    "properties": {
                        "boxes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "class": {"type": "string",
                                              "enum": list(self.names)},
                                    "box": {"type": "array", "items": {"type": "number"},
                                            "minItems": 4, "maxItems": 4},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["class", "box"],
                            },
                        },
                    },
                    "required": ["boxes"],
                },
            }}
        return body

    # -- request ------------------------------------------------------------

    def label_one(self, path: Path, view: str = "") -> Result:
        t0 = time.monotonic()
        encoded = self.encode_image(path)
        if encoded is None:
            return Result(path.stem, path, False, [],
                          {"error": "unreadable image"}, error="unreadable image")
        data_url, sent_size, orig_size = encoded

        body = self.build_body(data_url, view)
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        last_err = ""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._session.post(url, json=body, timeout=self.cfg.timeout_s)
                if resp.status_code == 400 and "response_format" in body:
                    # A model that rejects response_format still works without it.
                    body.pop("response_format", None)
                    resp = self._session.post(url, json=body, timeout=self.cfg.timeout_s)
                    if resp.ok:
                        with self._fmt_lock:
                            if not self._no_response_format:
                                self._no_response_format = True
                                LOG.warning(
                                    "HTTP 400 with response_format=%s; it works "
                                    "without it, so it is dropped for the rest of "
                                    "this run. Set llm.response_format: none to "
                                    "skip this.", self.cfg.response_format)
                resp.raise_for_status()
                payload = resp.json()
                parsed = self.parse_reply(payload)
                parsed["_request"] = {
                    "model": body["model"],
                    "temperature": body["temperature"],
                    "max_tokens": body["max_tokens"],
                    "response_format": body.get("response_format"),
                    "view": view,
                    "image_bytes_b64": len(data_url),
                    "attempt": attempt,
                }
                parsed["_elapsed_s"] = round(time.monotonic() - t0, 2)
                return Result(
                    stem=path.stem, path=path, ok=parsed.get("ok", True),
                    boxes=parsed.get("boxes", []), raw=parsed,
                    error=parsed.get("error", ""),
                    elapsed_s=parsed["_elapsed_s"], attempts=attempt,
                    out_of_vocabulary=parsed.get("out_of_vocabulary", []),
                    notes=parsed.get("notes", []),
                    sent_size=sent_size, orig_size=orig_size)
            except Exception as exc:  # network, HTTP, timeout, bad JSON
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.cfg.max_retries:
                    backoff = self.cfg.retry_backoff_s * (2 ** (attempt - 1))
                    LOG.debug("%s attempt %d/%d failed (%s); retrying in %.1fs",
                              path.name, attempt, self.cfg.max_retries, last_err, backoff)
                    time.sleep(backoff)
        return Result(path.stem, path, False, [],
                      {"error": last_err, "attempts": self.cfg.max_retries},
                      error=last_err, elapsed_s=round(time.monotonic() - t0, 2),
                      attempts=self.cfg.max_retries,
                      sent_size=sent_size, orig_size=orig_size)

    # -- reply parsing ------------------------------------------------------

    def parse_reply(self, payload: dict) -> dict:
        """Pull boxes out of an OpenAI-shaped reply, tolerantly.

        Real models wrap JSON in prose, fence it, or emit a slightly different
        key. We try hard to recover a *valid* structure, and record what we had
        to do so 03 can see it.
        """
        notes: list[str] = []
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {"ok": False, "boxes": [], "error": "reply had no choices[0].message",
                    "notes": notes}

        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        text = str(content).strip()

        obj = _extract_json(text)
        if obj is None:
            return {"ok": False, "boxes": [], "error": f"no JSON object in reply: "
                    f"{text[:200]!r}", "notes": notes, "content": text}
        # "Pure" = the whole reply parses as one JSON object. Formatting
        # (indentation, key order, spacing) is irrelevant; prose or a fence is not.
        try:
            pure = isinstance(json.loads(text), dict)
        except json.JSONDecodeError:
            pure = False
        if not pure:
            notes.append("reply was not pure JSON; recovered by extraction")

        boxes = obj.get("boxes")
        if boxes is None:
            # Tolerate the other shapes we have seen in the wild.
            for alt in ("objects", "detections", "labels", "annotations"):
                if isinstance(obj.get(alt), list):
                    boxes = obj[alt]
                    notes.append(f"read boxes from '{alt}'")
                    break
        if boxes is None and _normalise_box(obj) is not None:
            # A bare single box at the top level: {"class": "dog", "box": [...]}.
            boxes = [obj]
            notes.append("reply was a single box, not a boxes list")
        if boxes is None:
            return {"ok": False, "boxes": [], "error": "JSON had no 'boxes' key",
                    "notes": notes, "parsed": obj}
        if not isinstance(boxes, list):
            return {"ok": False, "boxes": [], "error": "'boxes' was not a list",
                    "notes": notes, "parsed": obj}

        clean: list[dict] = []
        oov: list[str] = []
        for item in boxes:
            norm = _normalise_box(item)
            if norm is None:
                notes.append(f"dropped malformed box entry: {item!r}")
                continue
            name = str(norm.get("class", "")).strip()
            alias = self.aliases.get(" ".join(name.lower().split()))
            if name.lower() not in self.allowed and alias:
                notes.append(f"alias {name!r} -> {alias!r}")
                norm["alias_of"] = name
                name = alias
            if name.lower() not in self.allowed:
                oov.append(name or "<empty>")
                notes.append(f"out-of-vocabulary class: {name!r}")
                continue
            norm["class"] = next(n for n in self.names if n.lower() == name.lower())
            clean.append(norm)

        return {"ok": True, "boxes": clean, "out_of_vocabulary": oov,
                "notes": notes, "content": text, "parsed": obj}


def _extract_json(text: str) -> dict | None:
    """Find the first balanced JSON object in ``text``."""
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # ```json fenced
    if "```" in text:
        chunk = text.split("```")
        for part in chunk[1:]:
            body = part[4:] if part[:4].lower() == "json" else part
            try:
                parsed = json.loads(body.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    # first {...} with balanced braces, ignoring braces inside strings
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalise_box(item: object) -> dict | None:
    """Coerce one box entry into {class, box:[4], confidence}.

    Accepts the plausible spellings: ``box``/``bbox``/``box_2d``, xyxy or xywh,
    confidence/conf/score, and numeric-ish strings.
    """
    if not isinstance(item, dict):
        return None
    coords = None
    for key in ("box", "bbox", "box_2d", "bbox_2d", "coordinates", "rect"):
        if key in item:
            coords = item[key]
            break
    if coords is None and all(k in item for k in ("x1", "y1", "x2", "y2")):
        coords = [item["x1"], item["y1"], item["x2"], item["y2"]]
    if coords is None and all(k in item for k in ("x", "y", "w", "h")):
        coords = _xywh_to_xyxy([item["x"], item["y"], item["w"], item["h"]])
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    try:
        box = [float(v) for v in coords]
    except (TypeError, ValueError):
        return None

    # A dict with x/y/w/h *and* a "box" key means xyxy won: fine. But an entry
    # whose own key set suggests xywh should be converted here.
    if all(k in item for k in ("x", "y", "w", "h")) and "box" not in item:
        box = _xywh_to_xyxy(box)
    # "format": "xywh" hint, if the model was explicit.
    fmt = str(item.get("format", item.get("box_format", ""))).lower()
    if fmt in ("xywh", "coco", "yolo"):
        box = _xywh_to_xyxy(box)

    conf = None
    for key in ("confidence", "conf", "score", "probability", "prob"):
        if key in item:
            try:
                conf = float(item[key])
            except (TypeError, ValueError):
                conf = None
            break
    name = item.get("class", item.get("label", item.get("name", item.get("category"))))
    return {"class": str(name if name is not None else "").strip(),
            "box": box, "confidence": conf}


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


# ---------------------------------------------------------------------------
# session bookkeeping
# ---------------------------------------------------------------------------

def load_done(out_dir: Path, retry_errors: bool) -> set[str]:
    """Stems already labelled. With --retry-errors, failures are re-queued."""
    index = out_dir / "index.jsonl"
    done: set[str] = set()
    if not index.is_file():
        return done
    with index.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_errors and not rec.get("ok"):
                done.discard(rec.get("stem"))
                continue
            done.add(rec.get("stem", ""))
    return {d for d in done if d}


def append_index(path: Path, rec: dict, lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch frames to an LM Studio (OpenAI-compatible) vision model "
                    "for suggested bounding boxes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.")
    add_common_args(parser)

    parser.add_argument("--images", required=True,
                        help="frame directory from 01_capture.py")
    parser.add_argument("--base-url", help="override llm.base_url")
    parser.add_argument("--model", help="override llm.model (default: auto-detect)")
    parser.add_argument("--concurrency", type=int, help="parallel requests")
    parser.add_argument("--limit", type=int, help="max frames this run")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N frames")
    parser.add_argument("--out", help="output dir (default: labels_dir/<session>)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="re-label frames whose last attempt failed")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be sent, send nothing")
    parser.add_argument("--print-response", action="store_true",
                        help="dump the raw reply for each frame (sanity check)")
    parser.add_argument("--health", action="store_true",
                        help="query /v1/models, report, then exit")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config, args.overrides)
    names = class_names(cfg)

    images_dir = Path(args.images).expanduser()
    if not images_dir.is_dir():
        LOG.error("image dir not found: %s", images_dir)
        return 2

    lcfg = LabelerConfig(
        base_url=(args.base_url or get_dotted(cfg, "llm.base_url")).rstrip("/"),
        model=args.model or str(get_dotted(cfg, "llm.model", "")),
        api_key=str(get_dotted(cfg, "llm.api_key", "lm-studio")),
        timeout_s=float(get_dotted(cfg, "llm.timeout_s", 180)),
        max_retries=int(get_dotted(cfg, "llm.max_retries", 3)),
        retry_backoff_s=float(get_dotted(cfg, "llm.retry_backoff_s", 2.0)),
        temperature=float(get_dotted(cfg, "llm.temperature", 0.0)),
        max_tokens=int(get_dotted(cfg, "llm.max_tokens", 1024)),
        top_p=float(get_dotted(cfg, "llm.top_p", 1.0)),
        response_format=str(get_dotted(cfg, "llm.response_format", "json_object")),
        max_edge=int(get_dotted(cfg, "llm.max_edge", 1024)),
        jpeg_quality=int(get_dotted(cfg, "llm.jpeg_quality", 85)),
        system_prompt=str(get_dotted(cfg, "llm.system_prompt", "")).strip(),
        user_prompt=str(get_dotted(cfg, "llm.user_prompt",
                                   "Label this frame. View: {view}. "
                                   "Objects at mower deck height, {condition}.")),
        condition=str(get_dotted(cfg, "llm.condition", "daylight outdoor lawn")),
        view_hint=str(get_dotted(cfg, "llm.view_hint", None)
                      or get_dotted(cfg, "mower.front_camera_hint", None)
                      or "a robot mower camera at ground level"),
    )
    aliases = build_aliases(names, get_dotted(cfg, "llm.class_aliases", {}) or {})
    labeler = Labeler(lcfg, names, aliases)

    ok, detail = labeler.health()
    print(f"endpoint: {detail}")
    if not ok:
        print("\nLM Studio checklist:")
        print("  1. Developer tab -> 'Start Server' (default port 1234)")
        print("  2. load a VISION model (a text-only model will return nothing useful)")
        print("  3. to reach it from another machine, start the server bound to the "
              "LAN\n     and allow the port through the firewall")
        return 2
    if args.health:
        return 0

    try:
        labeler.cfg.model = labeler.pick_model()
    except Exception as exc:
        LOG.error("cannot select a model: %s", exc)
        return 2

    session = images_dir.name
    out_dir = Path(args.out).expanduser() if args.out else \
        (cfg_path(cfg, "paths.labels_dir") / session)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.jsonl"

    skip = get_dotted(cfg, "llm.skip_patterns", []) or []
    files = [p for p in list_images(images_dir, recursive=True)
             if not any_match(p.name, skip)]
    if args.offset:
        files = files[args.offset:]
    already = load_done(out_dir, args.retry_errors)
    todo = [p for p in files if p.stem not in already]
    if args.limit is not None and args.limit >= 0:
        todo = todo[:args.limit]

    # A session-local view hint: frame stems are <seq>_<stamp_ns>.jpg inside a
    # per-camera directory, so the camera name is recoverable from the path.
    manifest = _read_manifest(images_dir)

    concurrency = args.concurrency if args.concurrency is not None else \
        int(get_dotted(cfg, "llm.concurrency", 2))
    concurrency = max(1, concurrency)

    print(f"\nmodel      : {labeler.cfg.model}")
    print(f"classes    : {len(names)} (closed vocabulary)")
    print(f"images dir : {images_dir}")
    print(f"output     : {out_dir}")
    print(f"to label   : {len(todo)} frame(s)"
          + (f" (skipped {len(already)} already done)" if already else "")
          + f"  concurrency={concurrency}")
    print(f"budget     : ~{len(todo) * 0.15 / concurrency:.0f}s at 150 ms/frame "
          f"(a 7B VLM on a GPU is usually 1-4 s/frame)")

    if args.dry_run:
        for path in todo[:50]:
            print(f"  would send {path}")
        if len(todo) > 50:
            print(f"  ... and {len(todo) - 50} more")
        return 0
    if not todo:
        print("\nnothing to do. (use --retry-errors to re-run failures)")
        return 0

    print()
    counters = {"ok": 0, "empty": 0, "failed": 0, "oov": 0, "boxes": 0}
    lock = threading.Lock()
    t_start = time.monotonic()
    done_count = 0

    def view_hint_for(path: Path) -> str:
        rec = manifest.get(path.name)
        topic = (rec or {}).get("topic") or ""
        return topic if topic else lcfg.view_hint

    def work(path: Path) -> None:
        nonlocal done_count
        result = labeler.label_one(path, view_hint_for(path))
        raw_path = raw_dir / f"{result.stem}.json"
        write_json_atomic(raw_path, {
            "stem": result.stem,
            "image": path.name,
            "image_path": str(path),
            # Pixel replies are in the sent image's space; 03 rescales by
            # orig/sent. Recorded per frame because frames differ in size.
            "sent_width": result.sent_size[0],
            "sent_height": result.sent_size[1],
            "orig_width": result.orig_size[0],
            "orig_height": result.orig_size[1],
            "max_edge": lcfg.max_edge,
            "ok": result.ok,
            "error": result.error,
            "boxes": result.boxes,
            "out_of_vocabulary": result.out_of_vocabulary,
            "notes": result.notes,
            "elapsed_s": result.elapsed_s,
            "attempts": result.attempts,
            "labelled_at": now_iso(),
            "model": labeler.cfg.model,
            "raw": result.raw,
        })
        append_index(index, {
            "stem": result.stem,
            "image": path.name,
            "image_path": str(path),
            "ok": result.ok,
            "error": result.error,
            "n_boxes": len(result.boxes),
            "classes": sorted({b["class"] for b in result.boxes}),
            "out_of_vocabulary": result.out_of_vocabulary,
            "notes": result.notes,
            "elapsed_s": result.elapsed_s,
            "attempts": result.attempts,
            "model": labeler.cfg.model,
            "raw_file": str(raw_path.relative_to(out_dir.parent)),
            "labelled_at": now_iso(),
        }, lock)

        with lock:
            done_count += 1
            if result.ok:
                counters["ok"] += 1
                counters["boxes"] += len(result.boxes)
                if not result.boxes:
                    counters["empty"] += 1
            else:
                counters["failed"] += 1
            if result.out_of_vocabulary:
                counters["oov"] += 1
            n = done_count
            rate = (time.monotonic() - t_start) / max(n, 1)
            eta = rate * (len(todo) - n)
            if result.ok and result.boxes:
                tag = ",".join(sorted({b["class"] for b in result.boxes}))
                msg = f"{n}/{len(todo)} {path.name} -> {len(result.boxes)} box(es) [{tag}]"
            elif result.ok:
                msg = f"{n}/{len(todo)} {path.name} -> no objects (negative)"
            else:
                msg = f"{n}/{len(todo)} {path.name} -> FAILED: {result.error[:80]}"
            if args.print_response and result.raw.get("content") is not None:
                msg += "\n    " + str(result.raw["content"])[:1000]
            if n % 5 == 0 or not result.ok or n == 1:
                LOG.info("%s  %.1fs/frame  eta %s", msg, rate,
                         f"{eta / 60:.1f}m")

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [pool.submit(work, path) for path in todo]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
        pool.shutdown(wait=True)
    except KeyboardInterrupt:
        # Without cancel_futures the pool would label every queued frame
        # before exiting; only the in-flight requests are waited for.
        pool.shutdown(wait=True, cancel_futures=True)
        LOG.warning("interrupted — %d/%d done, partial results are on disk",
                    done_count, len(todo))
    elapsed = time.monotonic() - t_start

    print()
    print(f"labelled   : {done_count}/{len(todo)} in {elapsed / 60:.1f}m")
    print(f"with boxes : {counters['ok']} ({counters['boxes']} boxes total)")
    print(f"negatives  : {counters['empty']} (empty replies — keep these)")
    print(f"failed     : {counters['failed']}")
    print(f"oov frames : {counters['oov']} (the model invented a class name)")
    print(f"index      : {index}")
    print(f"raw audit  : {raw_dir}")

    if counters["empty"] > len(todo) * 0.5 and counters["ok"]:
        print("\nWARNING: over half the frames came back empty. Before building a "
              "dataset, check that the\n         model actually sees the images — a "
              "text-only model in LM Studio returns\n         {\"boxes\": []} "
              "forever and the result looks like 'a clean lawn'.")
    if counters["oov"]:
        print("\nWARNING: the model used class names outside the vocabulary (after "
              "aliasing). Those\n         frames go to review/ in 03 — the "
              "unlabelled object would otherwise\n         train as background. "
              "Add an llm.class_aliases entry or use\n         "
              "response_format: json_schema.")

    print(f"\nnext: python3 scripts/03_to_yolo.py --labels {out_dir} "
          "--images " + str(images_dir))
    return 0


def _read_manifest(images_dir: Path) -> dict[str, dict]:
    """filename -> manifest record, when 01_capture wrote one next to the frames."""
    manifest = images_dir / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    out: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("path", "")
            out[Path(name).name or rec.get("file", "")] = rec
            if rec.get("file"):
                out[rec["file"]] = rec
    return {k: v for k, v in out.items() if k}


if __name__ == "__main__":
    sys.exit(main())
