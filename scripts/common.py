"""Shared helpers for the airseekers-yolo-retrain scripts.

Deliberately dependency-light: ``yaml`` + ``requests`` are the only hard
requirements, and both are already present in any sane venv. ``Pillow`` is
needed only by the capture step (perceptual hash + optional re-encode);
``numpy`` is optional and only speeds up the hash.

Nothing in here touches ROS. The mower is reached over plain HTTP
(``web_video_server``), so these scripts run on any LAN host.

Import path: the numbered scripts do ``sys.path.insert(0, str(Path(__file__).parent))``
so that running them by absolute path works from any cwd.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - dependency check
    sys.exit("ERROR: pyyaml is required.  pip install pyyaml")

try:
    import requests
except ImportError:  # pragma: no cover - dependency check
    sys.exit("ERROR: requests is required.  pip install requests")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"

LOG = logging.getLogger("retrain")

# MJPEG multipart boundary as emitted by web_video_server / most HTTP MJPEG
# servers. web_video_server uses a fixed "--frame"; we accept anything.
_SOI = b"\xff\xd8\xff"
_EOI = b"\xff\xd9"


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # requests/urllib3 debug logging is very noisy at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _coerce(text: str) -> Any:
    """Turn a --set value string into a typed Python value.

    Used so ``--set capture.duration_s=30`` is an int, not the string "30".
    """
    low = text.strip().lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if re.fullmatch(r"[+-]?\d+", text.strip()):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", text.strip()):
        return float(text)
    if text.strip().startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def set_dotted(cfg: dict, dotted: str, value: str) -> None:
    """Set ``a.b.c=value`` in a nested dict, creating intermediate dicts."""
    parts = dotted.split(".")
    node = cfg
    for key in parts[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[parts[-1]] = _coerce(value)


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: Sequence[str] = ()) -> dict:
    """Load pipeline.yaml, apply ``--set`` overrides, return the dict."""
    cfg_path = Path(path or DEFAULT_CONFIG)
    if not cfg_path.is_file():
        sys.exit(f"ERROR: config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        sys.exit(f"ERROR: {cfg_path} did not parse to a mapping")
    for item in overrides:
        if "=" not in item:
            sys.exit(f"ERROR: --set expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), value)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(path_like: str | os.PathLike[str]) -> Path:
    """Resolve a config path against the repo root (absolute paths pass through)."""
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def cfg_path(cfg: dict, dotted: str, default: str | None = None) -> Path:
    """Fetch a config value as a repo-root-relative Path."""
    value = get_dotted(cfg, dotted, default)
    if value is None:
        sys.exit(f"ERROR: config value '{dotted}' is not set")
    return resolve(value)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="path to pipeline.yaml (default: %(default)s)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config value, e.g. --set capture.duration_s=30")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# class vocabulary
# ---------------------------------------------------------------------------

def class_names(cfg: dict) -> list[str]:
    """The 22 class names, in the order the vendor detector expects.

    If ``classes.list_file`` is set and readable it wins over ``classes.names``,
    so the vocabulary can never silently drift from the vendor's own file.
    """
    list_file = get_dotted(cfg, "classes.list_file")
    if list_file:
        path = resolve(list_file)
        if not path.is_file():
            sys.exit(f"ERROR: classes.list_file not found: {path}")
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        LOG.info("class list from %s (%d classes)", path, len(names))
    else:
        names = [str(n).strip() for n in get_dotted(cfg, "classes.names", [])]
    if not names:
        sys.exit("ERROR: no class names configured (classes.names / classes.list_file)")
    if len(set(names)) != len(names):
        sys.exit(f"ERROR: duplicate class names in vocabulary: {names}")
    return names


def class_index(names: Sequence[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


# ---------------------------------------------------------------------------
# image helpers (Pillow only; numpy optional)
# ---------------------------------------------------------------------------

def pillow():
    try:
        from PIL import Image  # noqa: WPS433
    except ImportError:
        return None
    return Image


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) without a full decode when Pillow is available."""
    img_cls = pillow()
    if img_cls is None:
        return (0, 0)
    try:
        with img_cls.open(path) as im:
            return im.size
    except Exception:  # pragma: no cover - corrupt file
        return (0, 0)


def dhash(path_or_bytes: Path | bytes, size: int = 8) -> int:
    """64-bit difference hash of an image, as an int.

    Small, cheap and good enough to reject "the same parked scene re-encoded 40
    times". Not a semantic hash: it will miss two different objects in an
    identical background, which is why ``phash_min_distance`` is a threshold and
    not a boolean.
    """
    import io

    img_cls = pillow()
    if img_cls is None:
        return 0
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            with img_cls.open(io.BytesIO(bytes(path_or_bytes))) as im:
                small = im.convert("L").resize((size + 1, size), img_cls.BILINEAR)
                px = list(small.getdata())
        else:
            with img_cls.open(path_or_bytes) as im:
                small = im.convert("L").resize((size + 1, size), img_cls.BILINEAR)
                px = list(small.getdata())
    except Exception:  # pragma: no cover - unreadable image
        return 0
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def downscaled_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """The size ``maybe_reencode`` produces for a ``width`` x ``height`` image.

    Kept as its own function because step 03 needs the *same* arithmetic to
    recover the size a frame was sent to the LLM at, for raw replies written
    before 02 started recording it.
    """
    if not max_edge or max(width, height) <= max_edge:
        return width, height
    scale = max_edge / float(max(width, height))
    return int(width * scale), int(height * scale)


def image_size_bytes(payload: bytes) -> tuple[int, int]:
    """(width, height) of encoded image bytes via Pillow, (0, 0) when unavailable."""
    import io

    img_cls = pillow()
    if img_cls is None:
        return (0, 0)
    try:
        with img_cls.open(io.BytesIO(payload)) as im:
            return im.size
    except Exception:
        return (0, 0)


def maybe_reencode(payload: bytes, max_edge: int, quality: int) -> bytes:
    """Downscale a JPEG to ``max_edge`` on its long side if needed."""
    img_cls = pillow()
    if img_cls is None:
        return payload
    import io

    try:
        with img_cls.open(io.BytesIO(payload)) as im:
            w, h = im.size
            if max(w, h) <= max_edge:
                return payload
            im = im.convert("RGB").resize(downscaled_size(w, h, max_edge), img_cls.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
    except Exception:  # pragma: no cover
        return payload


# ---------------------------------------------------------------------------
# mower profiles (ROS 1 vendor firmware vs the ROS 2 port)
# ---------------------------------------------------------------------------

def mower_profile_name(cfg: dict) -> str:
    return str(get_dotted(cfg, "mower.profile", "ros1") or "ros1").lower()


def mower_profile(cfg: dict) -> dict:
    """The ``profiles.<mower.profile>`` block, or exit with a clear error."""
    name = mower_profile_name(cfg)
    profiles = get_dotted(cfg, "profiles", {}) or {}
    prof = profiles.get(name)
    if not isinstance(prof, dict):
        sys.exit(f"ERROR: mower.profile is {name!r} but profiles.{name} is not "
                 f"defined (have: {sorted(profiles)})")
    return prof


def mower_base_url(cfg: dict) -> str:
    """``mower.base_url`` when set, else ``http://<mower.host>:<profile port>``."""
    explicit = get_dotted(cfg, "mower.base_url")
    if explicit:
        return str(explicit).rstrip("/")
    host = get_dotted(cfg, "mower.host")
    if not host:
        sys.exit("ERROR: set mower.host or mower.base_url in the config")
    port = mower_profile(cfg).get("port")
    return f"http://{host}:{port}" if port else f"http://{host}"


def camera_presets(cfg: dict) -> dict:
    """Camera presets: a top-level ``cameras:`` block wins (legacy configs),
    otherwise the active profile's ``cameras``."""
    legacy = get_dotted(cfg, "cameras")
    if isinstance(legacy, dict) and legacy:
        return legacy
    return mower_profile(cfg).get("cameras", {}) or {}


def silent_topic_hint(cfg: dict, topic: str = "<topic>") -> list[str]:
    """Profile-specific advice for a camera that returned nothing."""
    hints = mower_profile(cfg).get("silent_hints") or []
    ns = topic.rsplit("/", 1)[0] if topic.count("/") > 1 else topic
    return [str(h).replace("{topic}", topic).replace("{ns}", ns) for h in hints]


# ---------------------------------------------------------------------------
# MJPEG over HTTP
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StreamSpec:
    """Where one camera's frames come from."""
    name: str
    url: str
    mode: str = "stream"      # "stream" (multipart) | "snapshot" (single JPEG)
    rate_hz: float = 1.0
    topic: str = ""


def build_stream_url(base_url: str, template: str, topic: str) -> str:
    base = base_url.rstrip("/")
    path = template.format(topic=topic)
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def iter_mjpeg(url: str, connect_timeout: float = 5.0,
               read_timeout: float = 30.0) -> Iterator[bytes]:
    """Yield JPEG payloads from an HTTP MJPEG (multipart/x-mixed-replace) URL.

    Two things make this fiddly and both are handled here:

    1. The stream never ends. This is a generator; the caller must stop
       iterating (the ``with`` in ``capture`` closes the socket on exit).
    2. web_video_server emits ``Content-Length`` per part in some builds and
       nothing in others, and some builds return the first frame only after the
       ROS topic is publishing. We therefore do not trust headers and simply
       scan for JPEG SOI..EOI pairs, which is robust across servers.

    A snapshot request is handled by the same scanner, so ``/snapshot`` works
    even though it is not multipart.
    """
    with requests.get(
        url, stream=True, timeout=(connect_timeout, read_timeout)
    ) as resp:
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                start = buf.find(_SOI)
                if start < 0:
                    # Keep a short tail in case SOI straddles a chunk boundary.
                    if len(buf) > 3:
                        del buf[:-3]
                    break
                end = buf.find(_EOI, start + 3)
                if end < 0:
                    # Incomplete frame: drop what we have before start and wait.
                    del buf[:start]
                    break
                yield bytes(buf[start:end + 2])
                del buf[:end + 2]


def first_mjpeg_frame(url: str, connect_timeout: float = 5.0,
                      read_timeout: float = 30.0) -> bytes | None:
    """First JPEG off a multipart stream, then close the connection.

    Used by ``--probe`` for ``mode: stream`` sources. A plain (non-streaming)
    GET on an MJPEG URL never completes, because the body never ends; this
    reads with ``stream=True`` and stops at the first complete frame.
    """
    gen = iter_mjpeg(url, connect_timeout, read_timeout)
    try:
        return next(gen, None)
    finally:
        gen.close()


def first_snapshot(url: str, connect_timeout: float = 5.0,
                   read_timeout: float = 30.0) -> bytes | None:
    """Fetch a single JPEG. Returns None on an empty/topic-silent response."""
    with requests.get(url, timeout=(connect_timeout, read_timeout)) as resp:
        if resp.status_code != 200:
            LOG.debug("snapshot %s -> HTTP %s", url, resp.status_code)
            return None
        body = resp.content
    if not body:
        return None
    start = body.find(_SOI)
    if start < 0:
        LOG.debug("snapshot %s returned no JPEG (len=%d)", url, len(body))
        return None
    end = body.rfind(_EOI)
    return body[start:end + 2] if end > start else body[start:]


# ---------------------------------------------------------------------------
# manifest / provenance io
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON object and flush, so a crash loses at most the tail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                LOG.warning("%s:%d: bad JSON (%s) — skipped", path, line_no, exc)
    return out


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def file_sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """Hash of at most ``limit`` bytes. Models are large; full hashing is slow."""
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while read < limit:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def any_match(name: str, patterns: Iterable[str]) -> bool:
    base = Path(name).name
    return any(fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(name, pat)
               for pat in patterns)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def list_images(root: Path, recursive: bool = True,
                extensions: Sequence[str] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    """Sorted image list. Sorted so runs are reproducible across machines."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in exts)
