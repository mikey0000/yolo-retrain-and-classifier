#!/usr/bin/env python3
"""01_capture.py — grab frames from the mower's LAN MJPEG streams, RTSP, or a local folder.

Three sources, one output layout:

  A. mower   — ``web_video_server`` on the mower: port 8181 on the vendor ROS 1
               firmware (``--profile ros1``), 8080 on the ROS 2 port
               (``--profile ros2``, where it is off until launched with
               ``cameras.launch.py web_video_server:=true``). Per camera, either
               a held MJPEG multipart connection (``mode: stream``, cheap
               because frames are already JPEG on the wire) or one HTTP GET per
               kept frame (``mode: snapshot``, politer to the single-threaded
               server). Stream and snapshot cameras run together, one thread
               each, in one session.

  B. rtsp    — the mower's mediamtx RTSP relay (``mower.transport: rtsp``, or
               ``--rtsp name=rtsp://...``), decoded by ffmpeg. One thread and
               one ffmpeg per camera; ffmpeg decimates to the camera's keep
               rate, and everything downstream (dedupe, quota, manifest) is
               identical to the HTTP path. Use this when web_video_server is
               not running or is contended.

  C. folder  — an existing directory of JPEGs (and video files, decoded with
               ffmpeg when ``capture.folder.videos`` is on). No mower, no
               network, no ROS.

Output::

    work/frames/<session>/<camera>/<seq>_<stamp_ns>.jpg
    work/frames/<session>/manifest.jsonl     one JSON object per kept frame

Why the manifest: later steps need to know *which* camera a frame came from, when
it was captured, and (if you add it) where the mower was. Without it, a dataset
built from a mixed session is a bag of pixels. The manifest is the only link
between an image and its provenance.

Near-duplicate rejection (``phash_min_distance``) matters more than it looks: a
parked mower produces identical frames at the full topic rate, and a dataset full
of near-duplicates inflates train accuracy while teaching the model nothing.

This script never opens ``/dev/video*``: the vendor nodes own the devices and a
second ``VIDIOC_STREAMON`` on the same node fails with ``EBUSY``. The HTTP path
sidesteps that entirely, which is also why this runs on the LAN host rather than
on the mower.

Examples
--------
    # the two OA cameras det_ros reads, 60 s, rates from the config
    python3 scripts/01_capture.py --cameras left_oa,right_oa

    # against the ROS 2 port (web_video_server:=true on port 8080)
    python3 scripts/01_capture.py --profile ros2 --probe

    # explicit stream URL (any MJPEG server, mower or not)
    python3 scripts/01_capture.py --url rear='http://192.168.1.105:8181/stream?topic=/rear_camera/image_raw&type=mjpeg'

    # the mower's RTSP relay instead of web_video_server (needs ffmpeg)
    python3 scripts/01_capture.py --rtsp left_oa=rtsp://192.168.1.105:8554/left_oa
    python3 scripts/01_capture.py --transport rtsp --cameras left_oa,right_oa

    # custom web_video_server endpoint from a base URL + ROS topics
    python3 scripts/01_capture.py --base-url http://192.168.1.105:8181 \
        --topic front=/vio/right/image_raw --topic left=/left_oa_camera/image_raw

    # probe what the mower is publishing without capturing
    python3 scripts/01_capture.py --probe --base-url http://192.168.1.105:8181

    # no mower: import a folder of frames (and video files, if ffmpeg is there)
    python3 scripts/01_capture.py --folder /path/to/frames --rate 2
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LOG, REPO_ROOT, StreamSpec, add_common_args, append_jsonl, build_rtsp_url,
    build_stream_url, camera_presets, cfg_path, class_names, dhash, dir_size,
    ffmpeg_available, first_mjpeg_frame, first_rtsp_frame, first_snapshot,
    get_dotted, hamming, human_bytes, image_size_bytes, iter_mjpeg, iter_rtsp,
    iter_video, list_images, list_videos, load_config, maybe_reencode,
    mower_base_url, mower_host, mower_profile, mower_profile_name,
    mower_transport, now_iso, setup_logging, silent_topic_hint,
)

_STOP = threading.Event()


def _display_path(path: Path) -> str:
    """Repo-relative when the frame lives under the repo, else absolute.

    The manifest is an audit trail, so its paths have to stay usable when the
    repo moves. Absolute paths for out-of-tree captures are correct rather than
    quietly wrong.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _on_signal(signum, _frame) -> None:  # pragma: no cover - signal path
    LOG.info("signal %s received — finishing current frame and stopping", signum)
    _STOP.set()


# ---------------------------------------------------------------------------
# source construction
# ---------------------------------------------------------------------------

def stream_from_config(cfg: dict) -> list[StreamSpec]:
    """Turn the active profile's camera presets into StreamSpecs.

    With ``mower.transport: rtsp`` every camera comes from the profile's
    ``rtsp_url`` template (the mediamtx relay) instead of web_video_server;
    a camera may also carry its own ``rtsp:`` URL to opt in individually.
    """
    cameras = camera_presets(cfg)
    if not cameras:
        return []
    rtsp_transport = mower_transport(cfg)
    tpl = str(mower_profile(cfg).get("rtsp_url") or "")
    if rtsp_transport == "rtsp" and not tpl:
        name = mower_profile_name(cfg)
        sys.exit(f"ERROR: mower.transport=rtsp but profiles.{name}.rtsp_url is "
                 f"not set in the config (or drop --transport rtsp and use "
                 f"--rtsp NAME=rtsp://... instead)")
    url_base = mower_base_url(cfg)
    stream_tpl = get_dotted(cfg, "mower.stream_path", "/stream?topic={topic}&type=mjpeg")
    snap_tpl = get_dotted(cfg, "mower.snapshot_path", "/snapshot?topic={topic}")
    specs: list[StreamSpec] = []
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or "topic" not in cam:
            continue
        mode = str(cam.get("mode", "stream"))
        rate = float(cam.get("rate_hz", 1.0))
        topic = str(cam["topic"])
        cam_tpl = str(cam.get("rtsp") or "")
        if rtsp_transport == "rtsp" or cam_tpl:
            specs.append(StreamSpec(
                name=name,
                url=build_rtsp_url(cam_tpl or tpl, mower_host(cfg), name, topic),
                mode="rtsp", rate_hz=rate, topic=topic))
            continue
        specs.append(StreamSpec(
            name=name,
            url=build_stream_url(url_base, snap_tpl if mode == "snapshot" else stream_tpl,
                                 topic),
            mode=mode,
            rate_hz=rate,
            topic=topic,
        ))
    return specs


def as_snapshot(cfg: dict, spec: StreamSpec) -> StreamSpec:
    """Switch a spec to snapshot mode, rebuilding the URL when it has a topic.

    RTSP specs are left alone: there is no HTTP snapshot endpoint for them and
    the RTSP loop already keeps one frame per period.
    """
    if spec.mode in ("snapshot", "rtsp"):
        return spec
    url = spec.url
    if spec.topic:
        url = build_stream_url(mower_base_url(cfg),
                               get_dotted(cfg, "mower.snapshot_path",
                                          "/snapshot?topic={topic}"), spec.topic)
    return dataclasses.replace(spec, mode="snapshot", url=url)


def parse_rtsp_specs(items: list[str], default_rate: float = 1.0) -> list[StreamSpec]:
    """``--rtsp name=rtsp://...`` entries; a bare URL becomes cam1, cam2, ...

    The bare-URL check runs first because RTSP URLs may themselves contain
    ``=`` (query parameters), which would fool ``partition``.
    """
    specs = []
    for item in items:
        item = item.strip()
        if item.startswith("rtsp://") or "=" not in item:
            url, name = item, f"cam{len(specs) + 1}"
            if not url.startswith("rtsp://"):
                sys.exit(f"ERROR: --rtsp expects an rtsp:// URL, got {url!r}")
        else:
            name, _, url = item.partition("=")
            name, url = name.strip(), url.strip()
        specs.append(StreamSpec(name=name, url=url, mode="rtsp",
                                rate_hz=default_rate))
    return specs


def parse_url_specs(items: list[str]) -> list[StreamSpec]:
    """``--url name=URL`` entries. Bare URLs get a name derived from the host+path."""
    specs = []
    for item in items:
        name, sep, url = item.partition("=")
        if not sep:
            url, name = name, f"cam{len(specs) + 1}"
        mode = "snapshot" if "/snapshot" in url else "stream"
        specs.append(StreamSpec(name=name.strip(), url=url.strip(), mode=mode))
    return specs


def parse_topic_specs(items: list[str], base_url: str,
                      stream_tpl: str, default_rate: float) -> list[StreamSpec]:
    """``--topic name=/ros/topic`` entries, expanded against ``base_url``."""
    specs = []
    for item in items:
        name, sep, topic = item.partition("=")
        if not sep:
            topic, name = name, f"cam{len(specs) + 1}"
        specs.append(StreamSpec(
            name=name.strip(),
            url=build_stream_url(base_url, stream_tpl, topic.strip()),
            mode="stream",
            rate_hz=default_rate,
            topic=topic.strip(),
        ))
    return specs


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe(cfg: dict, specs: list[StreamSpec]) -> int:
    """One frame attempt per source, through the URL capture would use.

    ``mode: stream`` sources are read with a streaming GET and stopped at the
    first JPEG (a plain GET on an MJPEG URL never finishes); ``mode: snapshot``
    sources get one ordinary GET; ``mode: rtsp`` sources run ffmpeg until the
    first frame. Exit 1 if nothing answered.
    """
    print(f"probing {mower_base_url(cfg)}  (profile {mower_profile_name(cfg)}, "
          f"transport {mower_transport(cfg)})")
    connect_to = float(get_dotted(cfg, "mower.connect_timeout_s", 5.0))
    read_to = float(get_dotted(cfg, "mower.read_timeout_s", 30.0))
    rtsp_transport = str(get_dotted(cfg, "capture.rtsp_transport", "tcp"))
    alive = 0
    for spec in specs:
        t0 = time.monotonic()
        try:
            if spec.mode == "stream":
                payload = first_mjpeg_frame(spec.url, connect_to, read_to)
            elif spec.mode == "rtsp":
                payload = first_rtsp_frame(spec.url, transport=rtsp_transport,
                                           connect_timeout=max(connect_to, 10.0),
                                           read_timeout=read_to)
            else:
                payload = first_snapshot(spec.url, connect_to, read_to)
            err = ""
        except Exception as exc:  # connection refused, timeout, HTTP error
            payload, err = None, f"{type(exc).__name__}: {exc}"
        dt = (time.monotonic() - t0) * 1000.0
        # For RTSP the URL *is* the thing to check (ffprobe it); for HTTP the
        # topic name is what the hints below refer to.
        label = spec.url if spec.mode == "rtsp" else (spec.topic or spec.url)
        if payload:
            alive += 1
            w, h = image_size_bytes(payload)
            size = f"{w}x{h}  " if w else ""
            print(f"  OK    {spec.name:<12} {label}  {size}"
                  f"{len(payload) / 1024:.0f} KiB  {dt:.0f} ms  [{spec.mode}]")
        else:
            print(f"  EMPTY {spec.name:<12} {label}  "
                  f"{err or 'no JPEG'} after {dt:.0f} ms  [{spec.mode}]")
            if spec.mode == "rtsp":
                print(f"         -> list what the relay publishes: ffprobe {spec.url}")
                print("         -> or check the template: profiles.<profile>.rtsp_url, "
                      "mower.transport")
                print("         -> ffmpeg is missing or broken? (sudo apt install ffmpeg)")
            else:
                for hint in silent_topic_hint(cfg, spec.topic or "<topic>"):
                    print(f"         -> {hint}")
    print(f"\n{alive}/{len(specs)} streams returned a frame")
    return 0 if alive else 1


# ---------------------------------------------------------------------------
# session writer
# ---------------------------------------------------------------------------

class SessionBudget:
    """Byte quota for the whole session, shared by every camera's writer.

    ``capture.quota_bytes`` is a per-SESSION cap: N cameras must not get N
    times the budget.
    """

    def __init__(self, quota_bytes: int) -> None:
        self.quota_bytes = int(quota_bytes)
        self.used = 0
        self.hit = False
        self._lock = threading.Lock()

    def charge(self, nbytes: int) -> bool:
        """Account ``nbytes``; return True when this write crossed the cap."""
        with self._lock:
            self.used += nbytes
            if self.quota_bytes > 0 and not self.hit and self.used >= self.quota_bytes:
                self.hit = True
                return True
            return False


@dataclasses.dataclass
class CameraWriter:
    """Per-camera state: atomic writes, seq counter, dedupe; quota is shared."""
    spec: StreamSpec
    session_dir: Path
    phash_min_distance: int
    budget: SessionBudget
    total_quota_bytes: int
    reencode: bool = False
    jpeg_quality: int = 90
    max_edge: int = 1280

    kept: int = 0
    dropped_dup: int = 0
    dropped_quota: int = 0
    consecutive_dups: int = 0
    last_hash: int = 0
    have_hash: bool = False

    def __post_init__(self) -> None:
        self.cam_dir = self.session_dir / self.spec.name
        self.cam_dir.mkdir(parents=True, exist_ok=True)

    @property
    def quota_hit(self) -> bool:
        return self.budget.hit

    def write(self, payload: bytes, stamp_ns: int) -> dict | None:
        """Return a manifest record, or None when the frame was dropped."""
        if self.budget.hit:
            self.dropped_quota += 1
            return None

        h = dhash(payload)
        if self.have_hash and self.phash_min_distance > 0:
            if hamming(h, self.last_hash) < self.phash_min_distance:
                self.dropped_dup += 1
                self.consecutive_dups += 1
                return None
        self.last_hash, self.have_hash = h, True
        self.consecutive_dups = 0

        if self.reencode:
            payload = maybe_reencode(payload, self.max_edge, self.jpeg_quality)

        self.kept += 1
        seq = self.kept
        width, height = image_size_bytes(payload)
        name = f"{seq:06d}_{stamp_ns}.jpg"
        path = self.cam_dir / name
        tmp = path.with_name(name + ".tmp")
        # Write-then-rename: a crash mid-write must not leave a truncated JPEG
        # that the next step happily labels.
        tmp.write_bytes(payload)
        os.replace(tmp, path)

        if self.budget.charge(len(payload)):
            LOG.warning(
                "session quota reached (%s) on '%s' — capture stopping, not failing",
                human_bytes(self.budget.quota_bytes), self.spec.name)
        self._prune_sessions()
        return {
            "seq": seq,
            "camera": self.spec.name,
            "topic": self.spec.topic,
            "url": self.spec.url,
            "stamp_ns": stamp_ns,
            "wall_iso": now_iso(),
            "encoding": "jpeg",
            "width": width,
            "height": height,
            "bytes": len(payload),
            "phash": f"{h:016x}",
            "session": self.session_dir.name,
            "source": self.spec.mode,
            "path": _display_path(path),
        }

    def _prune_sessions(self) -> None:
        """Delete the oldest sessions until total usage fits the cap.

        Never touches the current session: it may be mid-capture and losing it
        silently would look like data corruption much later.
        """
        frames_root = self.session_dir.parent
        if self.total_quota_bytes <= 0:
            return
        total = dir_size(frames_root)
        if total < self.total_quota_bytes:
            return
        for old in sorted(p for p in frames_root.iterdir()
                          if p.is_dir() and p != self.session_dir):
            LOG.warning("pruning old capture session %s (total %s over cap)",
                        old.name, human_bytes(total))
            for child in sorted(old.rglob("*"), reverse=True):
                try:
                    child.unlink() if child.is_file() else child.rmdir()
                except OSError:
                    pass
            try:
                old.rmdir()
            except OSError:
                pass
            total = dir_size(frames_root)
            if total < self.total_quota_bytes:
                return
        LOG.warning("total quota reached but no older session to prune; "
                    "disk usage %s", human_bytes(total))


# ---------------------------------------------------------------------------
# capture loops
# ---------------------------------------------------------------------------

def _open_session(cfg: dict, out_dir: Path) -> tuple[Path, Path]:
    session = get_dotted(cfg, "capture.session") or time.strftime("%Y%m%d-%H%M%S")
    session_dir = out_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, session_dir / "manifest.jsonl"


def run_capture(cfg: dict, specs: list[StreamSpec], out_dir: Path,
                duration_s: float, max_frames: int) -> dict:
    """Capture every source at once, one thread per camera, into one session.

    ``mode: stream`` cameras hold an MJPEG connection and sample off it;
    ``mode: snapshot`` cameras issue one GET per kept frame; ``mode: rtsp``
    cameras get one ffmpeg each, which decimates to the keep rate. Running each
    camera in its own thread means a silent snapshot topic (which blocks for
    up to ``read_timeout_s``) cannot stall the other cameras, and a mixed
    config captures all kinds in the same session.
    """
    session_dir, manifest = _open_session(cfg, out_dir)
    LOG.info("session %s -> %s", session_dir.name, session_dir)

    budget = SessionBudget(int(get_dotted(cfg, "capture.quota_bytes", 2 << 30)))
    phash_min = int(get_dotted(cfg, "capture.phash_min_distance", 6))
    total_quota = int(get_dotted(cfg, "capture.total_quota_bytes", 12 << 30))
    reencode = bool(get_dotted(cfg, "capture.reencode", False))
    quality = int(get_dotted(cfg, "capture.jpeg_quality", 90))
    max_edge = int(get_dotted(cfg, "capture.max_edge", 1280))
    connect_to = float(get_dotted(cfg, "mower.connect_timeout_s", 5.0))
    read_to = float(get_dotted(cfg, "mower.read_timeout_s", 30.0))
    # Dedupe can reject every frame of a static scene, so "kept" plateaus and
    # max_frames is never reached; polling on would burn the whole session
    # re-fetching the same picture from a single-threaded server. Stop a camera
    # after this many consecutive near-duplicate rejects (0 = never).
    dup_cap = int(get_dotted(cfg, "capture.stop_after_dup_run", 5))
    # Give up on a snapshot camera after this many back-to-back failures.
    error_cap = 10

    writers: dict[str, CameraWriter] = {
        spec.name: CameraWriter(
            spec=spec, session_dir=session_dir, phash_min_distance=phash_min,
            budget=budget, total_quota_bytes=total_quota, reencode=reencode,
            jpeg_quality=quality, max_edge=max_edge)
        for spec in specs}

    manifest_lock = threading.Lock()
    deadline = time.monotonic() + duration_s if duration_s > 0 else None

    def should_stop(writer: CameraWriter) -> bool:
        if _STOP.is_set() or budget.hit:
            return True
        if deadline and time.monotonic() > deadline:
            return True
        if max_frames and writer.kept >= max_frames:
            return True
        if dup_cap and writer.consecutive_dups >= dup_cap:
            LOG.info("'%s': %d near-duplicate frames in a row — the scene is "
                     "static, stopping this camera", writer.spec.name, dup_cap)
            return True
        return False

    def keep(writer: CameraWriter, payload: bytes) -> None:
        rec = writer.write(payload, int(time.time() * 1e9))
        if rec is not None:
            with manifest_lock:
                append_jsonl(manifest, rec)

    def stream_loop(spec: StreamSpec) -> None:
        writer = writers[spec.name]
        min_period = 1.0 / spec.rate_hz if spec.rate_hz > 0 else 0.0
        last_keep = 0.0
        try:
            for payload in iter_mjpeg(spec.url, connect_to, read_to):
                if should_stop(writer):
                    break
                # The stream runs at the camera's publish rate; we sample off
                # it. Holding the connection and discarding is far cheaper
                # than reconnecting.
                now = time.monotonic()
                if now - last_keep < min_period:
                    continue
                last_keep = now
                keep(writer, payload)
        except Exception as exc:  # network, HTTP, or a silent topic
            LOG.error("stream '%s' stopped: %s", spec.name, exc)
            LOG.error("  url=%s", spec.url)

    def snapshot_loop(spec: StreamSpec) -> None:
        writer = writers[spec.name]
        period = 1.0 / spec.rate_hz if spec.rate_hz > 0 else 0.0
        errors = 0
        while not should_stop(writer):
            t0 = time.monotonic()
            try:
                payload = first_snapshot(spec.url, connect_to, read_to)
            except Exception as exc:
                errors += 1
                LOG.warning("snapshot '%s' failed (%s); backing off", spec.name, exc)
                if errors >= error_cap:
                    LOG.error("'%s' failed %d times in a row — giving up on it",
                              spec.name, error_cap)
                    return
                payload = None
            else:
                errors = 0
            if payload is not None:
                keep(writer, payload)
            wait = period - (time.monotonic() - t0)
            if wait > 0 and _STOP.wait(wait):
                return

    def rtsp_loop(spec: StreamSpec) -> None:
        """ffmpeg pulls the feed; it already decimates to ``rate_hz``, so every
        frame it hands us goes straight to keep() — dedupe, quota and the
        consecutive-duplicate stop all apply exactly as on the HTTP path.

        RTSP feeds drop (Wi-Fi, mediamtx restarts), so a dead feed reconnects
        with a short backoff and gives up only after ``error_cap`` failures in
        a row, which is the same contract as the other two loops.
        """
        writer = writers[spec.name]
        transport = str(get_dotted(cfg, "capture.rtsp_transport", "tcp"))
        fps = spec.rate_hz if spec.rate_hz > 0 else 0.0
        failures = 0
        while not should_stop(writer):
            try:
                for payload in iter_rtsp(spec.url, fps=fps, transport=transport,
                                         connect_timeout=max(connect_to, 10.0),
                                         read_timeout=read_to):
                    failures = 0
                    if should_stop(writer):
                        return
                    keep(writer, payload)
                failures += 1   # ffmpeg ended on its own: reconnect, with backoff
                LOG.debug("rtsp '%s': feed ended, reconnecting", spec.name)
            except Exception as exc:
                failures += 1
                LOG.warning("rtsp '%s' failed (%s)", spec.name, exc)
            if failures >= error_cap:
                LOG.error("'%s' failed %d times in a row — giving up on it",
                          spec.name, error_cap)
                return
            if _STOP.wait(min(float(failures), 5.0)):
                return

    threads = []
    for spec in specs:
        target = {"snapshot": snapshot_loop, "rtsp": rtsp_loop}.get(spec.mode,
                                                                    stream_loop)
        th = threading.Thread(target=target, args=(spec,), name=f"cap-{spec.name}",
                              daemon=True)
        th.start()
        threads.append(th)

    # Watchdog: return at the deadline even if a socket is wedged (the threads
    # are daemons; a blocked read cannot hold the process open).
    while any(t.is_alive() for t in threads):
        for th in threads:
            th.join(timeout=0.5)
        if _STOP.is_set() or budget.hit or (deadline and time.monotonic() > deadline + 1.0):
            break

    return summarise(cfg, session_dir, manifest, writers)


def run_folder_capture(cfg: dict, folder: Path, out_dir: Path,
                       rate_hz: float, max_frames: int) -> dict:
    """Import existing images and (when enabled) video files.

    No network, no ROS, no mower involved. Dedupe is OFF for the image files —
    those were curated by hand, so dropping half of them would be worse than
    useless — but ON for decoded video frames, which are raw and full of
    near-duplicates, exactly like the live path.
    """
    fcfg = get_dotted(cfg, "capture.folder", {}) or {}
    recursive = bool(fcfg.get("recursive", True))
    extensions = fcfg.get("extensions", [".jpg", ".jpeg", ".png", ".bmp"])
    want_videos = bool(fcfg.get("videos", True))
    cam_name = str(args_folder_camera(cfg))

    session_dir, manifest = _open_session(cfg, out_dir)

    files = list_images(folder, recursive, extensions)
    videos = list_videos(folder, recursive) if want_videos else []
    if not files and not videos:
        if want_videos:
            LOG.error("no images or videos under %s (image extensions=%s)",
                      folder, extensions)
        else:
            LOG.error("nothing to import under %s (image extensions=%s, and %d "
                      "video(s) present but capture.folder.videos is false)",
                      folder, extensions, len(list_videos(folder, recursive)))
        return {"kept": 0, "errors": 1}
    if max_frames and files:
        files = files[:max_frames]
    LOG.info("importing %d images and %d videos from %s",
             len(files), len(videos), folder)

    writer = CameraWriter(
        spec=StreamSpec(name=cam_name, url=str(folder), mode="snapshot",
                        rate_hz=rate_hz, topic=""),
        # Dedupe OFF for an import: the user already curated these files, and
        # silently dropping half of a hand-assembled set is worse than useless.
        session_dir=session_dir,
        phash_min_distance=int(get_dotted(cfg, "capture.phash_min_distance", 6)),
        budget=SessionBudget(int(get_dotted(cfg, "capture.quota_bytes", 2 << 30))),
        total_quota_bytes=int(get_dotted(cfg, "capture.total_quota_bytes", 12 << 30)),
    )
    writer.phash_min_distance = 0

    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    next_at = time.monotonic()
    for path in files:
        if _STOP.is_set():
            break
        if period:
            sleep_for = next_at - time.monotonic()
            if sleep_for > 0 and _STOP.wait(sleep_for):
                break
            next_at = time.monotonic() + period
        try:
            payload = path.read_bytes()
        except OSError as exc:
            LOG.warning("cannot read %s: %s", path, exc)
            continue
        rec = writer.write(payload, int(path.stat().st_mtime * 1e9))
        if rec is not None:
            rec["original_name"] = path.name
            rec["source"] = "folder"
            append_jsonl(manifest, rec)

    if videos:
        _decode_videos(videos, writer, manifest, cfg, video_rate=float(
            fcfg.get("video_rate_hz", 2.0)), max_frames=max_frames)

    return summarise(cfg, session_dir, manifest, {cam_name: writer}, folder=True)


def _decode_videos(videos: list[Path], writer: CameraWriter, manifest: Path,
                   cfg: dict, video_rate: float, max_frames: int) -> None:
    """Feed decoded video frames through the same writer as everything else.

    ``capture.folder.videos`` is the switch (config) — with it false a folder
    import behaves exactly as it always has. One ffmpeg per file, frames at
    ``capture.folder.video_rate_hz`` frames of video per second.
    """
    if not ffmpeg_available():
        LOG.error("%d video(s) skipped: ffmpeg is not installed "
                  "(sudo apt install ffmpeg)", len(videos))
        return
    # Raw video is the case phash dedupe exists for: turn it back on here and
    # leave it off for the hand-picked images imported above.
    writer.phash_min_distance = int(get_dotted(cfg, "capture.phash_min_distance", 6))
    for vpath in videos:
        if _STOP.is_set() or (max_frames and writer.kept >= max_frames):
            break
        LOG.info("decoding %s at %g fps", vpath, video_rate)
        try:
            for payload in iter_video(vpath, fps=video_rate):
                rec = writer.write(payload, int(time.time() * 1e9))
                if rec is not None:
                    rec["original_name"] = vpath.name
                    rec["source"] = "video"
                    append_jsonl(manifest, rec)
                if _STOP.is_set() or (max_frames and writer.kept >= max_frames):
                    break
        except Exception as exc:  # corrupt file, ffmpeg failure
            LOG.error("%s: %s", vpath.name, exc)
    writer.phash_min_distance = 0


def args_folder_camera(cfg: dict) -> str:
    return get_dotted(cfg, "capture.folder.camera_name", "folder")


def summarise(cfg: dict, session_dir: Path, manifest: Path, writers: dict,
              folder: bool = False) -> dict:
    kept = sum(w.kept for w in writers.values())
    dup = sum(w.dropped_dup for w in writers.values())
    quota = sum(w.dropped_quota for w in writers.values())
    on_disk = dir_size(session_dir)
    print()
    print(f"session dir : {session_dir}")
    print(f"manifest    : {manifest}")
    for name, w in writers.items():
        print(f"  {name:<12} kept={w.kept:<6} dup={w.dropped_dup:<6} "
              f"quota_skips={w.dropped_quota}")
    print(f"total kept={kept}  near-duplicates dropped={dup}  quota skips={quota}")
    print(f"on disk: {human_bytes(on_disk)}")
    if kept == 0:
        print("\nNo frames kept. Common causes:")
        if not folder:
            modes = {w.spec.mode for w in writers.values()}
            if modes == {"rtsp"}:
                print(f"  - (profile {mower_profile_name(cfg)}, transport rtsp)")
                print("  - the RTSP path is wrong — ask the relay what it publishes:")
                print(f"      ffprobe {next(iter(writers.values())).spec.url}")
                print("  - profiles.<profile>.rtsp_url does not match the relay's paths")
                print("  - ffmpeg failed: its own message is in the log above "
                      "(run with --probe)")
            else:
                print(f"  - (profile {mower_profile_name(cfg)})")
                for hint in silent_topic_hint(cfg):
                    print(f"  - {hint}")
                print("  - the stream URL is wrong (run with --probe)")
        print("  - phash_min_distance too high for a moving scene (try 4, or 0)")
    else:
        print("\nnext: python3 scripts/02_llm_label.py "
              f"--images {session_dir}")
    return {"kept": kept, "dropped_dup": dup, "dropped_quota": quota,
            "session_dir": str(session_dir), "errors": 0 if kept else 1}


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.")
    add_common_args(parser)

    src = parser.add_argument_group("source (pick one)")
    src.add_argument("--base-url", help="mower web_video_server base, e.g. "
                                        "http://192.168.1.105:8181 (overrides "
                                        "mower.base_url for every source)")
    src.add_argument("--profile", choices=["ros1", "ros2"],
                     help="mower stack: ros1 = vendor firmware (8181), ros2 = the "
                          "ROS 2 port (8080); default: mower.profile")
    src.add_argument("--transport", choices=["http", "rtsp"],
                     help="how to pull the profile's cameras: http = "
                          "web_video_server MJPEG (default), rtsp = the mediamtx "
                          "relay, decoded by ffmpeg; default: mower.transport")
    src.add_argument("--topic", action="append", default=[], metavar="NAME=TOPIC",
                     help="ROS image topic on the mower (repeatable)")
    src.add_argument("--url", action="append", default=[], metavar="NAME=URL",
                     help="full MJPEG or snapshot URL (repeatable)")
    src.add_argument("--rtsp", action="append", default=[], metavar="NAME=URL",
                     help="explicit RTSP camera, e.g. "
                          "left_oa=rtsp://192.168.1.105:8554/left_oa (repeatable; "
                          "needs ffmpeg)")
    src.add_argument("--folder", help="import images from this local folder instead")
    src.add_argument("--cameras", help="comma list of cameras from the config "
                                       "(default: all of them)")

    parser.add_argument("--out", help="output dir (default: capture.out_dir)")
    parser.add_argument("--session", help="session name (default: timestamp)")
    parser.add_argument("--duration", type=float, help="seconds (0 = until Ctrl-C)")
    parser.add_argument("--rate", type=float, help="keep rate per stream, Hz")
    parser.add_argument("--max-frames", type=int, help="max frames per camera")
    parser.add_argument("--snapshot", action="store_true",
                        help="force snapshot mode (one HTTP GET per frame; "
                             "no effect on RTSP sources)")
    parser.add_argument("--probe", action="store_true",
                        help="check each stream for a frame, then exit")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config(args.config, args.overrides)

    if args.session:
        cfg.setdefault("capture", {})["session"] = args.session
    # Write CLI choices into cfg so every consumer (stream specs, snapshot
    # URLs, probe) sees the same base URL rather than only some of them.
    if args.profile:
        cfg.setdefault("mower", {})["profile"] = args.profile
    if args.base_url:
        cfg.setdefault("mower", {})["base_url"] = args.base_url.rstrip("/")
    if args.transport:
        cfg.setdefault("mower", {})["transport"] = args.transport

    # Build the source list.
    specs: list[StreamSpec] = []
    if args.folder:
        folder = Path(args.folder).expanduser()
        if not folder.is_dir():
            LOG.error("folder not found: %s", folder)
            return 2
        if args.probe:
            recursive = bool(get_dotted(cfg, "capture.folder.recursive", True))
            count = len(list_images(folder, recursive,
                                    get_dotted(cfg, "capture.folder.extensions",
                                               [".jpg"])))
            vids = (list_videos(folder, recursive) if bool(get_dotted(
                cfg, "capture.folder.videos", True)) else [])
            print(f"{count} images and {len(vids)} videos under {folder}")
            return 0 if (count or vids) else 1
        out = Path(args.out).expanduser() if args.out else cfg_path(cfg, "capture.out_dir")
        rate = args.rate if args.rate is not None else float(
            get_dotted(cfg, "capture.folder.rate_hz", 5.0))
        maxf = args.max_frames if args.max_frames is not None else int(
            get_dotted(cfg, "capture.max_frames_per_camera", 500))
        LOG.info("capturing classes: %s", ", ".join(class_names(cfg)))
        return run_folder_capture(cfg, folder, out, rate, maxf)["errors"]

    if args.rtsp:
        specs.extend(parse_rtsp_specs(args.rtsp, float(args.rate or 1.0)))
    if args.url:
        specs.extend(parse_url_specs(args.url))
    if args.topic:
        base = mower_base_url(cfg)
        tpl = get_dotted(cfg, "mower.stream_path", "/stream?topic={topic}&type=mjpeg")
        specs.extend(parse_topic_specs(args.topic, base, tpl,
                                       float(args.rate or 1.0)))
    if not specs:
        specs = stream_from_config(cfg)
        if args.cameras:
            wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}
            specs = [s for s in specs if s.name in wanted]
            if not specs:
                LOG.error("no configured camera matched %s", sorted(wanted))
                return 2
    if not specs:
        LOG.error("no sources. Use --rtsp, --topic, --url or --folder, or check the "
                  "profiles.<mower.profile>.cameras block in the config.")
        return 2

    if any(s.mode == "rtsp" for s in specs) and not ffmpeg_available():
        LOG.error("RTSP source requested but ffmpeg is not installed — "
                  "sudo apt install ffmpeg (or brew install ffmpeg)")
        return 2

    if args.rate is not None:
        specs = [dataclasses.replace(spec, rate_hz=args.rate) for spec in specs]
    if args.snapshot:
        specs = [as_snapshot(cfg, spec) for spec in specs]

    if args.probe:
        return probe(cfg, specs)

    limit = int(get_dotted(cfg, "mower.max_concurrent_sources", 2))
    n_streams = sum(1 for s in specs if s.mode == "stream")
    if n_streams > limit:
        LOG.warning("%d held streams requested but mower.max_concurrent_sources=%d. "
                    "web_video_server is roughly single-threaded: expect contention.",
                    n_streams, limit)
        LOG.warning("  use --snapshot for one of them, or raise the limit if it holds up.")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    duration = args.duration if args.duration is not None else float(
        get_dotted(cfg, "capture.duration_s", 60))
    maxf = args.max_frames if args.max_frames is not None else int(
        get_dotted(cfg, "capture.max_frames_per_camera", 500))
    out = Path(args.out).expanduser() if args.out else cfg_path(cfg, "capture.out_dir")
    out.mkdir(parents=True, exist_ok=True)

    LOG.info("capturing classes: %s", ", ".join(class_names(cfg)))
    result = run_capture(cfg, specs, out, duration, maxf)
    return int(result.get("errors", 0))


if __name__ == "__main__":
    sys.exit(main())
