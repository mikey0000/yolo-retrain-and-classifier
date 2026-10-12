"""01: profiles, base URL plumbing, shared session quota, MJPEG probe, RTSP."""

from __future__ import annotations

import http.server
import io
import os
import threading

import pytest


def test_profile_ports_and_explicit_base_url(common, cfg):
    cfg["mower"]["base_url"] = None
    cfg["mower"]["profile"] = "ros1"
    assert common.mower_base_url(cfg) == "http://192.168.1.105:8181"
    cfg["mower"]["profile"] = "ros2"
    assert common.mower_base_url(cfg) == "http://192.168.1.105:8080"
    cfg["mower"]["base_url"] = "http://10.0.0.5:9000/"
    assert common.mower_base_url(cfg) == "http://10.0.0.5:9000"


def test_profiles_default_cameras_and_hints_differ(common, capture, cfg):
    cfg["mower"]["profile"] = "ros1"
    assert list(common.camera_presets(cfg))[:2] == ["left_oa", "right_oa"]
    cfg["mower"]["profile"] = "ros2"
    # ROS 2: only the vio right eye by default, paired with the onboard detector.
    specs = capture.stream_from_config(cfg)
    assert [(s.name, s.topic, s.reference) for s in specs] == [
        ("front", "/vio/right/image_raw", "/ai/det/image_annotated")]
    every = {s.name for s in capture.stream_from_config(cfg, include_disabled=True)}
    assert {"front", "left_oa", "right_oa", "rear"} <= every
    assert any("web_video_server:=true" in h for h in common.silent_topic_hint(cfg))
    cfg["mower"]["profile"] = "ros1"
    hints = common.silent_topic_hint(cfg, "/rear_camera/image_raw")
    assert any("rosservice call /rear_camera/start_capture" in h for h in hints)


def test_specs_use_base_url_and_mode_specific_path(capture, cfg):
    cfg["mower"]["profile"] = "ros1"
    cfg["mower"]["base_url"] = "http://h:1"
    specs = {s.name: s for s in capture.stream_from_config(cfg)}
    assert specs["left_oa"].url == "http://h:1/snapshot?topic=/left_oa_camera/image_raw"
    assert specs["front"].url.startswith("http://h:1/stream?topic=/vio/right/image_raw")
    snap = capture.as_snapshot(cfg, specs["front"])
    assert snap.mode == "snapshot" and snap.url == "http://h:1/snapshot?topic=/vio/right/image_raw"


def test_session_budget_is_shared(capture):
    budget = capture.SessionBudget(100)
    assert budget.charge(60) is False
    assert budget.charge(60) is True and budget.hit
    assert budget.charge(1) is False  # reported once


def _jpeg(color):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def mjpeg_server():
    """A local endless multipart/x-mixed-replace server (never closes)."""
    frames = [_jpeg((i * 40, 0, 0)) for i in range(5)]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=frame")
            self.end_headers()
            try:
                i = 0
                while True:
                    f = frames[i % len(frames)]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
                    self.wfile.flush()
                    i += 1
                    if i > 200:
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/stream"
    srv.shutdown()


def test_probe_of_an_endless_stream_returns_first_frame(common, mjpeg_server):
    payload = common.first_mjpeg_frame(mjpeg_server, 2.0, 2.0)
    assert payload and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")


# --- rtsp -------------------------------------------------------------------

def _stub_ffmpeg(tmp_path, monkeypatch, script: str) -> None:
    """Put a fake ffmpeg first on PATH: no network, no real decoding."""
    stub = tmp_path / "ffmpeg"
    stub.write_text("#!/bin/sh\n" + script)
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def test_transport_rtsp_builds_profile_urls(cfg, capture):
    cfg["mower"]["profile"] = "ros2"
    cfg["mower"]["transport"] = "rtsp"
    specs = {s.name: s for s in capture.stream_from_config(cfg, include_disabled=True)}
    assert set(specs) == {"front", "left_oa", "right_oa", "rear"}
    assert all(s.mode == "rtsp" for s in specs.values())
    assert specs["left_oa"].url == "rtsp://192.168.1.105:8554/left_oa"
    assert specs["left_oa"].rate_hz == 1.0          # keep rate survives the switch
    assert specs["left_oa"].topic == "/left_oa_camera/image_raw"


def test_rtsp_transport_without_a_template_exits_clearly(cfg, capture):
    cfg["mower"]["profile"] = "ros1"                # has no rtsp_url
    cfg["mower"]["transport"] = "rtsp"
    with pytest.raises(SystemExit, match="rtsp_url"):
        capture.stream_from_config(cfg)


def test_a_camera_may_opt_into_rtsp_individually(cfg, capture):
    cfg["mower"]["profile"] = "ros2"
    cfg["mower"]["transport"] = "http"
    cfg["profiles"]["ros2"]["cameras"]["rear"]["rtsp"] = "rtsp://{host}:8554/{name}"
    specs = {s.name: s for s in capture.stream_from_config(cfg, include_disabled=True)}
    assert specs["rear"].mode == "rtsp"
    assert specs["rear"].url == "rtsp://192.168.1.105:8554/rear"
    assert specs["left_oa"].mode == "stream"        # the rest stay on HTTP


def test_parse_rtsp_specs_handles_names_and_query_equals(capture):
    named, bare = capture.parse_rtsp_specs(
        ["left_oa=rtsp://192.168.1.105:8554/left_oa",
         "rtsp://192.168.1.105:8554/rear?transport=tcp"],
        default_rate=2.0)
    assert (named.name, named.mode, named.rate_hz) == ("left_oa", "rtsp", 2.0)
    assert bare.name == "cam2"                      # bare URL, '=' in its query
    assert bare.url.endswith("?transport=tcp")
    with pytest.raises(SystemExit):
        capture.parse_rtsp_specs(["http://192.168.1.105:8181/stream"])


def test_rtsp_decodes_frames_and_probe_returns_the_first(common, tmp_path,
                                                         monkeypatch):
    _stub_ffmpeg(tmp_path, monkeypatch,
                 "printf '\\377\\330\\377LEFT\\377\\331'\n"
                 "printf '\\377\\330\\377RIGHT\\377\\331'\n")
    assert common.ffmpeg_available()
    frames = list(common.iter_rtsp("rtsp://192.0.2.1/left_oa", fps=1.0,
                                   connect_timeout=2.0))
    assert frames == [b"\xff\xd8\xffLEFT\xff\xd9", b"\xff\xd8\xffRIGHT\xff\xd9"]
    assert common.first_rtsp_frame("rtsp://192.0.2.1/left_oa",
                                   connect_timeout=2.0) == frames[0]


def test_rtsp_failures_carry_ffmpegs_own_diagnostics(common, tmp_path,
                                                     monkeypatch):
    _stub_ffmpeg(tmp_path, monkeypatch,
                 'echo "401 Unauthorized (auth failed)" >&2\nexit 1\n')
    with pytest.raises(RuntimeError, match="exited 1.*401 Unauthorized"):
        list(common.iter_rtsp("rtsp://192.0.2.1/left_oa", connect_timeout=2.0))


def test_rtsp_says_so_when_ffmpeg_is_missing(common, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        list(common.iter_rtsp("rtsp://192.0.2.1/left_oa"))


# --- folder import: videos --------------------------------------------------

def _noise_jpeg(seed: int, size=(64, 48)) -> bytes:
    """A frame whose phash differs from any other noise frame."""
    import random

    from PIL import Image
    rng = random.Random(seed)
    buf = io.BytesIO()
    img = Image.new("RGB", size)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(size[0] * size[1])])
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _manifest_records(result) -> list[dict]:
    import json
    from pathlib import Path
    manifest = Path(result["session_dir"]) / "manifest.jsonl"
    return [json.loads(line) for line in manifest.read_text().splitlines()]


def test_folder_import_decodes_video_files(capture, cfg, tmp_path, monkeypatch):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"never read: ffmpeg is stubbed")
    frames = tmp_path / "frames.bin"
    frames.write_bytes(_noise_jpeg(1) + _noise_jpeg(2))   # two distinct frames
    _stub_ffmpeg(tmp_path, monkeypatch, f"cat {frames}\n")

    cfg["capture"]["folder"]["videos"] = True
    result = capture.run_folder_capture(cfg, folder, tmp_path / "out", 5.0, 0)

    records = _manifest_records(result)
    assert result["errors"] == 0 and result["kept"] == 2
    assert {r["source"] for r in records} == {"video"}
    assert {r["original_name"] for r in records} == {"clip.mp4"}
    assert {r["camera"] for r in records} == {"folder"}


def test_folder_import_without_videos_ignores_them(capture, cfg, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "clip.mp4").write_bytes(b"not decoded")
    cfg["capture"]["folder"]["videos"] = False
    result = capture.run_folder_capture(cfg, folder, tmp_path / "out", 5.0, 0)
    assert result == {"kept": 0, "errors": 1}   # explains itself in the log


def test_llm_base_url_auto_falls_back_to_first_candidate(common, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1/")
    assert common.llm_base_url_candidates() == ["http://127.0.0.1:9/v1"]
    assert common.resolve_llm_base_url("auto", timeout_s=0.2) == "http://127.0.0.1:9/v1"
    assert common.resolve_llm_base_url("http://x:1/v1/") == "http://x:1/v1"
