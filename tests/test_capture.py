"""01: profiles, base URL plumbing, shared session quota, MJPEG probe."""

from __future__ import annotations

import http.server
import io
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


def test_profiles_put_oa_cameras_first_and_hints_differ(common, cfg):
    for prof in ("ros1", "ros2"):
        cfg["mower"]["profile"] = prof
        cams = common.camera_presets(cfg)
        assert list(cams)[:2] == ["left_oa", "right_oa"]
    cfg["mower"]["profile"] = "ros2"
    assert "/vio/right/image_raw" not in {c["topic"] for c in common.camera_presets(cfg).values()}
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
