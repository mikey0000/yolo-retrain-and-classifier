"""05 helper: calibration stratified by manifest camera, letterboxed like det_ros."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_script, make_jpeg

calib = load_script("_calib")


def test_letterbox_geometry_matches_mower_rknn():
    # 1920x1080 OA frame into 480x640: scale 0.25 -> 480x270, 185 px bands.
    assert calib.letterbox_geometry(1920, 1080, 480, 640) == (480, 270, 0, 185, 0.25)
    # 640x480 vio frame: scale 0.75 -> 480x360, 140 px bands.
    assert calib.letterbox_geometry(640, 480, 480, 640) == (480, 360, 0, 140, 0.75)
    # ROS 2 publish sizes: OA 960x540 (oa_publish_width) and rear 1280x720 both
    # land exactly where a 1920x1080 frame does.
    assert calib.letterbox_geometry(960, 540, 480, 640) == (480, 270, 0, 185, 0.5)
    assert calib.letterbox_geometry(1280, 720, 480, 640) == (480, 270, 0, 185, 0.375)


@pytest.mark.parametrize("size", [(960, 540), (640, 480), (1280, 720)])
def test_letterbox_image_handles_every_ros2_camera_size(size):
    from PIL import Image
    out = calib.letterbox_image(Image.new("RGB", size, (200, 200, 200)), 480, 640)
    nw, nh, px, py, _ = calib.letterbox_geometry(*size, 480, 640)
    assert out.size == (480, 640)
    assert out.getpixel((240, py - 1)) == (0, 0, 0)      # last band row
    assert out.getpixel((240, py)) == (200, 200, 200)    # first image row
    assert out.getpixel((240, py + nh - 1)) == (200, 200, 200)
    assert out.getpixel((240, py + nh)) == (0, 0, 0)


def test_config_calibration_weights_cover_every_det_ros_camera(cfg):
    weights = cfg["export"]["rknn"]["calib_camera_weights"]
    ros2 = set(cfg["profiles"]["ros2"]["cameras"])
    assert ros2 <= set(weights)
    # front (forward, feeds the costmaps) first, rear (gated) last
    assert weights["front"] > weights["left_oa"] == weights["right_oa"] > weights["rear"] > 0
    q = calib.allocate_quotas({c: 1000 for c in ros2}, 300, weights)
    assert q == {"front": 105, "left_oa": 75, "right_oa": 75, "rear": 45}


def test_letterbox_image_pads_black():
    from PIL import Image
    out = calib.letterbox_image(Image.new("RGB", (1920, 1080), (200, 200, 200)), 480, 640)
    assert out.size == (480, 640)
    assert out.getpixel((240, 10)) == (0, 0, 0)          # top band
    assert out.getpixel((240, 320)) == (200, 200, 200)   # image
    assert out.getpixel((240, 630)) == (0, 0, 0)         # bottom band


def test_quotas_follow_weights_and_redistribute():
    weights = {"left_oa": 0.4, "right_oa": 0.4, "*": 0.2}
    q = calib.allocate_quotas({"left_oa": 500, "right_oa": 500, "front": 500}, 300, weights)
    assert q == {"left_oa": 120, "right_oa": 120, "front": 60}
    # left_oa short: its unmet share moves to the cameras that still have frames
    q = calib.allocate_quotas({"left_oa": 20, "right_oa": 500, "front": 500}, 300, weights)
    assert q["left_oa"] == 20 and sum(q.values()) == 300 and q["right_oa"] > q["front"]
    # absent OA cameras: '*' takes the whole budget
    assert calib.allocate_quotas({"front": 1000}, 300, weights) == {"front": 300}
    # fewer frames than requested
    assert sum(calib.allocate_quotas({"left_oa": 5, "front": 5}, 300, weights).values()) == 10
    # weight 0 means never
    assert calib.allocate_quotas({"a": 10, "b": 10}, 10, {"a": 1.0, "b": 0.0}) == {"a": 10, "b": 0}


def test_select_is_deterministic_and_stratified():
    by_cam = {"left_oa": [Path(f"l{i}.jpg") for i in range(50)],
              "rear": [Path(f"r{i}.jpg") for i in range(50)]}
    a = calib.select_calibration(by_cam, 20, {"left_oa": 0.75, "*": 0.25}, seed=1)
    b = calib.select_calibration(by_cam, 20, {"left_oa": 0.75, "*": 0.25}, seed=1)
    assert a == b and len(a) == 20
    assert sum(1 for p in a if p.name.startswith("l")) == 15


def test_camera_comes_from_manifest_not_parent_dir(tmp_path):
    frames = tmp_path / "frames" / "S1"
    frames.mkdir(parents=True)
    (frames / "manifest.jsonl").write_text(
        json.dumps({"camera": "left_oa", "path": "work/frames/S1/left_oa/000001_5.jpg"}) + "\n")
    cam_map = calib.load_camera_map(tmp_path / "frames")
    # in the dataset the parent dir is the split, which must not be read as a camera
    assert calib.camera_for(Path("dataset/images/train/000001_5.jpg"), cam_map) == "left_oa"
    assert calib.camera_for(Path("dataset/images/train/other.jpg"), cam_map) == "unknown"


def test_main_writes_letterboxed_list(tmp_path):
    frames = tmp_path / "frames" / "S1"
    for i in range(3):
        make_jpeg(frames / "left_oa" / f"{i}.jpg", size=(192, 108))
    (frames / "manifest.jsonl").write_text("".join(
        json.dumps({"camera": "left_oa", "path": f"x/{i}.jpg"}) + "\n" for i in range(3)))
    out, lst = tmp_path / "calib", tmp_path / "calib_list.txt"
    rc = calib.main(["--source", "frames", "--dataset-dir", str(tmp_path / "nope"),
                     "--frames-dir", str(tmp_path / "frames"), "--count", "2",
                     "--weights", '{"left_oa": 1}', "--out-dir", str(out), "--list", str(lst)])
    assert rc == 0
    files = [Path(p) for p in lst.read_text().split()]
    assert len(files) == 2
    from PIL import Image
    with Image.open(files[0]) as im:
        assert im.size == (480, 640)
