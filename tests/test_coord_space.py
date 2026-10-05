"""03: coordinate-space resolution per frame and pixel rescale by sent size."""

from __future__ import annotations

import pytest


def box(cls, x1, y1, x2, y2, conf=0.9):
    return {"class": cls, "box": [x1, y1, x2, y2], "confidence": conf}


def test_scale_is_built_from_each_frames_own_size(to_yolo):
    small = to_yolo.CoordSpace.for_frame(to_yolo.NORMALISED, 640, 480)
    large = to_yolo.CoordSpace.for_frame(to_yolo.NORMALISED, 1920, 1080)
    assert (small.scale_x, small.scale_y) == pytest.approx((0.64, 0.48))
    assert (large.scale_x, large.scale_y) == pytest.approx((1.92, 1.08))


def test_unit_space_scales_to_the_frame(to_yolo):
    # Regression: 0-1 used to get scale 1.0, turning every box sub-pixel.
    sp = to_yolo.CoordSpace.for_frame(to_yolo.UNIT, 640, 480)
    assert to_yolo.to_pixel_box([0.5, 0.5, 1.0, 1.0], sp) == [320, 240, 640, 480]


def test_pixels_rescaled_by_original_over_sent(to_yolo):
    sp = to_yolo.CoordSpace.for_frame(to_yolo.PIXELS, 1920, 1080, 1024, 576)
    assert sp.scale_x == pytest.approx(1.875)
    assert sp.scale_y == pytest.approx(1.875)
    assert to_yolo.to_pixel_box([512, 288, 1024, 576], sp) == pytest.approx(
        [960, 540, 1920, 1080])


def test_classify_uses_axis_bound(to_yolo):
    c = to_yolo.CoordSpace.classify
    assert c([box("dog", 0.1, 0.1, 0.4, 0.5)], 640, 480) == to_yolo.UNIT
    assert c([box("dog", -5, 10, 40, 50)], 640, 480) == to_yolo.PIXELS
    assert c([box("dog", 10, 10, 1500, 50)], 1920, 1080) == to_yolo.PIXELS
    assert c([box("dog", 10, 10, 900, 600)], 640, 480) == to_yolo.NORMALISED
    assert c([box("dog", 100, 100, 300, 300)], 640, 480) == to_yolo.AMBIGUOUS
    assert c([], 640, 480) == to_yolo.AMBIGUOUS


def test_resolve_space_name_majority_and_fallbacks(to_yolo):
    A, N, P = to_yolo.AMBIGUOUS, to_yolo.NORMALISED, to_yolo.PIXELS
    assert to_yolo.resolve_space_name([A, A]) == N
    assert to_yolo.resolve_space_name([P, P, N, A]) == P
    assert to_yolo.resolve_space_name([P, N]) == N   # tie -> what the prompt asked


def test_frame_verdict_wins_over_session(to_yolo):
    # Session consensus says pixels, but this frame cannot be pixels (x > width).
    name, how = to_yolo.frame_space_name(
        [box("dog", 10, 10, 900, 600)], 640, 480, None, to_yolo.PIXELS)
    assert (name, how) == (to_yolo.NORMALISED, "frame")
    # An ambiguous frame takes the session's name.
    name, how = to_yolo.frame_space_name(
        [box("dog", 100, 100, 300, 300)], 640, 480, None, to_yolo.PIXELS)
    assert (name, how) == (to_yolo.PIXELS, "session")
    # The explicit override beats both.
    name, how = to_yolo.frame_space_name(
        [box("dog", 10, 10, 900, 600)], 640, 480, to_yolo.PIXELS, to_yolo.NORMALISED)
    assert (name, how) == (to_yolo.PIXELS, "override")


def test_mixed_resolutions_in_one_session(to_yolo, cfg, names, idx):
    """A 640x480 and a 1920x1080 frame in one session: each frame is scaled by
    its own size (the old code scaled by the session's largest frame)."""
    small_rec = {"ok": True, "boxes": [box("dog", 250, 250, 750, 750)],
                 "sent_width": 640, "sent_height": 480}
    large_rec = {"ok": True, "boxes": [box("dog", 250, 250, 750, 750)],
                 "sent_width": 1024, "sent_height": 576}
    for rec, (w, h) in ((small_rec, (640, 480)), (large_rec, (1920, 1080))):
        out = to_yolo.validate_frame("s", None, rec, w, h, names, idx, cfg,
                                     session_space=to_yolo.NORMALISED)
        assert out.decision == "accept", out.reason
        cid, cx, cy, bw, bh = out.boxes[0]
        assert (cx, cy, bw, bh) == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_validate_rescales_pixel_reply_from_sent_space(to_yolo, cfg, names, idx):
    rec = {"ok": True, "boxes": [box("dog", 512, 288, 1024, 576)],
           "sent_width": 1024, "sent_height": 576}
    out = to_yolo.validate_frame("s", None, rec, 1920, 1080, names, idx, cfg)
    assert out.decision == "accept", out.reason
    assert out.coord_space == to_yolo.PIXELS and out.coord_decided_by == "frame"
    _cid, cx, cy, bw, bh = out.boxes[0]
    assert (cx, cy, bw, bh) == pytest.approx((0.75, 0.75, 0.5, 0.5))


def test_sent_size_inferred_for_old_raw_files(to_yolo):
    assert to_yolo.sent_size_of({}, 1920, 1080, 1024) == (1024, 576, True)
    assert to_yolo.sent_size_of({}, 640, 480, 1024) == (640, 480, True)
    assert to_yolo.sent_size_of({"sent_width": 800, "sent_height": 450},
                                1920, 1080, 1024) == (800, 450, False)


def test_downscaled_size_matches_reencode(common):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (1920, 1080)).save(buf, format="JPEG")
    out = common.maybe_reencode(buf.getvalue(), 1024, 85)
    assert common.image_size_bytes(out) == common.downscaled_size(1920, 1080, 1024)
