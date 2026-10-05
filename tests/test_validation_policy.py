"""03: lenient/strict policy, OOV routing, confidence band."""

from __future__ import annotations


def box(cls, x1, y1, x2, y2, conf=0.9):
    return {"class": cls, "box": [x1, y1, x2, y2], "confidence": conf}


def rec(*boxes, **extra):
    return {"ok": True, "boxes": list(boxes), "sent_width": 640, "sent_height": 480,
            **extra}


def test_lenient_partial_is_accepted_with_a_note(to_yolo, cfg, names, idx):
    cfg["convert"]["bad_box_policy"] = "lenient"
    r = rec(box("dog", 100, 100, 400, 400), box("dog", 0, 0, 1000, 1000))  # 2nd: whole frame
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg)
    assert out.decision == "accept"
    assert out.note.startswith("partial:")
    assert len(out.labels) == 1 and out.dropped


def test_strict_partial_goes_to_review(to_yolo, cfg, names, idx):
    cfg["convert"]["bad_box_policy"] = "strict"
    r = rec(box("dog", 100, 100, 400, 400), box("dog", 0, 0, 1000, 1000))
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg)
    assert out.decision == "review" and out.reason.startswith("strict policy")


def test_kept_low_confidence_box_goes_to_review(to_yolo, cfg, names, idx):
    r = rec(box("dog", 100, 100, 400, 400, conf=0.45))
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg,
                                 session_space=to_yolo.NORMALISED)
    assert out.decision == "review" and "confidence band" in out.reason


def test_ambiguous_frame_without_session_space_goes_to_review(to_yolo, cfg, names, idx):
    r = rec(box("dog", 100, 100, 400, 400))
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg)
    assert out.decision == "review" and "coordinate space" in out.reason


def test_below_min_conf_is_dropped_not_reviewed(to_yolo, cfg, names, idx):
    r = rec(box("dog", 100, 100, 400, 400), box("cat", 500, 500, 700, 700, conf=0.1))
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg)
    assert out.decision == "accept" and len(out.labels) == 1


def test_out_of_vocabulary_reply_goes_to_review(to_yolo, cfg, names, idx):
    # 02 stripped a "robot" box: the frame must not become a clean negative.
    r = rec(out_of_vocabulary=["robot"])
    out = to_yolo.validate_frame("s", None, r, 640, 480, names, idx, cfg)
    assert out.decision == "review" and "robot" in out.reason


def test_empty_reply_is_negative(to_yolo, cfg, names, idx):
    out = to_yolo.validate_frame("s", None, rec(), 640, 480, names, idx, cfg)
    assert out.decision == "accept" and out.negative and out.labels == []


def test_failed_request_goes_to_review(to_yolo, cfg, names, idx):
    out = to_yolo.validate_frame("s", None, {"ok": False, "error": "timeout"},
                                 640, 480, names, idx, cfg)
    assert out.decision == "review"
