"""02: alias mapping and reply parsing."""

from __future__ import annotations

import json


def _labeler(labeler, names):
    cfg = labeler.LabelerConfig(
        base_url="http://127.0.0.1:9/v1", model="m", api_key="k", timeout_s=1,
        max_retries=1, retry_backoff_s=0, temperature=0, max_tokens=10, top_p=1,
        response_format="none", max_edge=1024, jpeg_quality=85, system_prompt="",
        user_prompt="{view}{condition}", condition="", view_hint="")
    return labeler.Labeler(cfg, names)


def reply(obj, pretty=False):
    text = json.dumps(obj, indent=2 if pretty else None)
    return {"choices": [{"message": {"content": text}}]}


def test_aliases_map_onto_vocabulary(labeler, names):
    lab = _labeler(labeler, names)
    out = lab.parse_reply(reply({"boxes": [
        {"class": "Rock", "box": [1, 2, 3, 4]},
        {"class": "trash  can", "box": [1, 2, 3, 4]},
        {"class": "charging station", "box": [1, 2, 3, 4]},
        {"class": "robot", "box": [1, 2, 3, 4]}]}))
    assert [b["class"] for b in out["boxes"]] == ["stone", "trashbin", "dock"]
    assert out["out_of_vocabulary"] == ["robot"]


def test_alias_table_respects_custom_vocabulary(labeler):
    al = labeler.build_aliases(["stone", "dog"], {"boulder": "stone", "x": "nope"})
    assert al["rock"] == "stone" and al["boulder"] == "stone"
    assert "trash can" not in al and "x" not in al


def test_pretty_json_is_pure(labeler, names):
    lab = _labeler(labeler, names)
    out = lab.parse_reply(reply({"boxes": []}, pretty=True))
    assert out["ok"] and not any("not pure JSON" in n for n in out["notes"])
    fenced = {"choices": [{"message": {"content": "```json\n{\"boxes\": []}\n```"}}]}
    assert any("not pure JSON" in n for n in lab.parse_reply(fenced)["notes"])


def test_encode_image_reports_sent_and_original_size(labeler, names, tmp_path):
    from conftest import make_jpeg
    lab = _labeler(labeler, names)
    img = make_jpeg(tmp_path / "big.jpg", size=(1920, 1080))
    _url, sent, orig = lab.encode_image(img)
    assert orig == (1920, 1080) and sent == (1024, 576)
