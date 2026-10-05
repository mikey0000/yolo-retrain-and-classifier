"""03: hand-label parsing, name->id mapping, discovery in dataset/review,
and the end-to-end --include-review path."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_jpeg


def test_yolo_rows_valid_and_normalised(to_yolo):
    rows, errors = to_yolo.parse_yolo_hand_label("0 0.5 0.5 0.2 0.3\n\n21 0.1 0.1 0.05 0.05\n", 22)
    assert errors == []
    assert rows == ["0 0.500000 0.500000 0.200000 0.300000",
                    "21 0.100000 0.100000 0.050000 0.050000"]


def test_yolo_rows_reject_out_of_range_and_names(to_yolo):
    _rows, errors = to_yolo.parse_yolo_hand_label("22 0.5 0.5 0.2 0.2", 22)
    assert errors and "out of range" in errors[0]
    _rows, errors = to_yolo.parse_yolo_hand_label("dog 0.5 0.5 0.2 0.2", 22)
    assert errors and "integer" in errors[0]
    _rows, errors = to_yolo.parse_yolo_hand_label("1 1.5 0.5 0.2 0.2", 22)
    assert errors and "normalised" in errors[0]


def test_yolo_rows_remapped_through_export_class_file(to_yolo, tmp_path, names):
    (tmp_path / "classes.txt").write_text("dog\nperson\n")
    remap, err = to_yolo.review_class_remap(tmp_path, names)
    assert err == "" and remap == {0: 1, 1: 0}
    rows, errors = to_yolo.parse_yolo_hand_label("0 0.5 0.5 0.2 0.2", 22, remap)
    assert errors == [] and rows[0].startswith("1 ")
    (tmp_path / "classes.txt").write_text("dog\nrobot\n")
    _remap, err = to_yolo.review_class_remap(tmp_path, names)
    assert "robot" in err


def test_json_names_map_to_ids(to_yolo, idx):
    data = {"boxes": [{"class": "dock", "box": [0, 0, 500, 500]},
                      {"class_id": 1, "box": [500, 500, 1000, 1000]}]}
    rows, errors = to_yolo.parse_json_hand_label(data, idx)
    assert errors == []
    assert rows[0].startswith(f"{idx['dock']} 0.250000 0.250000 0.500000 0.500000")
    assert rows[1].startswith("1 ")


def test_json_unknown_name_is_rejected_not_class_zero(to_yolo, idx):
    rows, errors = to_yolo.parse_json_hand_label(
        {"boxes": [{"class": "robot", "box": [0, 0, 10, 10]}]}, idx)
    assert rows == [] and "robot" in errors[0]
    _rows, errors = to_yolo.parse_json_hand_label(
        {"boxes": [{"class_id": 99, "box": [0, 0, 10, 10]}]}, idx)
    assert "out of range" in errors[0]


def test_find_hand_label_accepts_cvat_txt(to_yolo, tmp_path):
    review = tmp_path / "dataset" / "review"
    review.mkdir(parents=True)
    (review / "000001_1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (review / "000001_1.reason.json").write_text("{}")
    assert to_yolo.find_hand_label([review], "000001_1") == review / "000001_1.txt"
    assert to_yolo.hand_label_stem(review / "000001_1.reason.json") is None
    assert to_yolo.hand_label_stem(review / "x.label.txt") == "x"
    assert to_yolo.hand_label_stem(review / "classes.txt") is None


def test_load_hand_label_attaches_image_and_counts(to_yolo, tmp_path, names):
    lbl = tmp_path / "a.txt"
    lbl.write_text("1 0.5 0.5 0.2 0.2\n")
    img = make_jpeg(tmp_path / "a.jpg")
    out = to_yolo.load_hand_label(lbl, "a", img, names)
    assert out.decision == "accept" and out.image == img and out.source == "human"
    assert out.class_counts == {"dog": 1}
    empty = tmp_path / "b.txt"
    empty.write_text("")
    neg = to_yolo.load_hand_label(empty, "b", img, names)
    assert neg.decision == "accept" and neg.negative
    noimg = to_yolo.load_hand_label(lbl, "a", None, names)
    assert noimg.decision == "review"


def _session(tmp_path: Path, n: int = 4):
    """frames/S1/front/*.jpg + manifest, labels/S1/raw/*.json (all LLM negatives)."""
    frames = tmp_path / "frames" / "S1"
    labels = tmp_path / "labels" / "S1"
    stems = []
    for i in range(n):
        stem = f"{i + 1:06d}_{1000 + i}"
        stems.append(stem)
        make_jpeg(frames / "front" / f"{stem}.jpg", size=(640, 480))
        with (frames / "manifest.jsonl").open("a") as fh:
            fh.write(json.dumps({"camera": "front", "session": "S1",
                                 "path": str(frames / "front" / f"{stem}.jpg")}) + "\n")
        raw = labels / "raw" / f"{stem}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps({"ok": True, "boxes": [], "sent_width": 640,
                                   "sent_height": 480, "model": "m"}))
    return frames, labels, stems


def test_include_review_folds_hand_label_into_dataset(to_yolo, tmp_path):
    frames, labels, stems = _session(tmp_path)
    dataset = tmp_path / "dataset"
    review = dataset / "review"
    review.mkdir(parents=True)
    (review / f"{stems[0]}.txt").write_text("21 0.5 0.5 0.2 0.2\n")       # CVAT style
    (review / f"{stems[1]}.json").write_text(json.dumps(
        {"boxes": [{"class": "dog", "box": [100, 100, 300, 300]}]}))       # prompt schema

    args = ["--labels", str(labels), "--images", str(frames), "--dataset", str(dataset),
            "--include-review", "--set", "convert.val_fraction=0.5"]
    assert to_yolo.main(args) == 0

    written = {p.stem: p.read_text() for p in (dataset / "labels").rglob("*.txt")}
    assert written[stems[0]].startswith("21 0.500000")
    assert written[stems[1]].startswith("1 ")
    assert written[stems[2]] == ""                        # LLM negative still there
    prov = [json.loads(ln) for ln in (dataset / "provenance.jsonl").read_text().splitlines()]
    src = {r["stem"]: r["source"] for r in prov}
    assert src[stems[0]] == "human" and src[stems[2]] == "llm_suggestion"
    # images are attached (each label has its image)
    imgs = {p.stem for p in (dataset / "images").rglob("*.jpg")}
    assert set(stems) <= imgs


def test_rebuild_empties_splits_and_clean_keeps_hand_labels(to_yolo, tmp_path):
    frames, labels, stems = _session(tmp_path)
    dataset = tmp_path / "dataset"
    review = dataset / "review"
    review.mkdir(parents=True)
    (review / f"{stems[0]}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (review / f"{stems[0]}.jpg").write_bytes((frames / "front" / f"{stems[0]}.jpg").read_bytes())
    (review / "stale.reason.json").write_text("{}")
    make_jpeg(review / "stale.jpg")
    stale = dataset / "images" / "val" / "ghost.jpg"
    make_jpeg(stale)
    (dataset / "notes.tmp").write_text("x")

    base = ["--labels", str(labels), "--images", str(frames), "--dataset", str(dataset)]
    assert to_yolo.main(base + ["--clean"]) == 0
    assert not stale.exists()                              # split leak gone
    assert (review / f"{stems[0]}.txt").exists()           # hand label survives --clean
    assert (review / f"{stems[0]}.jpg").exists()           # ...and its image
    assert not (review / "stale.reason.json").exists()
    assert not (review / "stale.jpg").exists()
    assert not (dataset / "notes.tmp").exists()
