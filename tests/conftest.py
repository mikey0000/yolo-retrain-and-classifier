"""Shared fixtures: the numbered scripts are not importable by name
(``01_capture``), so they are loaded from their files once per session."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

_CACHE: dict[str, object] = {}


def load_script(stem: str):
    """Import ``scripts/<stem>.py`` as a module (cached)."""
    if stem not in _CACHE:
        path = SCRIPTS / f"{stem}.py"
        name = "script_" + stem.lstrip("0123456789_")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _CACHE[stem] = mod
    return _CACHE[stem]


@pytest.fixture(scope="session")
def capture():
    return load_script("01_capture")


@pytest.fixture(scope="session")
def labeler():
    return load_script("02_llm_label")


@pytest.fixture(scope="session")
def to_yolo():
    return load_script("03_to_yolo")


@pytest.fixture(scope="session")
def common():
    import common as mod
    return mod


@pytest.fixture()
def cfg(common):
    """The real pipeline.yaml, freshly loaded (tests may mutate it)."""
    return common.load_config(REPO / "config" / "pipeline.yaml")


@pytest.fixture(scope="session")
def names(common):
    return common.class_names(common.load_config(REPO / "config" / "pipeline.yaml"))


@pytest.fixture(scope="session")
def idx(common, names):
    return common.class_index(names)


def make_jpeg(path: Path, size=(64, 48), color=(10, 120, 30)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path
