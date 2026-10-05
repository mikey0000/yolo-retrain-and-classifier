"""04 backend detection and the 05 ONNX contract check (pure parts)."""

from __future__ import annotations

import pytest

from conftest import load_script

backend = load_script("_backend")
contract = load_script("_onnx_contract")


@pytest.mark.parametrize("avail,hip,want", [
    (True, None, "cuda"),             # NVIDIA: hip is None but the GPU is usable
    (True, "6.2.41133", "rocm"),
    (False, "6.2.41133", "rocm-unavailable"),
    (False, None, "cpu"),
])
def test_classify_backend(avail, hip, want):
    assert backend.classify_backend(avail, hip) == want


def test_contract_accepts_airockchip_layout():
    outs = contract.expected_outputs(22)
    assert outs[0] == [1, 64, 80, 60] and outs[1] == [1, 22, 80, 60] and outs[8] == [1, 1, 20, 15]
    assert contract.check_shapes([1, 3, 640, 480], outs, 22) == []


def test_contract_rejects_stock_head_and_square_input():
    errs = contract.check_shapes([1, 3, 640, 640], [[1, 26, 6300]], 22)
    assert any("input shape" in e for e in errs)
    assert any("single output is the stock ultralytics head" in e for e in errs)


def test_contract_rejects_wrong_class_count():
    outs = contract.expected_outputs(80)
    errs = contract.check_shapes([1, 3, 640, 480], outs, 22)
    assert errs and "output 1" in errs[0]
