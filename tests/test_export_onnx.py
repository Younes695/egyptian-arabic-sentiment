"""ONNX export + int8 quantization on a tiny random model: files are written and match PyTorch."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("torch")

from export_onnx import check, export  # noqa: E402
from test_train_smoke import tiny_model_and_tokenizer  # noqa: E402


def test_export_writes_both_files_and_matches_pytorch(tmp_path):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    fp32, int8 = export(model.eval(), tokenizer, tmp_path / "out")

    assert fp32.name == "model.onnx" and int8.name == "model_quantized.onnx"
    report = check(model, tokenizer, fp32, int8, ["حلو اوي", "وحش جدا", "الماتش بكرة", "حلو بس غالي"])
    assert report["texts"] == 4
    assert report["fp32_max_logit_diff"] < 1e-3
    assert report["fp32_same_label"] == 1.0
