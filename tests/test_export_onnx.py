"""ONNX export + int8 quantization on a tiny random model: files are written and fp32 matches PyTorch."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("torch")

import export_onnx  # noqa: E402
from test_train_smoke import tiny_model_and_tokenizer  # noqa: E402

TEXTS = ["حلو اوي", "وحش جدا", "الماتش بكرة", "حلو بس غالي"]
GOLD = ["positive", "negative", "objective", "mixed"]


def test_fp32_matches_and_every_int8_option_is_scored(tmp_path):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    report = export_onnx.export_and_check(model.eval(), tokenizer, tmp_path / "out", TEXTS, GOLD)

    assert report["texts"] == 4
    assert report["fp32"]["max_logit_diff"] < 1e-3
    assert report["fp32"]["same_label"] == 1.0
    assert report["fp32"]["macro_f1"] == report["pytorch_macro_f1"]
    assert [o["name"] for o in report["int8_options"]] == [o["name"] for o in export_onnx.QUANT_OPTIONS]
    assert len(report["reference_labels"]) == 4
    assert (tmp_path / "out" / "onnx" / "model.onnx").exists()


def test_smallest_passing_int8_file_is_kept(tmp_path, monkeypatch):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    monkeypatch.setattr(export_onnx, "MAX_INT8_F1_DROP", 1.0)  # every option passes
    report = export_onnx.export_and_check(model.eval(), tokenizer, tmp_path / "out", TEXTS, GOLD)

    smallest = min(report["int8_options"], key=lambda o: o["mb"])["name"]
    assert report["int8_chosen"] == smallest
    assert (tmp_path / "out" / "onnx" / "model_quantized.onnx").exists()


def test_nothing_is_kept_when_no_option_passes(tmp_path, monkeypatch):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    monkeypatch.setattr(export_onnx, "MAX_INT8_F1_DROP", -1.0)  # impossible
    report = export_onnx.export_and_check(model.eval(), tokenizer, tmp_path / "out", TEXTS, GOLD)

    assert "int8_chosen" not in report
    assert not (tmp_path / "out" / "onnx" / "model_quantized.onnx").exists()
