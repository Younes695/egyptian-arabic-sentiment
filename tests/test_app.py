"""The demo app loads a model folder and returns one score per class (tiny random model)."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("gradio")
pytest.importorskip("torch")

from test_train_smoke import tiny_model_and_tokenizer  # noqa: E402


def test_predict_returns_a_score_per_class(tmp_path, monkeypatch):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    monkeypatch.setenv("MODEL_ID", str(tmp_path / "model"))

    app = importlib.import_module("app")
    scores = app.predict("حلو اوي")

    assert set(scores) == set(app.DISPLAY.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert app.predict("   ") == {}
