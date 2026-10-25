"""Smoke tests for the training code with a tiny, randomly initialized model.

They check that the loops run end to end and return the expected structure; they say
nothing about accuracy. Real training runs on a GPU in Colab. Skipped when torch,
transformers or scikit-learn are not installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from egysent.data import LABELS, Example  # noqa: E402
from egysent.train import FinetuneConfig, class_weight_tensor, finetune, run_baseline  # noqa: E402

SAMPLES = {
    "positive": ["حلو اوي", "ممتاز جدا", "جميل وتحفة", "عجبني خالص"],
    "negative": ["وحش جدا", "زفت ومقرف", "مش عاجبني خالص", "سيء جدا"],
    "mixed": ["حلو بس غالي", "كويس بس بطيء", "جميل بس متاخر", "ممتاز بس زحمة"],
    "objective": ["الماتش الساعة تسعة", "الاجتماع بكرة", "القطار وصل", "النتيجة اتنشرت"],
}


def make_splits():
    examples = [Example(text, label) for label, texts in SAMPLES.items() for text in texts]
    return {"train": examples, "validation": examples[::2], "test": examples[1::2]}


def tiny_model_and_tokenizer(tmp_path):
    letters = sorted({ch for texts in SAMPLES.values() for t in texts for ch in t if ch != " "})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *letters]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
    )
    return transformers.BertForSequenceClassification(config), tokenizer


def test_class_weights_favor_rare_classes():
    train = [Example("x", "objective")] * 6 + [Example("y", "positive")] * 2 + [Example("z", "negative")] * 1 + [Example("w", "mixed")] * 1
    weights = class_weight_tensor(train)
    assert weights[LABELS.index("negative")] > weights[LABELS.index("positive")] > weights[LABELS.index("objective")]


def test_finetune_runs_end_to_end(tmp_path):
    model, tokenizer = tiny_model_and_tokenizer(tmp_path)
    cfg = FinetuneConfig(model_name="tiny-bert", epochs=2, batch_size=4, max_length=16, output_dir=str(tmp_path / "ckpt"))
    logs = []
    result = finetune(cfg, make_splits(), model=model, tokenizer=tokenizer, log=logs.append)

    assert len(result["history"]) == 2 and len(logs) == 2
    assert result["best_epoch"] in (1, 2)
    assert len(result["test_predictions"]) == len(make_splits()["test"])
    assert set(result["test_predictions"]) <= set(LABELS)
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    assert (Path(result["checkpoint"]) / "config.json").exists()


def test_baseline_picks_c_on_validation_and_scores_test():
    result = run_baseline(make_splits(), c_values=(1.0, 10.0))
    assert result["C"] in (1.0, 10.0)
    assert len(result["test_predictions"]) == len(make_splits()["test"])
    assert set(result["test"]["f1"]) == set(LABELS)
