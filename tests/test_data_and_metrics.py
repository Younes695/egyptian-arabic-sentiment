import os
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egysent.data import FILES, LABELS, load_splits, read_tweets  # noqa: E402
from egysent.metrics import confusion, summarize  # noqa: E402


def write_fake_astd(folder: Path) -> None:
    (folder / FILES["tweets"]).write_text(
        "حلو اوي\tPOS\nوحش جدا\tNEG\nحلو بس غالي\tNEUTRAL\nالمباراة الساعة 9\tOBJ\nممتاز\tPOS\n",
        encoding="utf-8",
    )
    (folder / FILES["train"]).write_text("0\n1\n2", encoding="utf-8")
    (folder / FILES["validation"]).write_text("3", encoding="utf-8")
    (folder / FILES["test"]).write_text("4\n", encoding="utf-8")


def test_splits_follow_the_index_files(tmp_path):
    write_fake_astd(tmp_path)
    splits = load_splits(tmp_path)
    assert [e.label for e in splits["train"]] == ["positive", "negative", "mixed"]
    assert [e.text for e in splits["validation"]] == ["المباراة الساعة 9"]
    assert [e.label for e in splits["test"]] == ["positive"]


def test_overlapping_splits_are_rejected(tmp_path):
    write_fake_astd(tmp_path)
    (tmp_path / FILES["test"]).write_text("0", encoding="utf-8")
    with pytest.raises(ValueError):
        load_splits(tmp_path)


def test_unknown_label_is_rejected(tmp_path):
    (tmp_path / "t.txt").write_text("نص\tHAPPY\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_tweets(tmp_path / "t.txt")


def test_summarize_reports_macro_f1_and_per_class_scores():
    gold = ["positive", "negative", "objective", "objective"]
    pred = ["positive", "objective", "objective", "objective"]
    result = summarize(gold, pred)
    assert result["accuracy"] == 0.75
    assert result["f1"]["positive"] == 1.0
    assert result["f1"]["negative"] == 0.0
    assert result["f1"]["objective"] == pytest.approx(0.8)
    # "mixed" never appears, so it scores 0 and pulls macro-F1 down: (1 + 0 + 0 + 0.8) / 4.
    assert result["macro_f1"] == pytest.approx(0.45)


def test_confusion_rows_are_true_labels():
    matrix = confusion(["positive", "negative"], ["positive", "positive"])
    assert matrix[LABELS.index("positive")][LABELS.index("positive")] == 1
    assert matrix[LABELS.index("negative")][LABELS.index("positive")] == 1


# Checks the real dataset when its files are available locally (set ASTD_DIR).
ASTD_DIR = os.environ.get("ASTD_DIR")


@pytest.mark.skipif(not ASTD_DIR, reason="set ASTD_DIR to a folder with the downloaded ASTD files")
def test_real_astd_split_sizes_and_labels():
    splits = load_splits(ASTD_DIR)
    assert {k: len(v) for k, v in splits.items()} == {"train": 6008, "validation": 1999, "test": 1999}
    everything = Counter(e.label for split in splits.values() for e in split)
    assert everything == {"objective": 6691, "negative": 1684, "mixed": 832, "positive": 799}
