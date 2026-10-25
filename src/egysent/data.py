"""Load ASTD (Arabic Sentiment Tweets Dataset) with its official 4-class unbalanced splits.

ASTD: ~10k Egyptian-dialect tweets labelled by annotators as positive, negative,
mixed ("NEUTRAL" in the files) or objective. Source and license (GPL-2.0):
https://github.com/mahmoudnabil/ASTD. The data is downloaded at run time and is
not redistributed in this repository.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Pinned to a commit so every run trains and tests on exactly the same data.
ASTD_COMMIT = "0eac69a559bbc38a2cf06266411fa8bf47eed042"
ASTD_BASE_URL = f"https://raw.githubusercontent.com/mahmoudnabil/ASTD/{ASTD_COMMIT}/data"
FILES = {
    "tweets": "Tweets.txt",
    "train": "4class-unbalanced-train.txt",
    "validation": "4class-unbalanced-validation.txt",
    "test": "4class-unbalanced-test.txt",
}

# File label -> name shown to users. ASTD calls the mixed class "NEUTRAL".
LABELS = ["positive", "negative", "mixed", "objective"]
FILE_LABEL_TO_NAME = {"POS": "positive", "NEG": "negative", "NEUTRAL": "mixed", "OBJ": "objective"}
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


@dataclass(frozen=True)
class Example:
    text: str
    label: str  # one of LABELS


def download(folder: str | Path) -> Path:
    """Download the ASTD files into `folder` (skips files already there)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = folder / name
        if not target.exists():
            urllib.request.urlretrieve(f"{ASTD_BASE_URL}/{name}", target)
    return folder


def read_tweets(path: str | Path) -> list[Example]:
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        text, _, file_label = line.rpartition("\t")
        if file_label not in FILE_LABEL_TO_NAME or not text:
            raise ValueError(f"line {line_no}: expected '<tweet>\\t<label>', got {line[:60]!r}")
        examples.append(Example(text=text, label=FILE_LABEL_TO_NAME[file_label]))
    return examples


def read_split_ids(path: str | Path) -> list[int]:
    """Split files list 0-based row numbers into Tweets.txt, one per line."""
    return [int(x) for x in Path(path).read_text(encoding="utf-8").split()]


def load_splits(folder: str | Path) -> dict[str, list[Example]]:
    folder = Path(folder)
    tweets = read_tweets(folder / FILES["tweets"])
    splits = {}
    seen: set[int] = set()
    for split in ("train", "validation", "test"):
        ids = read_split_ids(folder / FILES[split])
        if seen & set(ids):
            raise ValueError(f"{split} shares rows with an earlier split")
        seen |= set(ids)
        splits[split] = [tweets[i] for i in ids]
    return splits
