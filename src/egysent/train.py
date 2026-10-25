"""Training code: a TF-IDF + logistic regression baseline and transformer fine-tuning.

The fine-tuning loop is plain PyTorch (no Trainer class) so every step is visible:
tokenize, batch, forward pass, weighted loss, backward pass, scheduler, evaluation.
Meant to run on a GPU (Google Colab); see notebooks/train_colab.ipynb.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import LABEL2ID, LABELS, Example
from .metrics import summarize
from .preprocess import clean_tweet


# ---------------------------------------------------------------- baseline


def run_baseline(splits: dict[str, list[Example]], c_values=(0.3, 1.0, 3.0, 10.0)) -> dict:
    """Word + character n-gram TF-IDF with logistic regression.

    Character n-grams cope with Arabic spelling variation and dialect; class weights
    stop the model from predicting "objective" (two thirds of the data) for everything.
    C is chosen on the validation split; the test split is scored once at the end.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    def texts(split):
        return [clean_tweet(e.text) for e in splits[split]]

    def labels(split):
        return [e.label for e in splits[split]]

    best = None
    for c in c_values:
        model = make_pipeline(
            make_union(
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            ),
            LogisticRegression(C=c, class_weight="balanced", max_iter=3000),
        )
        model.fit(texts("train"), labels("train"))
        val = summarize(labels("validation"), list(model.predict(texts("validation"))))
        if best is None or val["macro_f1"] > best[1]["macro_f1"]:
            best = (c, val, model)

    c, val, model = best
    test_pred = list(model.predict(texts("test")))
    return {
        "model": "TF-IDF + logistic regression",
        "C": c,
        "validation": val,
        "test": summarize(labels("test"), test_pred),
        "test_predictions": test_pred,
    }


# ---------------------------------------------------------------- fine-tuning


@dataclass
class FinetuneConfig:
    model_name: str
    seed: int = 42
    epochs: int = 4
    lr: float = 2e-5
    batch_size: int = 32
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    class_weights: bool = True
    output_dir: str = "checkpoints"


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weight_tensor(train: list[Example]):
    """Inverse-frequency weights ("balanced"): rare classes cost more when missed."""
    import torch

    counts = Counter(e.label for e in train)
    total = sum(counts.values())
    return torch.tensor([total / (len(LABELS) * counts[label]) for label in LABELS], dtype=torch.float)


def _batches(encodings, label_ids, batch_size, shuffle, generator=None):
    import torch

    order = list(range(len(label_ids)))
    if shuffle:
        order = torch.randperm(len(order), generator=generator).tolist()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = {k: v[idx] for k, v in encodings.items()}
        batch["labels"] = torch.tensor([label_ids[i] for i in idx])
        yield batch


def _predict(model, tokenizer, texts: list[str], device, max_length: int, batch_size: int = 64) -> list[str]:
    import torch

    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**enc).logits
            preds.extend(LABELS[i] for i in logits.argmax(-1).tolist())
    return preds


def finetune(cfg: FinetuneConfig, splits: dict[str, list[Example]], model=None, tokenizer=None, log=print) -> dict:
    """Fine-tune `cfg.model_name` on the train split, keep the epoch with the best
    validation macro-F1, then score the test split once with that checkpoint.

    `model` / `tokenizer` can be passed in (used by the tests with a tiny model).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    model.to(device)

    texts = {s: [clean_tweet(e.text) for e in splits[s]] for s in splits}
    gold = {s: [e.label for e in splits[s]] for s in splits}
    train_enc = tokenizer(texts["train"], truncation=True, max_length=cfg.max_length, padding=True, return_tensors="pt")
    train_ids = [LABEL2ID[label] for label in gold["train"]]

    weights = class_weight_tensor(splits["train"]).to(device) if cfg.class_weights else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)
    # Linear warm-up, then linear decay to zero.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(cfg.seed)

    best = {"macro_f1": -1.0}
    best_path = Path(cfg.output_dir) / f"{cfg.model_name.split('/')[-1]}-seed{cfg.seed}"
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for batch in _batches(train_enc, train_ids, cfg.batch_size, shuffle=True, generator=generator):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val = summarize(gold["validation"], _predict(model, tokenizer, texts["validation"], device, cfg.max_length))
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "val_macro_f1": val["macro_f1"]})
        log(f"epoch {epoch}: loss {running / steps_per_epoch:.4f}  val macro-F1 {val['macro_f1']:.4f}  ({time.time() - started:.0f}s)")
        if val["macro_f1"] > best["macro_f1"]:
            best = {**val, "epoch": epoch}
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_path).to(device)
    test_pred = _predict(best_model, tokenizer, texts["test"], device, cfg.max_length)
    return {
        "model": cfg.model_name,
        "seed": cfg.seed,
        "best_epoch": best["epoch"],
        "validation": {k: v for k, v in best.items() if k != "epoch"},
        "test": summarize(gold["test"], test_pred),
        "test_predictions": test_pred,
        "history": history,
        "checkpoint": str(best_path),
    }
