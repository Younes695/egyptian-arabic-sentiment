"""Publish the fine-tuned model and the Gradio demo to the Hugging Face account you are logged in with.

Run after `huggingface_hub.notebook_login()` (or `hf auth login`), from the repo root:

    python scripts/publish_hf.py checkpoints/bert-base-arabic-camelbert-mix-seed7

Creates or updates:
  <user>/egyptian-arabic-sentiment        the model, with a model card built from results/metrics.json
  <user>/egyptian-arabic-sentiment-demo   a Gradio Space that loads that model
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, whoami
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Younes695/egyptian-arabic-sentiment"
GRADIO_VERSION = "6.30.0"


def model_card(model_repo: str, results: dict) -> str:
    base = results["selected"]["model"]
    run = next(r for r in results["runs"] if r["model"] == base and r["seed"] == results["selected"]["seed"])
    summary = results["summary_mean_std"]
    f1 = run["test"]["f1"]
    return f"""---
language: ar
license: mit
base_model: {base}
pipeline_tag: text-classification
tags: [arabic, egyptian-arabic, sentiment-analysis, bert]
---

# Egyptian Arabic sentiment (CAMeLBERT fine-tuned on ASTD)

Classifies Egyptian-dialect Arabic text as **positive**, **negative**, **mixed** or **objective**.
Fine-tuned from [`{base}`](https://huggingface.co/{base}) on the
[ASTD](https://github.com/mahmoudnabil/ASTD) tweets, using the official 4-class split.
Code, training notebook and full results: {GITHUB}

## Results on the ASTD test set (1,999 tweets)

| Model | Test macro-F1 | Test accuracy |
|---|---|---|
| TF-IDF + logistic regression | {results['baseline']['test']['macro_f1']:.3f} | {results['baseline']['test']['accuracy']:.3f} |
| CAMeLBERT-DA, 3 seeds | {summary['CAMeL-Lab/bert-base-arabic-camelbert-da']['test_macro_f1']} | {summary['CAMeL-Lab/bert-base-arabic-camelbert-da']['test_accuracy']} |
| CAMeLBERT-Mix, 3 seeds | {summary['CAMeL-Lab/bert-base-arabic-camelbert-mix']['test_macro_f1']} | {summary['CAMeL-Lab/bert-base-arabic-camelbert-mix']['test_accuracy']} |

This checkpoint is the CAMeLBERT-Mix run chosen on validation (seed {run['seed']}): test macro-F1
{run['test']['macro_f1']:.3f}. F1 per class: positive {f1['positive']:.3f}, negative {f1['negative']:.3f},
mixed {f1['mixed']:.3f}, objective {f1['objective']:.3f}.

## Use

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{model_repo}", top_k=None)
clf("الفيلم ده تحفة بجد")
```

For best results, clean the text the same way as in training (`egysent.preprocess.clean_tweet` in the GitHub repo).

## Limitations

- Trained on tweets that are mostly political and from around 2013. Reviews and customer messages will look different.
- Mixed opinions and sarcasm are the weakest cases (mixed F1 about 0.33).
- ASTD is GPL-2.0; check that its terms fit your use before deploying this model commercially.
"""


SPACE_README = """---
title: Egyptian Arabic Sentiment
emoji: 💬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: {gradio}
app_file: app.py
license: mit
---

Demo of an Egyptian Arabic sentiment model (CAMeLBERT fine-tuned on ASTD).
Model: https://huggingface.co/{model_repo} · Code: {github}
"""


def main(checkpoint: str) -> None:
    user = whoami()["name"]
    model_repo = f"{user}/egyptian-arabic-sentiment"
    space_repo = f"{user}/egyptian-arabic-sentiment-demo"
    api = HfApi()
    results = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))

    # 1. Model + tokenizer + model card.
    AutoModelForSequenceClassification.from_pretrained(checkpoint).push_to_hub(model_repo)
    AutoTokenizer.from_pretrained(checkpoint).push_to_hub(model_repo)
    api.upload_file(
        path_or_fileobj=model_card(model_repo, results).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model_repo,
    )

    # 2. Gradio Space: the app, the cleaning code it imports, and its requirements.
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "src" / "egysent").mkdir(parents=True)
        (tmp / "app.py").write_text((ROOT / "app.py").read_text(encoding="utf-8"), encoding="utf-8")
        for name in ("__init__.py", "preprocess.py"):
            src = (ROOT / "src" / "egysent" / name).read_text(encoding="utf-8")
            (tmp / "src" / "egysent" / name).write_text(src, encoding="utf-8")
        (tmp / "requirements.txt").write_text("transformers==5.20.2\ntorch\n", encoding="utf-8")
        (tmp / "README.md").write_text(
            SPACE_README.format(gradio=GRADIO_VERSION, model_repo=model_repo, github=GITHUB), encoding="utf-8"
        )
        api.upload_folder(folder_path=str(tmp), repo_id=space_repo, repo_type="space")
    api.add_space_variable(space_repo, "MODEL_ID", model_repo)

    print(f"MODEL: https://huggingface.co/{model_repo}")
    print(f"SPACE: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
