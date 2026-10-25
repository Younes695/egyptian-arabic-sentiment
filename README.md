# Egyptian Arabic Sentiment Analysis

Classifies Egyptian-dialect Arabic text as **positive**, **negative**, **mixed** or **objective**. Compares a classic TF-IDF baseline with fine-tuned Arabic BERT models (CAMeLBERT) on ASTD, a dataset of about 10k Egyptian tweets.

> Training results are being added. The tables below will be filled from `results/metrics.json`.

## Approach

- **Data:** [ASTD](https://github.com/mahmoudnabil/ASTD), with its official 4-class unbalanced train / validation / test split (6,008 / 1,999 / 1,999 tweets). Two thirds of the tweets are objective, so models are compared on **macro-F1**, where every class counts equally.
- **Cleaning:** links and user handles become placeholder words, hashtag words are kept, diacritics and letter variants are normalized, and stretched letters are shortened ("حلوووو" becomes "حلوو"). Emojis and negation words are kept because they carry sentiment.
- **Baseline:** TF-IDF on word 1-2 grams and character 2-5 grams, with class-weighted logistic regression.
- **Fine-tuning:** `CAMeL-Lab/bert-base-arabic-camelbert-da` (dialectal Arabic) and `CAMeL-Lab/bert-base-arabic-camelbert-mix`, trained with a plain PyTorch loop: class-weighted loss, linear warm-up, mixed precision, 3 seeds each.
- **No test-set tuning:** each run keeps the epoch with the best validation macro-F1, and the final model is also chosen on validation. The test split is only used to report results.

## Run it

Training needs a GPU. Open `notebooks/train_colab.ipynb` in Google Colab, choose a T4 GPU and run all cells (about 20 minutes).

Tests (no GPU needed):

```bash
pip install pytest torch transformers scikit-learn
pytest
```

## Project layout

```
src/egysent/
  data.py         download ASTD and load the official splits
  preprocess.py   tweet cleaning shared by all models
  train.py        TF-IDF baseline and the PyTorch fine-tuning loop
  metrics.py      accuracy, macro-F1, per-class F1, confusion matrix
notebooks/
  train_colab.ipynb   full training and evaluation run
tests/            unit tests and a tiny-model smoke test
```

## Licenses

The code is MIT. ASTD is GPL-2.0, so it is downloaded at run time and not redistributed here. CAMeLBERT models are Apache-2.0.
