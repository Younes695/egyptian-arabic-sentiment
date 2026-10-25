"""Draw the README charts from results/metrics.json (no model or GPU needed)."""

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
results = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. Test macro-F1 per model: baseline as one bar, fine-tuned models as mean with seed spread.
names = ["TF-IDF +\nlogistic regression", "CAMeLBERT-DA", "CAMeLBERT-Mix"]
means, errors = [results["baseline"]["test"]["macro_f1"]], [0.0]
for model in ("CAMeL-Lab/bert-base-arabic-camelbert-da", "CAMeL-Lab/bert-base-arabic-camelbert-mix"):
    scores = [r["test"]["macro_f1"] for r in results["runs"] if r["model"] == model]
    means.append(st.mean(scores))
    errors.append(st.stdev(scores))
fig, ax = plt.subplots(figsize=(6, 3.6))
bars = ax.bar(names, means, yerr=errors, capsize=6, color=["#9aa4b5", "#2dd4bf", "#0ea5e9"])
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015, f"{m:.3f}", ha="center")
ax.set_ylim(0, 0.65)
ax.set_ylabel("test macro-F1")
ax.set_title("ASTD test set (3 seeds for fine-tuned models)")
fig.tight_layout()
fig.savefig(ROOT / "results" / "model_comparison.png", dpi=160)

# 2. Confusion matrix of the selected model.
cm = results["selected_confusion"]
labels, matrix = cm["labels"], cm["rows_true_columns_predicted"]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(matrix, cmap="Blues")
ax.set_xticks(range(len(labels)), labels, rotation=25)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
peak = max(map(max, matrix))
for i, row in enumerate(matrix):
    for j, v in enumerate(row):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > peak / 2 else "black")
ax.set_title("Selected model, test set")
fig.tight_layout()
fig.savefig(ROOT / "results" / "confusion_matrix.png", dpi=160)
print("saved results/model_comparison.png and results/confusion_matrix.png")
