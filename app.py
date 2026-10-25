"""Gradio demo: type Egyptian Arabic text, see the sentiment and the model's confidence.

Runs on a free Hugging Face Space (CPU). The model is loaded from the Hub; set
MODEL_ID to use another one (a local folder works too).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from egysent.preprocess import clean_tweet  # noqa: E402

MODEL_ID = os.getenv("MODEL_ID", "Younes695/egyptian-arabic-sentiment")
DISPLAY = {"positive": "Positive 😊", "negative": "Negative 😠", "mixed": "Mixed 😐", "objective": "Objective (no opinion) 📰"}

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)


def predict(text: str) -> dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {}
    scores = classifier(clean_tweet(text))
    # Older and newer transformers versions nest the list differently.
    scores = scores[0] if scores and isinstance(scores[0], list) else scores
    return {DISPLAY.get(s["label"], s["label"]): float(s["score"]) for s in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Text (Egyptian Arabic)", lines=3, rtl=True, placeholder="اكتب جملة بالمصري..."),
    outputs=gr.Label(label="Sentiment", num_top_classes=4),
    title="Egyptian Arabic Sentiment",
    description=(
        "CAMeLBERT fine-tuned on ASTD (Egyptian tweets). Classes: positive, negative, mixed, objective. "
        "Test macro-F1 is about 0.54, so expect mistakes, especially on mixed opinions and sarcasm."
    ),
    examples=[
        ["الفيلم ده تحفة بجد"],
        ["الخدمة زفت ومش هتعامل معاهم تاني"],
        ["الأكل حلو بس الأسعار غالية اوي"],
        ["الماتش هيبدأ الساعة تسعة"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
