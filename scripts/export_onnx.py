"""Export a fine-tuned checkpoint to ONNX (fp32 + int8) for Transformers.js, check it, optionally upload.

    python scripts/export_onnx.py CHECKPOINT OUT_DIR [--upload USER/REPO] [--dump preds.json]

Writes OUT_DIR/onnx/model.onnx and OUT_DIR/onnx/model_quantized.onnx, the file names
Transformers.js loads for dtype "fp32" and "q8". Both are compared with the PyTorch model on
the ASTD validation tweets. Several int8 settings are tried. An int8 file passes when its
validation macro-F1 (against the true labels) is at most MAX_INT8_F1_DROP below PyTorch's;
the smallest passing file is kept. Nothing is uploaded if no setting passes. Label agreement
with PyTorch is reported too, but it isn't the gate: a flipped label can flip to the right one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]
MAX_INT8_F1_DROP = 0.01

# Signed int8 weights times unsigned int8 activations can saturate on CPUs without VNNI
# instructions, which flipped about 11% of labels in the first export (and 56% with
# per-channel weights). Unsigned int8 weights, or 7-bit weights (reduce_range), avoid that.
# "all" also quantizes the embedding lookup (Gather), which roughly halves the file again.
QUANT_OPTIONS = [
    {"name": "all-uint8", "op_types_to_quantize": None, "weight_type": "QUInt8", "reduce_range": False},
    {"name": "matmul-uint8", "op_types_to_quantize": ["MatMul"], "weight_type": "QUInt8", "reduce_range": False},
]


class LogitsOnly(torch.nn.Module):
    """ONNX export wants plain tensors in and out, not a ModelOutput object."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def export_fp32(model, tokenizer, path: Path) -> Path:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(["نص قصير", "نص تاني أطول شوية من الأول"], padding=True, return_tensors="pt")
    dims = {name: {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO} for name in INPUTS}
    torch.onnx.export(
        LogitsOnly(model).eval(),
        tuple(sample[name] for name in INPUTS),
        str(path),
        input_names=INPUTS,
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes=dims,
        external_data=False,  # one self-contained file is simplest for the browser
    )
    # The exporter stores intermediate shape annotations that ONNX's own shape inference
    # (run inside quantize_dynamic) can disagree with; drop them and let it recompute.
    graph = onnx.load(str(path))
    del graph.graph.value_info[:]
    onnx.save(graph, str(path))
    return path


def quantize(fp32: Path, out: Path, option: dict) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(fp32),
        str(out),
        weight_type=getattr(QuantType, option["weight_type"]),
        reduce_range=option["reduce_range"],
        op_types_to_quantize=option["op_types_to_quantize"],
    )
    return out


def reference_logits(model, tokenizer, texts: list[str], batch_size: int = 32):
    """PyTorch logits plus the tokenized batches, so every ONNX file sees identical inputs."""
    model.eval()
    batches, logits = [], []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits.append(model(**{k: enc[k] for k in INPUTS}).logits.numpy())
        batches.append({k: enc[k].numpy().astype(np.int64) for k in INPUTS})
    return batches, np.concatenate(logits)


def onnx_logits(path: Path, batches) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([session.run(None, feeds)[0] for feeds in batches])


def compare(ref: np.ndarray, other: np.ndarray, gold: list[str], id2label: dict) -> dict:
    from egysent.metrics import summarize

    return {
        "max_logit_diff": float(np.abs(other - ref).max()),
        "same_label": float((other.argmax(-1) == ref.argmax(-1)).mean()),
        "macro_f1": summarize(gold, [id2label[i] for i in other.argmax(-1).tolist()])["macro_f1"],
    }


def export_and_check(model, tokenizer, out_dir: Path, texts: list[str], gold: list[str], options=QUANT_OPTIONS) -> dict:
    """Export fp32, try each int8 option, keep the smallest int8 file that passes."""
    onnx_dir, candidates_dir = out_dir / "onnx", out_dir / "candidates"
    fp32 = export_fp32(model, tokenizer, onnx_dir / "model.onnx")
    batches, ref = reference_logits(model, tokenizer, texts)
    id2label = model.config.id2label
    report = {
        "texts": len(texts),
        "pytorch_macro_f1": compare(ref, ref, gold, id2label)["macro_f1"],
        "fp32": {**compare(ref, onnx_logits(fp32, batches), gold, id2label), "mb": round(fp32.stat().st_size / 1e6, 1)},
        "int8_options": [],
        "reference_labels": [id2label[i] for i in ref.argmax(-1).tolist()],
    }
    for option in options:
        path = quantize(fp32, candidates_dir / f"{option['name']}.onnx", option)
        result = {"name": option["name"], **compare(ref, onnx_logits(path, batches), gold, id2label), "mb": round(path.stat().st_size / 1e6, 1)}
        report["int8_options"].append(result)
    passing = [r for r in report["int8_options"] if report["pytorch_macro_f1"] - r["macro_f1"] <= MAX_INT8_F1_DROP]
    if passing:
        chosen = min(passing, key=lambda r: r["mb"])
        shutil.copyfile(candidates_dir / f"{chosen['name']}.onnx", onnx_dir / "model_quantized.onnx")
        report["int8_chosen"] = chosen["name"]
    return report


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from egysent.data import download, load_splits
    from egysent.preprocess import clean_tweet

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--upload", help="Hugging Face model repo to add the onnx/ folder to")
    parser.add_argument("--dump", help="write the cleaned texts and PyTorch labels to this JSON file")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    validation = load_splits(download("data"))["validation"]
    texts = [clean_tweet(e.text) for e in validation]
    report = export_and_check(model, tokenizer, Path(args.out_dir), texts, [e.label for e in validation])
    reference = report.pop("reference_labels")
    print("ONNX_CHECK", json.dumps(report))
    if args.dump:
        Path(args.dump).write_text(
            json.dumps([{"text": t, "label": l} for t, l in zip(texts, reference)], ensure_ascii=False), encoding="utf-8"
        )
    if report["fp32"]["same_label"] < 0.999 or "int8_chosen" not in report:
        sys.exit("ONNX outputs differ too much from PyTorch; not uploading.")

    if args.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(Path(args.out_dir) / "onnx"), path_in_repo="onnx", repo_id=args.upload)
        print(f"UPLOADED onnx/ ({report['int8_chosen']}) to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
