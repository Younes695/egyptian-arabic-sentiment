"""Export a fine-tuned checkpoint to ONNX (fp32 + int8) for Transformers.js, check it, optionally upload.

    python scripts/export_onnx.py CHECKPOINT OUT_DIR [--upload USER/REPO]

Writes OUT_DIR/onnx/model.onnx and OUT_DIR/onnx/model_quantized.onnx, the file names
Transformers.js loads for dtype "fp32" and "q8". Before uploading it compares both files
with the PyTorch model on the ASTD validation tweets: logit difference for fp32, and how
often the int8 model predicts the same label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


class LogitsOnly(torch.nn.Module):
    """ONNX export wants plain tensors in and out, not a ModelOutput object."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def export(model, tokenizer, out_dir: Path) -> tuple[Path, Path]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = onnx_dir / "model.onnx", onnx_dir / "model_quantized.onnx"
    sample = tokenizer(["نص قصير", "نص تاني أطول شوية من الأول"], padding=True, return_tensors="pt")
    args = tuple(sample[name] for name in INPUTS)
    dims = {name: {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO} for name in INPUTS}
    torch.onnx.export(
        LogitsOnly(model).eval(),
        args,
        str(fp32),
        input_names=INPUTS,
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes=dims,
        external_data=False,  # one self-contained file is simplest for the browser
    )
    # The exporter stores intermediate shape annotations that ONNX's own shape inference
    # (run inside quantize_dynamic) can disagree with; drop them and let it recompute.
    import onnx

    graph_model = onnx.load(str(fp32))
    del graph_model.graph.value_info[:]
    onnx.save(graph_model, str(fp32))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


def check(model, tokenizer, fp32: Path, int8: Path, texts: list[str], batch_size: int = 32) -> dict:
    import onnxruntime as ort

    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for name, path in (("fp32", fp32), ("int8", int8))}
    max_diff, same_fp32, same_int8, total = 0.0, 0, 0, 0
    ref_labels: list[str] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            ref = model(**{k: enc[k] for k in INPUTS}).logits.numpy()
        feeds = {k: enc[k].numpy().astype(np.int64) for k in INPUTS}
        out32 = sessions["fp32"].run(None, feeds)[0]
        out8 = sessions["int8"].run(None, feeds)[0]
        max_diff = max(max_diff, float(np.abs(out32 - ref).max()))
        same_fp32 += int((out32.argmax(-1) == ref.argmax(-1)).sum())
        same_int8 += int((out8.argmax(-1) == ref.argmax(-1)).sum())
        total += len(ref)
        ref_labels.extend(model.config.id2label[i] for i in ref.argmax(-1).tolist())
    return {
        "reference_labels": ref_labels,
        "texts": total,
        "fp32_max_logit_diff": max_diff,
        "fp32_same_label": same_fp32 / total,
        "int8_same_label": same_int8 / total,
        "fp32_mb": fp32.stat().st_size / 1e6,
        "int8_mb": int8.stat().st_size / 1e6,
    }


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
    fp32, int8 = export(model, tokenizer, Path(args.out_dir))

    texts = [clean_tweet(e.text) for e in load_splits(download("data"))["validation"]]
    report = check(model, tokenizer, fp32, int8, texts)
    reference = report.pop("reference_labels")
    print("ONNX_CHECK", report)
    if args.dump:
        Path(args.dump).write_text(
            json.dumps([{"text": t, "label": l} for t, l in zip(texts, reference)], ensure_ascii=False), encoding="utf-8"
        )
    if report["fp32_same_label"] < 0.999 or report["int8_same_label"] < 0.97:
        sys.exit("ONNX outputs differ too much from PyTorch; not uploading.")

    if args.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(Path(args.out_dir) / "onnx"), path_in_repo="onnx", repo_id=args.upload)
        print(f"UPLOADED onnx/ to https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
