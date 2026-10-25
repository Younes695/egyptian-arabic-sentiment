// Check that Transformers.js (the library the browser demo uses) gives the same labels
// as PyTorch. Input: the JSON written by `export_onnx.py --dump` (cleaned texts + labels).
//
//   npm install @huggingface/transformers@4.3.1
//   node scripts/check_transformersjs.mjs val_preds.json USER/REPO
import { readFileSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";

const [file, repo] = process.argv.slice(2);
const rows = JSON.parse(readFileSync(file, "utf-8"));

for (const dtype of ["q8", "fp32"]) {
  const classify = await pipeline("text-classification", repo, { dtype });
  let same = 0;
  const started = Date.now();
  for (const { text, label } of rows) {
    const [top] = await classify(text);
    if (top.label === label) same += 1;
  }
  const ms = (Date.now() - started) / rows.length;
  console.log(`JS_CHECK dtype=${dtype} same_label=${(same / rows.length).toFixed(4)} (${same}/${rows.length}) avg_ms=${ms.toFixed(1)}`);
}
