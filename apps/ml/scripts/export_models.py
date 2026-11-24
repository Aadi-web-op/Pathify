"""Prune, export to ONNX, quantize to int8, and re-evaluate both encoders.

    .venv/bin/python -m scripts.export_models [--only interest|retriever]

Copies the serving files to apps/api/models/ and writes reports/export.json with
sizes, the fp32-vs-int8 output gap, and eval scores of the shipped int8 models.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

from pathify_ml.export import copy_tokenizer, export_onnx, load_fast_tokenizer, onnx_run, prune_vocab, quantize, size_mb
from pathify_ml.interest_data import build, student_eval
from pathify_ml.interest_model import BASE_MODEL, InterestReader, mean_pool
from pathify_ml.paths import ARTIFACTS, DIMS, REPORTS, SERVING

SAMPLES = [
    "I love fixing my cycle and taking apart old radios",
    "mujhe drawing aur painting karna bahut pasand hai",
    "मुझे बच्चों को पढ़ाना अच्छा लगता है",
    "doctor banne ke liye kya karna padega",
    "Direct or coordinate an organization's financial activities.",
]


class Scores(torch.nn.Module):
    def __init__(self, reader: InterestReader):
        super().__init__()
        self.reader = reader

    def forward(self, input_ids, attention_mask):
        return torch.sigmoid(self.reader(input_ids, attention_mask))


class Embedder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(mean_pool(hidden, attention_mask), dim=-1)


def shrink(encoder, work_dir):
    """Prune the encoder's vocabulary in place; returns the pruned fast tokenizer."""
    original = hf_hub_download(BASE_MODEL, "tokenizer.json")
    from pathlib import Path

    pruned = prune_vocab(Path(original), encoder.embeddings.word_embeddings, work_dir)
    encoder.embeddings.word_embeddings = pruned
    encoder.config.vocab_size = pruned.num_embeddings
    return load_fast_tokenizer(work_dir)


def check_same(a: np.ndarray, b: np.ndarray, label: str, tol: float) -> float:
    gap = float(np.abs(a - b).max())
    print(f"{label}: max abs difference {gap:.5f}")
    if gap > tol:
        sys.exit(f"{label}: difference {gap} is above {tol}")
    return gap


def export_interest(report: dict) -> None:
    from transformers import AutoTokenizer

    work = ARTIFACTS / "interest_reader" / "export"
    model = InterestReader()
    model.load_state_dict(torch.load(ARTIFACTS / "interest_reader" / "model.pt"))
    model.eval()
    full_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    with torch.no_grad():
        enc = full_tok(SAMPLES, padding=True, return_tensors="pt")
        before = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).numpy()
        tok = shrink(model.encoder, work)
        enc = tok(SAMPLES, padding=True, return_tensors="pt")
        after = torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).numpy()
    entry: dict = {"prune_gap": check_same(before, after, "interest pruning", 1e-4)}

    fp32, int8 = work / "model.fp32.onnx", work / "model.onnx"
    export_onnx(Scores(model), tok, fp32, "scores")
    quantize(fp32, int8)
    entry["onnx_fp32_gap"] = check_same(after, onnx_run(fp32, tok, SAMPLES), "interest onnx fp32", 1e-4)
    entry["onnx_int8_gap"] = check_same(after, onnx_run(int8, tok, SAMPLES), "interest onnx int8", 0.08)
    entry["size_mb"] = {"fp32": size_mb(fp32), "int8": size_mb(int8)}

    # Re-score the shipped int8 model on both test sets.
    rows = student_eval()
    pred = onnx_run(int8, tok, [r["text"] for r in rows])
    gold = np.array([DIMS.index(r["dim"]) for r in rows])
    entry["students_top1_int8"] = round(float((pred.argmax(1) == gold).mean()), 4)
    test = build()["test"]
    tp = np.concatenate([onnx_run(int8, tok, [e.text for e in test[i : i + 256]]) for i in range(0, len(test), 256)])
    ty = np.array([e.target for e in test])
    entry["onet_test_pearson_int8"] = round(float(np.mean([np.corrcoef(tp[:, i], ty[:, i])[0, 1] for i in range(6)])), 4)
    report["interest_reader"] = entry

    out = SERVING / "interest_reader"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(int8.read_bytes())
    copy_tokenizer(work, out)


def export_retriever(report: dict) -> None:
    from sentence_transformers import SentenceTransformer

    from scripts.train_retriever import load_eval, rank_metrics
    from pathify_ml.retrieval_data import catalog, doc_text

    work = ARTIFACTS / "retriever" / "export"
    st = SentenceTransformer(str(ARTIFACTS / "retriever" / "model"))
    encoder = st[0].auto_model
    encoder.eval()
    before = st.encode(SAMPLES, normalize_embeddings=True)
    tok = shrink(encoder, work)
    module = Embedder(encoder)
    with torch.no_grad():
        enc = tok(SAMPLES, padding=True, return_tensors="pt")
        after = module(enc["input_ids"], enc["attention_mask"]).numpy()
    entry: dict = {"prune_gap": check_same(before, after, "retriever pruning", 1e-4)}

    fp32, int8 = work / "model.fp32.onnx", work / "model.onnx"
    export_onnx(module, tok, fp32, "embedding")
    quantize(fp32, int8)
    entry["onnx_fp32_gap"] = check_same(after, onnx_run(fp32, tok, SAMPLES), "retriever onnx fp32", 1e-4)
    entry["onnx_int8_gap"] = check_same(after, onnx_run(int8, tok, SAMPLES), "retriever onnx int8", 0.1)
    entry["size_mb"] = {"fp32": size_mb(fp32), "int8": size_mb(int8)}

    careers = catalog().careers
    slugs = [c.slug for c in careers]
    docs = onnx_run(int8, tok, [doc_text(c) for c in careers], max_len=256)
    rows = load_eval()
    q = onnx_run(int8, tok, [r["query"] for r in rows])
    order = np.argsort(-(q @ docs.T), axis=1)
    entry["eval_int8"] = rank_metrics([[slugs[j] for j in row] for row in order], rows)
    report["retriever"] = entry

    out = SERVING / "retriever"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.onnx").write_bytes(int8.read_bytes())
    copy_tokenizer(work, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["interest", "retriever"])
    args = ap.parse_args()
    path = REPORTS / "export.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if args.only in (None, "interest"):
        export_interest(report)
    if args.only in (None, "retriever"):
        export_retriever(report)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
