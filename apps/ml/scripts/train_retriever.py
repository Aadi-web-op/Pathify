"""Fine-tune the Ksarth Retriever and compare it with keyword search.

    .venv/bin/python -m scripts.train_retriever [--epochs 3]

Compared on the hand-written eval set (75 questions, English/Hinglish/Hindi):
  bm25          the API's current keyword search (BM25 with aliases)
  base          the multilingual encoder before fine-tuning
  fine_tuned    after contrastive fine-tuning on generated question-guide pairs
  hybrid        fine-tuned encoder + BM25, fused with reciprocal rank fusion
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from pathify_ml.interest_model import BASE_MODEL
from pathify_ml.paths import ARTIFACTS, EVALS, REPORTS
from pathify_ml.retrieval_data import catalog, doc_text, pairs

RRF_K = 20


def load_eval() -> list[dict]:
    return [json.loads(x) for x in (EVALS / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines() if x]


def rank_metrics(rankings: list[list[str]], rows: list[dict]) -> dict:
    r1, r3, rr = [], [], []
    by_lang: dict[str, list[float]] = {}
    for ranked, row in zip(rankings, rows, strict=True):
        gold = set(row["relevant"])
        first = next((i for i, s in enumerate(ranked) if s in gold), None)
        r1.append(first == 0)
        r3.append(first is not None and first < 3)
        rr.append(0.0 if first is None or first >= 10 else 1 / (first + 1))
        by_lang.setdefault(row["lang"], []).append(float(first is not None and first < 3))
    return {
        "hit@1": round(float(np.mean(r1)), 4),
        "hit@3": round(float(np.mean(r3)), 4),
        "mrr@10": round(float(np.mean(rr)), 4),
        "hit@3_by_language": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_lang.items())},
        "n": len(rows),
    }


def dense_rankings(model: SentenceTransformer, queries: list[str], slugs: list[str], docs: list[str]) -> list[list[str]]:
    q = model.encode(queries, normalize_embeddings=True)
    d = model.encode(docs, normalize_embeddings=True)
    order = np.argsort(-(q @ d.T), axis=1)
    return [[slugs[j] for j in row] for row in order]


def bm25_rankings(queries: list[str], slugs: list[str]) -> list[list[str]]:
    from app.services.retrieval import career_search

    search = career_search()
    out = []
    for query in queries:
        hits = [h.career.slug for h in search.search(query, k=len(slugs), min_score=0.0)]
        out.append(hits + [s for s in slugs if s not in hits])
    return out


def fuse(a: list[list[str]], b: list[list[str]]) -> list[list[str]]:
    out = []
    for x, y in zip(a, b, strict=True):
        score: dict[str, float] = {}
        for ranked in (x, y):
            for i, s in enumerate(ranked):
                score[s] = score.get(s, 0.0) + 1 / (RRF_K + i + 1)
        out.append(sorted(score, key=lambda s: -score[s]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    careers = catalog().careers
    slugs = [c.slug for c in careers]
    docs = [doc_text(c) for c in careers]
    by_slug = dict(zip(slugs, docs, strict=True))
    rows = load_eval()
    queries = [r["query"] for r in rows]
    train = pairs()

    report: dict = {"base_model": BASE_MODEL, "train_pairs": len(train), "eval_queries": len(rows), "results": {}}
    bm25 = bm25_rankings(queries, slugs)
    report["results"]["bm25"] = rank_metrics(bm25, rows)

    base = SentenceTransformer(BASE_MODEL)
    base.max_seq_length = 256
    report["results"]["base"] = rank_metrics(dense_rankings(base, queries, slugs, docs), rows)
    print(json.dumps(report["results"]), flush=True)

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256
    data = Dataset.from_dict({"anchor": [p.query for p in train], "positive": [by_slug[p.slug] for p in train]})
    out_dir = ARTIFACTS / "retriever"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        # Two questions about the same career in one batch would count as wrong answers for each other.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=0,
        use_cpu=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args, train_dataset=data, loss=MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(str(out_dir / "model"))

    tuned = dense_rankings(model, queries, slugs, docs)
    report["results"]["fine_tuned"] = rank_metrics(tuned, rows)
    report["results"]["hybrid"] = rank_metrics(fuse(tuned, bm25), rows)
    report["rrf_k"] = RRF_K
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retriever.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["results"], indent=1), flush=True)


if __name__ == "__main__":
    main()
