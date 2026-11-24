"""Fine-tune the Interest Reader and compare it with two baselines on the same base model.

    .venv/bin/python -m scripts.train_interest_reader [--epochs 2] [--per-occupation 10]

Writes the best checkpoint to artifacts/interest_reader/ and a report to
reports/interest_reader.json. Baselines:
  zero-shot     cosine similarity to a one-line description of each type
  linear probe  ridge regression on frozen sentence embeddings
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from pathify_ml.interest_data import Example, build, student_eval
from pathify_ml.interest_model import BASE_MODEL, InterestReader, tokenizer
from pathify_ml.paths import ARTIFACTS, CATALOG, DIMS, REPORTS

MAX_LEN = 64


def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def encode(tok, texts: list[str]):
    return tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


@torch.no_grad()
def embed_all(model: InterestReader, tok, texts: list[str], size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for chunk in batches(texts, size):
        enc = encode(tok, chunk)
        out.append(model.embed(enc["input_ids"], enc["attention_mask"]).numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict(model: InterestReader, tok, texts: list[str], size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for chunk in batches(texts, size):
        enc = encode(tok, chunk)
        out.append(torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).numpy())
    return np.concatenate(out)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    per_dim = [float(np.corrcoef(pred[:, i], target[:, i])[0, 1]) for i in range(6)]
    return {
        "pearson_mean": round(float(np.mean(per_dim)), 4),
        "pearson_by_type": {d: round(r, 4) for d, r in zip(DIMS, per_dim)},
        "top_type_accuracy": round(float(np.mean(pred.argmax(1) == target.argmax(1))), 4),
        "n": int(len(target)),
    }


def student_metrics(scores: np.ndarray, rows: list[dict]) -> dict:
    gold = np.array([DIMS.index(r["dim"]) for r in rows])
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0] == gold
    top2 = (order[:, :2] == gold[:, None]).any(1)
    by_lang = {}
    for lang in sorted({r["lang"] for r in rows}):
        idx = np.array([r["lang"] == lang for r in rows])
        by_lang[lang] = {"top1": round(float(top1[idx].mean()), 4), "n": int(idx.sum())}
    return {"top1": round(float(top1.mean()), 4), "top2": round(float(top2.mean()), 4), "by_language": by_lang, "n": len(rows)}


def zero_shot_scores(model: InterestReader, tok, texts: list[str]) -> np.ndarray:
    bank = json.loads((CATALOG / "assessment.json").read_text(encoding="utf-8"))
    protos = [f"{d['name']}: {d['description']}" for d in bank["dimensions"]]
    a = embed_all(model, tok, texts)
    b = embed_all(model, tok, protos)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--per-occupation", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Keep the token embeddings fixed, which preserves the base model's cross-lingual alignment",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    split = build(args.per_occupation, args.seed)
    students = student_eval()
    tok = tokenizer()
    print({k: len(v) for k, v in split.items()}, "students", len(students), flush=True)

    def xy(rows: list[Example]):
        return [e.text for e in rows], np.array([e.target for e in rows], dtype=np.float32)

    train_x, train_y = xy(split["train"])
    val_x, val_y = xy(split["val"])
    test_x, test_y = xy(split["test"])
    student_x = [r["text"] for r in students]

    report: dict = {"base_model": BASE_MODEL, "data": {k: len(v) for k, v in split.items()}, "args": vars(args)}

    # Baselines on the untouched base model.
    base = InterestReader()
    t0 = time.time()
    report["zero_shot"] = {
        "onet_test": regression_metrics(zero_shot_scores(base, tok, test_x), test_y),
        "students": student_metrics(zero_shot_scores(base, tok, student_x), students),
    }
    probe_idx = np.random.default_rng(args.seed).choice(len(train_x), size=min(8000, len(train_x)), replace=False)
    ridge = Ridge(alpha=1.0).fit(embed_all(base, tok, [train_x[i] for i in probe_idx]), train_y[probe_idx])
    report["linear_probe"] = {
        "onet_test": regression_metrics(ridge.predict(embed_all(base, tok, test_x)), test_y),
        "students": student_metrics(ridge.predict(embed_all(base, tok, student_x)), students),
    }
    print("baselines", json.dumps({k: report[k] for k in ("zero_shot", "linear_probe")}), f"{time.time() - t0:.0f}s", flush=True)
    del base

    # Fine-tune.
    model = InterestReader()
    if args.freeze_embeddings:
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
    params = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train_x) / args.batch)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))

    best, step, history = -1.0, 0, []
    out_dir = ARTIFACTS / "interest_reader"
    for epoch in range(args.epochs):
        order = list(range(len(train_x)))
        random.shuffle(order)
        model.train()
        t0 = time.time()
        for chunk in batches(order, args.batch):
            enc = encode(tok, [train_x[i] for i in chunk])
            y = torch.from_numpy(train_y[chunk])
            logits = model(enc["input_ids"], enc["attention_mask"])
            # Regression on all six scores, plus a push to get the strongest type right.
            loss = F.mse_loss(torch.sigmoid(logits), y) + 0.2 * F.cross_entropy(logits * 2, y.argmax(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        val = regression_metrics(predict(model, tok, val_x), val_y)
        history.append({"epoch": epoch, "val": val})
        print("val", json.dumps(val), flush=True)
        if val["pearson_mean"] > best:
            best = val["pearson_mean"]
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "model.pt")
            tok.save_pretrained(out_dir)

    model.load_state_dict(torch.load(out_dir / "model.pt"))
    report["fine_tuned"] = {
        "onet_test": regression_metrics(predict(model, tok, test_x), test_y),
        "students": student_metrics(predict(model, tok, student_x), students),
        "history": history,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "interest_reader.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("final", json.dumps(report["fine_tuned"]["onet_test"]), json.dumps(report["fine_tuned"]["students"]), flush=True)


if __name__ == "__main__":
    main()
