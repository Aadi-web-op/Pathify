"""Train Pathfinder and compare it with the rule-based approach.

    .venv/bin/python -m scripts.train_pathfinder

Models compared on the same held-out people:
  congruence   rank fields by how closely a person's profile correlates with the
               field's average profile (the classic Holland "congruence" rule)
  logistic     multinomial logistic regression
  mlp          a small neural network (two hidden layers)
  boosting     histogram gradient boosting

The chosen model is calibrated with temperature scaling on the validation set
and exported as plain JSON weights (numpy inference, no ML runtime needed).
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathify_ml.pathfinder_data import FIELD_KEYS, FIELD_TO_CLUSTERS, FIELDS, load
from pathify_ml.paths import ARTIFACTS, REPORTS, SERVING

SEED = 7


def features(x: np.ndarray) -> np.ndarray:
    """Raw scores plus the profile's shape (each score minus the person's own average)."""
    shape = x - x.mean(1, keepdims=True)
    return np.hstack([x, shape, x.std(1, keepdims=True)])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics(prob: np.ndarray, y: np.ndarray) -> dict:
    order = np.argsort(-prob, axis=1)
    top1 = order[:, 0] == y
    top3 = (order[:, :3] == y[:, None]).any(1)
    # Expected calibration error on top-1 confidence, 10 bins.
    conf = prob.max(1)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.any():
            ece += m.mean() * abs(top1[m].mean() - conf[m].mean())
    per_class_top3 = [float(top3[y == k].mean()) for k in range(len(FIELD_KEYS)) if (y == k).any()]
    return {
        "top1": round(float(top1.mean()), 4),
        "top3": round(float(top3.mean()), 4),
        "macro_top3": round(float(np.mean(per_class_top3)), 4),
        "log_loss": round(float(log_loss(y, np.clip(prob, 1e-9, 1), labels=list(range(len(FIELD_KEYS))))), 4),
        "ece": round(float(ece), 4),
        "n": int(len(y)),
    }


def congruence(train_x: np.ndarray, train_y: np.ndarray):
    profiles = np.stack([train_x[train_y == k].mean(0) for k in range(len(FIELD_KEYS))])
    pc = profiles - profiles.mean(1, keepdims=True)

    def predict(x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(1, keepdims=True)
        r = (xc @ pc.T) / (np.linalg.norm(xc, axis=1, keepdims=True) * np.linalg.norm(pc, axis=1) + 1e-9)
        return softmax(r * 4)  # a fixed sharpening so log loss is defined

    return predict


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 51):
        ll = log_loss(y, softmax(logits / t), labels=list(range(len(FIELD_KEYS))))
        if ll < best:
            best_t, best = float(t), ll
    return best_t


def main() -> None:
    data = load()
    idx = np.arange(len(data.y))
    tr, rest = train_test_split(idx, test_size=0.2, stratify=data.y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.5, stratify=data.y[rest], random_state=SEED)
    x, y = data.x, data.y
    india_te = te[data.country[te] == "IN"]
    report: dict = {
        "data": {**data.stats, "train": len(tr), "val": len(va), "test": len(te), "india_test": len(india_te)},
        "fields": FIELDS,
        "base_rates": {FIELD_KEYS[k]: round(float((y[tr] == k).mean()), 4) for k in range(len(FIELD_KEYS))},
        "models": {},
    }

    def evaluate(name: str, predict) -> None:
        report["models"][name] = {
            "val": metrics(predict(x[va]), y[va]),
            "test": metrics(predict(x[te]), y[te]),
            "india_test": metrics(predict(x[india_te]), y[india_te]),
        }
        print(name, json.dumps(report["models"][name]["test"]), flush=True)

    prior = np.bincount(y[tr], minlength=len(FIELD_KEYS)) / len(tr)
    evaluate("base_rate", lambda z: np.tile(prior, (len(z), 1)))
    evaluate("congruence", congruence(x[tr], y[tr]))

    scaler = StandardScaler().fit(features(x[tr]))
    f = lambda z: scaler.transform(features(z))  # noqa: E731

    logistic = LogisticRegression(max_iter=2000, C=1.0).fit(f(x[tr]), y[tr])
    evaluate("logistic", lambda z: logistic.predict_proba(f(z)))

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=SEED
    ).fit(f(x[tr]), y[tr])
    evaluate("mlp", lambda z: mlp.predict_proba(f(z)))

    boosting = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, early_stopping=True, random_state=SEED
    ).fit(f(x[tr]), y[tr])
    evaluate("boosting", lambda z: boosting.predict_proba(f(z)))

    # Export the MLP: small, smooth in its inputs, and easy to run with numpy alone.
    def mlp_logits(z: np.ndarray, t: float = 1.0) -> np.ndarray:
        h = f(z)
        for w, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1], strict=True):
            h = np.maximum(h @ w + b, 0)
        return (h @ mlp.coefs_[-1] + mlp.intercepts_[-1]) / t

    assert np.allclose(softmax(mlp_logits(x[va[:50]])), mlp.predict_proba(f(x[va[:50]])), atol=1e-5)
    temperature = fit_temperature(mlp_logits(x[va]), y[va])
    evaluate("mlp_calibrated", lambda z: softmax(mlp_logits(z, temperature)))
    report["chosen"] = {"model": "mlp_calibrated", "temperature": temperature}

    export = {
        "version": "pathfinder-1",
        "about": (
            "Pathfinder: six interest scores (0-100) -> probability of each field of study. "
            "Trained on Open Psychometrics RIASEC data."
        ),
        "fields": [
            {"key": k, "label": FIELDS[k], "clusters": FIELD_TO_CLUSTERS[k], "base_rate": report["base_rates"][k]}
            for k in FIELD_KEYS
        ],
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "layers": [{"w": w.tolist(), "b": b.tolist()} for w, b in zip(mlp.coefs_, mlp.intercepts_, strict=True)],
        "temperature": temperature,
        "test_metrics": report["models"]["mlp_calibrated"]["test"],
    }
    for target in (ARTIFACTS / "pathfinder", SERVING):
        target.mkdir(parents=True, exist_ok=True)
        (target / "pathfinder.json").write_text(json.dumps(export))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "pathfinder.json").write_text(json.dumps(report, indent=2))
    print("chosen temperature", temperature)


if __name__ == "__main__":
    main()
