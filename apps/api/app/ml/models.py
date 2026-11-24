"""The three models, loaded lazily and shared across requests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.ml.runtime import OnnxEncoder, available

log = logging.getLogger(__name__)

DIMS = ["R", "I", "A", "S", "E", "C"]


# --- Interest Reader -------------------------------------------------------------------------


@dataclass(frozen=True)
class InterestReading:
    scores: dict[str, float]  # 0-100 per type
    top: list[str]  # types, strongest first


class InterestReader:
    """Reads free text (English, Hinglish or Hindi) and estimates the six interest scores."""

    def __init__(self, encoder: OnnxEncoder):
        self.encoder = encoder

    def read(self, texts: list[str]) -> list[InterestReading]:
        out = self.encoder(texts)
        readings = []
        for row in out:
            scores = {d: round(float(v) * 100, 1) for d, v in zip(DIMS, row, strict=True)}
            readings.append(InterestReading(scores, sorted(DIMS, key=lambda d: -scores[d])))
        return readings


@lru_cache
def interest_reader() -> InterestReader | None:
    directory = get_settings().models_dir / "interest_reader"
    if not available(directory):
        log.info("Interest Reader not found in %s; free-text reading is off", directory)
        return None
    return InterestReader(OnnxEncoder(directory, threads=get_settings().ml_threads))


# --- Retriever -------------------------------------------------------------------------------


@lru_cache
def retriever_encoder() -> OnnxEncoder | None:
    directory = get_settings().models_dir / "retriever"
    if not available(directory):
        log.info("Retriever not found in %s; career search uses keywords only", directory)
        return None
    return OnnxEncoder(directory, threads=get_settings().ml_threads)


# --- Pathfinder ------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldOdds:
    key: str
    label: str
    clusters: list[str]
    probability: float
    base_rate: float

    @property
    def lift(self) -> float:
        """How much more common this field is among people like you than among everyone."""
        return self.probability / self.base_rate if self.base_rate else 0.0


class Pathfinder:
    """Six interest scores (0-100) -> probability of each field of study, from 58k real people."""

    def __init__(self, spec: dict):
        self.fields = spec["fields"]
        self.mean = np.array(spec["scaler"]["mean"])
        self.scale = np.array(spec["scaler"]["scale"])
        self.layers = [(np.array(layer["w"]), np.array(layer["b"])) for layer in spec["layers"]]
        self.temperature = float(spec["temperature"])
        self.version = spec["version"]
        self.test_metrics = spec.get("test_metrics", {})

    def predict(self, scores: dict[str, float]) -> list[FieldOdds]:
        x = np.array([[scores[d] for d in DIMS]], dtype=np.float64)
        shape = x - x.mean(1, keepdims=True)
        h = (np.hstack([x, shape, x.std(1, keepdims=True)]) - self.mean) / self.scale
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        z = h[0] / self.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        odds = [
            FieldOdds(f["key"], f["label"], f["clusters"], float(prob), float(f["base_rate"]))
            for f, prob in zip(self.fields, p, strict=True)
        ]
        return sorted(odds, key=lambda o: -o.probability)


@lru_cache
def pathfinder() -> Pathfinder | None:
    path = get_settings().models_dir / "pathfinder.json"
    if not path.is_file():
        log.info("Pathfinder not found at %s; field odds are off", path)
        return None
    return Pathfinder(json.loads(path.read_text(encoding="utf-8")))


def status() -> dict[str, bool]:
    return {
        "interest_reader": interest_reader() is not None,
        "retriever": retriever_encoder() is not None,
        "pathfinder": pathfinder() is not None,
    }
