"""A small ONNX encoder runner: tokenizer.json + model.onnx -> numpy output."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxEncoder:
    def __init__(self, directory: Path, threads: int = 2, max_length: int = 128):
        self.directory = directory
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    def __call__(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._lock:  # the tokenizer's truncation setting is shared state
            if max_length:
                self.tokenizer.enable_truncation(max_length=max_length)
            encoded = self.tokenizer.encode_batch(texts)
            if max_length:
                self.tokenizer.enable_truncation(max_length=128)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        return self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]


def available(directory: Path) -> bool:
    return (directory / "model.onnx").is_file() and (directory / "tokenizer.json").is_file()
