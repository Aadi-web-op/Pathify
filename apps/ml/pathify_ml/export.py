"""Shrink and export encoders for CPU serving.

1. Vocabulary pruning. The base model's 250k-piece vocabulary covers ~100
   languages, and its embedding table is most of the model. Pathify needs
   English, Hinglish and Hindi, so we keep only pieces written in Latin or
   Devanagari (plus digits and punctuation). Text in those scripts tokenizes
   exactly as before, because a removed piece contains a character such text
   cannot have.
2. ONNX export of the full network (encoder + pooling + head).
3. Dynamic int8 quantization with onnxruntime.

Each step is checked against the previous one on sample sentences.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedTokenizerFast

KEEP = re.compile(r"^[▁ -~ -ÿऀ-ॿ‌‍‐-‧₹]+$")


def prune_vocab(tokenizer_json: Path, embeddings: torch.nn.Embedding, out_dir: Path) -> torch.nn.Embedding:
    """Write a pruned tokenizer.json to out_dir and return the matching, smaller embedding table."""
    spec = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = spec["model"]
    assert model["type"] == "Unigram", model["type"]
    vocab = model["vocab"]
    specials = {t["id"] for t in spec.get("added_tokens", [])}
    keep_ids = [i for i, (piece, _) in enumerate(vocab) if i in specials or i < 4 or KEEP.match(piece)]
    # Added tokens (e.g. <mask>) may sit past the Unigram vocab; keep them at the end.
    extra = sorted(i for i in specials if i >= len(vocab))
    old_to_new = {old: new for new, old in enumerate(keep_ids + extra)}

    model["vocab"] = [vocab[i] for i in keep_ids]
    model["unk_id"] = old_to_new[model["unk_id"]]
    for tok in spec.get("added_tokens", []):
        tok["id"] = old_to_new[tok["id"]]

    def remap(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "ids" and isinstance(value, list) and all(isinstance(v, int) for v in value):
                    node[key] = [old_to_new[v] for v in value]
                else:
                    remap(value)
            # Template post-processors store special tokens as {"id": "<s>", "ids": [0], "tokens": [...]}.
        elif isinstance(node, list):
            for v in node:
                remap(v)

    remap(spec.get("post_processor"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    index = torch.tensor(keep_ids + extra)
    pruned = torch.nn.Embedding(
        len(index), embeddings.embedding_dim, padding_idx=old_to_new.get(embeddings.padding_idx)
    )
    pruned.weight.data = embeddings.weight.data[index].clone()
    print(f"vocabulary {len(vocab) + len(extra)} -> {len(index)} pieces")
    return pruned


def load_fast_tokenizer(directory: Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def export_onnx(module: torch.nn.Module, tok, out_path: Path, output_name: str) -> None:
    module.eval()
    sample = tok(["a sample sentence", "दूसरा वाक्य"], padding=True, return_tensors="pt")
    torch.onnx.export(
        module,
        (sample["input_ids"], sample["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=[output_name],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            output_name: {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def quantize(src: Path, dst: Path, op_types: list[str] | None = None) -> None:
    """Dynamic int8. `op_types=["Gather"]` quantizes only the embedding table, which is most of
    the size, and keeps the transformer's matrix multiplications in fp32."""
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=op_types)


def onnx_run(path: Path, tok, texts: list[str], max_len: int = 128) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="np")
    return session.run(
        None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
    )[0]


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1)


def copy_tokenizer(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "tokenizer.json", dst_dir / "tokenizer.json")
