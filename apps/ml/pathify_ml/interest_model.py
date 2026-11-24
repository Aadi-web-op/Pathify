"""The Interest Reader network: a multilingual sentence encoder with a 6-way regression head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class InterestReader(nn.Module):
    """Text -> six interest logits (R I A S E C). Apply a sigmoid for 0-1 scores."""

    def __init__(self, base: str = BASE_MODEL, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        size = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(size, size), nn.GELU(), nn.Linear(size, 6))

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids, attention_mask))


def tokenizer(base: str = BASE_MODEL):
    return AutoTokenizer.from_pretrained(base)
