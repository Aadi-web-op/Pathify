"""Pathify's own models, served on CPU.

Training code lives in apps/ml. Here we only load the exported files:
  models/interest_reader/{model.onnx,tokenizer.json}  text -> six interest scores
  models/retriever/{model.onnx,tokenizer.json}        text -> embedding for career search
  models/pathfinder.json                              interest scores -> field of study

Each model is optional. If its files are missing the feature that needs it
reports itself unavailable, and everything else keeps working.
"""
