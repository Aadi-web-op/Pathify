# Pathify ML

Training code for Pathify's three models. Everything here is reproducible from
public data with one command per model, and every model is compared with simple
baselines on held-out data, so the numbers in `reports/` mean something.

| Model | What it does | Base | Trained on | Serves as |
|---|---|---|---|---|
| **Interest Reader** | Free text (English, Hinglish, Hindi) → six Holland interest scores | `paraphrase-multilingual-MiniLM-L12-v2`, fine-tuned | O*NET 31.0 task statements, job titles and activities labelled with each occupation's interest profile | ONNX int8 |
| **Pathfinder** | Six interest scores → probability of each of 12 fields of study | small MLP, temperature-calibrated | 58,554 people from the Open Psychometrics RIASEC dataset with a usable field of study | JSON weights, numpy |
| **Ksarth Retriever** | Student question → the most relevant career guides | `paraphrase-multilingual-MiniLM-L12-v2`, contrastively fine-tuned | Questions generated from each career's own facts in three languages | ONNX int8, fused with BM25 |

Results, model limits and data notes are in [MODEL_CARD.md](MODEL_CARD.md).

## Run it

```bash
cd apps/ml
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.fetch_data             # O*NET 31.0 + Open Psychometrics (≈20 MB)
.venv/bin/python -m scripts.train_pathfinder        # ≈1 min on CPU
.venv/bin/python -m scripts.train_interest_reader   # ≈30 min on 4 CPU cores
.venv/bin/python -m scripts.train_retriever         # ≈5 min on CPU
.venv/bin/python -m scripts.export_models           # prune, ONNX, int8, re-evaluate
```

The Interest Reader, retriever and Pathfinder train on a 4-core CPU. Only the optional LLM
distillation below needs a GPU, and it runs on Kaggle's free tier.

Training uses PyTorch; serving does not. `export_models` writes to
`apps/api/models/`, where the API loads the files with ONNX Runtime and the
`tokenizers` library. To share the encoders, `publish_models` uploads them with
this card to a Hugging Face model repo, and the API's `scripts/fetch_models.py`
downloads them.

## Phase two: Ksarth's own LLM (teacher → student distillation)

Ksarth answers with Gemini today. To run it on Pathify's own model:

1. **Teacher data.** `scripts/build_ksarth_dataset.py` runs Ksarth's real prompt and retrieval for
   2,000 (student, question) pairs in three languages, has Gemini answer, and keeps an answer only
   if its citations are valid, every number in it appears in the notes, a Hindi question gets a
   Hindi answer, and the length is sensible. It needs `GEMINI_API_KEY` and is resumable.
2. **QLoRA on a free GPU.** `notebooks/ksarth_distillation_kaggle.ipynb` fine-tunes
   Qwen2.5-1.5B-Instruct on Kaggle (T4), compares it with the base model on the same grounding
   checks, merges the adapter, and exports a 4-bit GGUF to the Hugging Face Hub.
3. **Serve it.** Run the GGUF with llama.cpp's server and start the API with `LLM_PROVIDER=local`
   (`app/services/llm/openai_compat.py` speaks the OpenAI API, so vLLM or Ollama also work).

## How the evaluation is kept honest

- **Held-out occupations.** The Interest Reader's O*NET test set holds 10% of
  occupations the model never sees in training (split by occupation, not by sentence).
- **Hand-written student sets.** `evals/interest_eval.jsonl` (72 statements) and
  `evals/retrieval_eval.jsonl` (75 questions) were written by hand in how students
  actually talk, in English, Hinglish and Hindi, and are never used for training.
- **Baselines on the same base model.** Zero-shot similarity and a linear probe on
  frozen embeddings show what fine-tuning itself adds; BM25 (the API's previous
  search) is the retrieval baseline; base rates and the classic Holland congruence
  rule are Pathfinder's.
- **Shipped model re-scored.** After pruning and int8 quantization the exported
  models are evaluated again, so the numbers describe what actually runs.

## Shrinking the encoders

The base model's 250k-piece vocabulary covers about 100 languages and its embedding
table is most of the network. `pathify_ml/export.py` keeps only pieces in Latin and
Devanagari script (plus digits and punctuation). English, Hinglish and Hindi text
tokenizes exactly as before, which the export checks, and the model shrinks a lot
before int8 quantization shrinks it again.

## Layout

```
pathify_ml/   data loading, model definitions, export helpers
scripts/      one entry point per step (fetch, train, export, publish)
evals/        hand-written evaluation sets (committed)
reports/      metrics from the last run (committed)
data/         downloaded datasets (ignored)
artifacts/    checkpoints (ignored)
```
