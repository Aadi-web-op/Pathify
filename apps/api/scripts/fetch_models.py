"""Download Pathify's exported models into apps/api/models/.

    python -m scripts.fetch_models [--repo OWNER/pathify-models] [--revision main]

The models are published on the Hugging Face Hub (see apps/ml/scripts/publish_models.py).
Pathfinder's small JSON already ships in the repo; this fetches the two ONNX encoders.
Without them the API still runs, with keyword-only search and no free-text reading.
"""

import argparse
import hashlib
import os
from pathlib import Path

import requests

FILES = [
    "interest_reader/model.onnx",
    "interest_reader/tokenizer.json",
    "retriever/model.onnx",
    "retriever/tokenizer.json",
]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("PATHIFY_MODELS_REPO", ""))
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    if not args.repo:
        raise SystemExit("Set --repo or PATHIFY_MODELS_REPO (for example: your-hf-name/pathify-models).")
    for name in FILES:
        target = MODELS_DIR / name
        if target.exists():
            print(f"have {name}")
            continue
        url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            tmp = target.with_suffix(target.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
        tmp.rename(target)
        print(f"got {name} ({target.stat().st_size / 1e6:.1f} MB, sha256 {digest.hexdigest()[:12]})")


if __name__ == "__main__":
    main()
