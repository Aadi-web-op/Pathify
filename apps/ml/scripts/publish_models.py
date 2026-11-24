"""Publish the exported models and their cards to a Hugging Face model repo.

    HF_TOKEN=hf_... .venv/bin/python -m scripts.publish_models --repo your-name/pathify-models

Uploads apps/api/models/{interest_reader,retriever}/ and pathfinder.json, plus
MODEL_CARD.md (as the repo README) and the eval reports, so anyone can check
the numbers. The API fetches the files with apps/api/scripts/fetch_models.py.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from pathify_ml.paths import ML_ROOT, REPORTS, SERVING


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face token with write access.")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for name in ("interest_reader", "retriever"):
            shutil.copytree(SERVING / name, staging / name)
        shutil.copy(SERVING / "pathfinder.json", staging / "pathfinder.json")
        shutil.copytree(REPORTS, staging / "reports")
        shutil.copy(ML_ROOT / "MODEL_CARD.md", staging / "README.md")
        api.upload_folder(folder_path=str(staging), repo_id=args.repo, repo_type="model", commit_message="Publish Pathify models")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
