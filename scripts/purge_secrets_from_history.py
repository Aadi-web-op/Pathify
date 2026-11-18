#!/usr/bin/env python3
"""Remove leaked credentials from every commit in this repository.

What it does
  1. Reads every version of every *.env file that was ever committed and collects
     the values that look like secrets (16+ characters).
  2. Also collects Gemini (AIza...) and Groq (gsk_...) keys hardcoded in source files.
  3. Rewrites history with git-filter-repo: deletes all *.env files from every
     commit and replaces each collected value with ***REMOVED*** everywhere else.
  4. Verifies that none of the values remain.

The secret values are written only to a temporary file outside the repository,
which is deleted afterwards. Nothing is pushed. After it finishes, push with:

    git push --force-with-lease origin --all

Rewriting history does NOT make leaked keys safe. Rotate them first.

Requires: pip install git-filter-repo
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

KEY_PATTERNS = [rb"AIza[0-9A-Za-z_\-]{30,}", rb"gsk_[0-9A-Za-z]{20,}"]
ENV_LINE = re.compile(r"""\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["']?([^"'\n#]+)""")
SOURCE_SUFFIXES = (".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".html")


def git(*args: str, binary: bool = False) -> str | bytes:
    out = subprocess.run(["git", *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def history_blobs() -> list[tuple[str, str]]:
    blobs = []
    for line in str(git("rev-list", "--all", "--objects")).splitlines():
        sha, _, path = line.partition(" ")
        if path and str(git("cat-file", "-t", sha)).strip() == "blob":
            blobs.append((sha, path))
    return blobs


def collect_secrets(blobs: list[tuple[str, str]]) -> set[str]:
    found: set[str] = set()
    for sha, path in blobs:
        if path.endswith(".env"):
            text = bytes(git("cat-file", "-p", sha, binary=True)).decode(errors="ignore")
            for line in text.splitlines():
                m = ENV_LINE.match(line)
                if not m:
                    continue
                value = m.group(2).strip()
                if len(value) >= 16 and not value.lower().startswith(("your", "change-me")):
                    found.add(value)
        elif path.endswith(SOURCE_SUFFIXES):
            data = bytes(git("cat-file", "-p", sha, binary=True))
            for pattern in KEY_PATTERNS:
                found.update(m.decode() for m in re.findall(pattern, data))
    return found


def remaining(blobs: list[tuple[str, str]], secrets: set[str]) -> list[str]:
    problems = []
    for sha, path in blobs:
        if path.endswith(".env"):
            problems.append(f"{path}: .env file still present")
            continue
        data = bytes(git("cat-file", "-p", sha, binary=True))
        if any(s.encode() in data for s in secrets):
            problems.append(f"{path}: secret value still present")
    return problems


def main() -> int:
    if not shutil.which("git-filter-repo"):
        print("git-filter-repo is not installed. Run: pip install git-filter-repo")
        return 1
    if str(git("status", "--porcelain")).strip():
        print("Commit or stash your changes first; the rewrite resets the working tree.")
        return 1

    secrets = collect_secrets(history_blobs())
    print(f"Found {len(secrets)} secret values and these .env paths in history:")
    for path in sorted({p for _, p in history_blobs() if p.endswith('.env')}):
        print(f"  {path}")
    if "--yes" not in sys.argv:
        answer = input("Rewrite every commit now? Type 'rewrite' to continue: ")
        if answer.strip() != "rewrite":
            print("Cancelled.")
            return 1

    remote = str(git("remote", "get-url", "origin")).strip()
    fd, replacements = tempfile.mkstemp(prefix="pathify-redact-", suffix=".txt")
    try:
        with os.fdopen(fd, "w") as fh:
            for value in sorted(secrets):
                fh.write(f"{value}==>***REMOVED***\n")
        subprocess.run(
            ["git", "filter-repo", "--force", "--invert-paths",
             "--path-glob", "*.env", "--path", ".env",
             "--replace-text", replacements],
            check=True,
        )
    finally:
        os.remove(replacements)

    # filter-repo removes the origin remote as a safety measure; put it back.
    subprocess.run(["git", "remote", "add", "origin", remote], check=False)
    # Re-fetch so --force-with-lease has remote-tracking refs to compare against.
    # This only updates refs/remotes/origin/*; local branches keep the clean history.
    subprocess.run(["git", "fetch", "origin"], check=False)

    problems = remaining(history_blobs(), secrets)
    if problems:
        print("Rewrite finished but these remain:", *problems, sep="\n  ")
        return 1
    print("\nHistory is clean. Next steps:")
    print("  git push --force-with-lease origin --all")
    print("  Everyone else must re-clone; an old clone would push the secrets back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
