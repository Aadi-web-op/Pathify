"""Build the teacher dataset for distilling Ksarth into a small open model.

    # Uses the API's code (prompt, retrieval), so run it with the API's environment:
    GEMINI_API_KEY=... ../api/.venv/bin/python -m scripts.build_ksarth_dataset --target 2000

For each (student profile, question) pair this runs Ksarth's real pipeline — the
production system prompt, the fine-tuned retriever's notes — and asks Gemini (the
teacher) to answer. An answer becomes training data only if it passes four checks:

  citations   every [n] points at a note it was given, and at least one is used
  numbers     every number in the answer appears in the notes or the question
              (Pathify's "no made-up numbers" rule, as a filter)
  language    a question asked in Devanagari is answered in Devanagari
  length      30-400 words

Output: data/ksarth_sft/{train,val}.jsonl in chat format, plus reports/ksarth_dataset.json.
Resumable: answered pairs are kept in data/ksarth_sft/raw.jsonl and skipped on rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from types import SimpleNamespace

from pathify_ml.paths import ML_ROOT, REPO_ROOT, REPORTS

sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.config import get_settings  # noqa: E402
from app.models.user import Board, Goal, Stage, Stream, StudentProfile  # noqa: E402
from app.services.ksarth import system_prompt  # noqa: E402
from app.services.llm import ChatMessage, GeminiProvider, LLMError  # noqa: E402
from app.services.retrieval import clean_citations, search_careers  # noqa: E402

from pathify_ml.retrieval_data import pairs  # noqa: E402

OUT = ML_ROOT / "data" / "ksarth_sft"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
ALWAYS_OK_NUMBERS = {"14416", "9", "10", "11", "12", "1", "2", "3"}  # helpline, class numbers, list markers

PROFILES = [
    (
        "Asha",
        StudentProfile(
            stage=Stage.SCHOOL_9_10, grade=10, board=Board.CBSE, goals=[Goal.CHOOSE_STREAM], guardian_aware=True
        ),
    ),
    (
        "Rohan",
        StudentProfile(
            stage=Stage.SCHOOL_11_12,
            grade=11,
            board=Board.STATE,
            stream=Stream.SCIENCE_PCM,
            goals=[Goal.CHOOSE_EXAMS_DEGREE],
            guardian_aware=True,
        ),
    ),
    (
        "Fatima",
        StudentProfile(
            stage=Stage.SCHOOL_11_12,
            grade=12,
            board=Board.CBSE,
            stream=Stream.SCIENCE_PCB,
            goals=[Goal.CHOOSE_EXAMS_DEGREE],
            language="hi",
            guardian_aware=True,
        ),
    ),
    (
        "Kabir",
        StudentProfile(
            stage=Stage.SCHOOL_11_12,
            grade=12,
            board=Board.ICSE,
            stream=Stream.COMMERCE,
            goals=[Goal.CHOOSE_EXAMS_DEGREE],
            guardian_aware=True,
        ),
    ),
    (
        "Meera",
        StudentProfile(
            stage=Stage.SCHOOL_11_12,
            grade=11,
            board=Board.CBSE,
            stream=Stream.HUMANITIES,
            goals=[Goal.EXPLORE],
            guardian_aware=True,
        ),
    ),
    (
        "Arjun",
        StudentProfile(
            stage=Stage.COLLEGE, degree="B.Sc Physics", year_of_study=2, goals=[Goal.FIRST_JOB, Goal.SWITCH_FIELD]
        ),
    ),
]

GENERAL = [
    "My parents want me to become an engineer but I like drawing. What should I do?",
    "Is taking a drop year for NEET a good idea?",
    "Which stream keeps the most options open?",
    "I am confused between Science and Commerce. How do I decide?",
    "What can I do if I don't clear JEE?",
    "How do I explain my career choice to my parents?",
    "Mere marks kam aaye hain, ab kya karun?",
    "Ghar wale doctor banana chahte hain par mujhe interest nahi hai",
    "मुझे समझ नहीं आ रहा कि कौन सी स्ट्रीम चुनूं",
    "बोर्ड परीक्षा के बाद कौन से कोर्स अच्छे हैं?",
    "I feel like I'm behind everyone else. Is it too late to change my plan?",
    "Can I do engineering after Commerce?",
]


def numbers_ok(answer: str, context: str) -> bool:
    allowed = set(NUMBER.findall(context)) | ALWAYS_OK_NUMBERS
    return all(n in allowed for n in NUMBER.findall(answer))


def check(answer: str, notes_count: int, question: str, prompt: str) -> str | None:
    """None if the answer is good training data, else the reason it isn't."""
    words = len(answer.split())
    if not 30 <= words <= 400:
        return "length"
    cleaned, used = clean_citations(answer, notes_count)
    if len(re.findall(r"\[\d{1,2}\]", answer)) != len(re.findall(r"\[\d{1,2}\]", cleaned)):
        return "bad_citation"
    if notes_count and not used:
        return "no_citation"
    if not numbers_ok(answer, prompt + " " + question):
        return "unsupported_number"
    if DEVANAGARI.search(question) and not DEVANAGARI.search(answer):
        return "wrong_language"
    return None


def jobs(target: int, seed: int = 3) -> list[dict]:
    rng = random.Random(seed)
    questions = list(dict.fromkeys([p.query for p in pairs()] + GENERAL))
    rng.shuffle(questions)
    out = []
    for i in range(target):
        # Past the first pass, the same question comes back from a different student.
        q = questions[i % len(questions)]
        name, profile = PROFILES[(i + i // len(questions)) % len(PROFILES)]
        key = hashlib.sha1(f"{name}|{q}".encode()).hexdigest()[:16]
        out.append({"id": key, "name": name, "profile": profile, "question": q})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--per-minute", type=int, default=12, help="stay under the API's rate limit")
    args = ap.parse_args()
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("Set GEMINI_API_KEY (any non-empty value works if the key is injected by a proxy).")
    teacher = GeminiProvider(settings.gemini_api_key, settings.gemini_model, timeout=60)

    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "raw.jsonl"
    done = {json.loads(line)["id"] for line in raw_path.open()} if raw_path.exists() else set()
    todo = [j for j in jobs(args.target) if j["id"] not in done]
    print(f"{len(done)} already answered, {len(todo)} to go", flush=True)

    gate = asyncio.Semaphore(args.concurrency)
    interval = 60 / args.per_minute

    async def run(job: dict) -> None:
        async with gate:
            user = SimpleNamespace(name=job["name"], profile=job["profile"])
            notes = [h.career for h in search_careers(job["question"], k=3)]
            prompt = system_prompt(user, None, [], notes)
            try:
                reply = await teacher.complete(
                    system=prompt, messages=[ChatMessage("user", job["question"])], temperature=0.4
                )
                answer, error = reply.text.strip(), None
            except LLMError as exc:
                answer, error = "", str(exc)
            row = {
                "id": job["id"],
                "system": prompt,
                "question": job["question"],
                "answer": answer,
                "notes": len(notes),
                "error": error,
                "rejected": error or check(answer, len(notes), job["question"], prompt),
            }
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            await asyncio.sleep(interval * args.concurrency)

    for start in range(0, len(todo), 50):
        await asyncio.gather(*(run(j) for j in todo[start : start + 50]))
        print(f"answered {min(start + 50, len(todo))}/{len(todo)}", flush=True)

    rows = [json.loads(line) for line in raw_path.open()]
    good = [r for r in rows if not r["rejected"]]
    random.Random(7).shuffle(good)
    n_val = max(20, len(good) // 20)
    for name, part in (("val", good[:n_val]), ("train", good[n_val:])):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                messages = [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    reasons: dict[str, int] = {}
    for r in rows:
        if r["rejected"]:
            key = "api_error" if r["error"] else r["rejected"]
            reasons[key] = reasons.get(key, 0) + 1
    report = {
        "teacher": settings.gemini_model,
        "answered": len(rows),
        "kept": len(good),
        "rejected": reasons,
        "val": n_val,
    }
    (REPORTS / "ksarth_dataset.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
