"""Training pairs for the Ksarth Retriever: (student question, career guide).

Questions are generated from each career's own catalog facts (title, Hindi
title, the names students use, day-to-day work, exams, skills) in English,
Hinglish and Hindi. The evaluation set (evals/retrieval_eval.jsonl) is written
by hand in different phrasing and never used here.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

from .paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from app.catalog import Career, catalog  # noqa: E402
from app.services.retrieval import ALIASES, embedding_text  # noqa: E402

EN = [
    "how to become a {t}",
    "what does a {t} do all day",
    "{t} career after 12th",
    "is {t} a good career in India",
    "what should I study to be a {t}",
    "{t} salary and future scope",
    "which entrance exam for {t}",
    "I want to be a {t}",
]
HINGLISH = [
    "{t} kaise bane",
    "{t} banne ke liye kya karna padega",
    "{t} ka scope kya hai",
    "{t} ke liye kaunsa exam dena hota hai",
    "mujhe {t} banna hai",
]
HINDI = ["{h} कैसे बनें", "{h} बनने के लिए क्या पढ़ें", "{h} का करियर कैसा है"]
WORK = [
    "a job where I {w}",
    "I enjoy work where you {w}",
    "mujhe aisa kaam chahiye jisme {w}",
]
EXAM = ["{e} preparation and careers", "careers you can get through {e}", "{e} dena hai, kaunsa career"]
ALIAS = ["{a} career", "{a} ke baad kya kare", "how to get into {a}"]
SKILL = ["career for someone good at {s}"]


def _lower(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].lower() + text[1:]


def doc_text(career: Career) -> str:
    """The API's embedding text, so training and serving embed exactly the same thing."""
    return embedding_text(career)


@dataclass
class Pair:
    query: str
    slug: str


def pairs(seed: int = 5) -> list[Pair]:
    rng = random.Random(seed)
    out: list[Pair] = []
    for c in catalog().careers:
        t = c.title.lower().replace(" / ", " or ")
        out += [Pair(x.format(t=t), c.slug) for x in EN + HINGLISH]
        out += [Pair(x.format(h=c.title_hi), c.slug) for x in HINDI]
        for w in c.work:
            out.append(Pair(rng.choice(WORK).format(w=_lower(w)), c.slug))
        exams = {s.label.split(",")[0].split(" or ")[0].strip() for r in c.routes for s in r.steps if s.kind == "exam"}
        for e in sorted(exams):
            out.append(Pair(rng.choice(EXAM).format(e=e), c.slug))
        for a in ALIASES.get(c.slug, "").split()[:6]:
            out.append(Pair(rng.choice(ALIAS).format(a=a), c.slug))
        for s in c.skills:
            out.append(Pair(rng.choice(SKILL).format(s=_lower(s)), c.slug))
    rng.shuffle(out)
    return out
