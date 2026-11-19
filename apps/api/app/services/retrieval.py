"""Find the career guides most relevant to a student's question, and cite them.

Ranking is BM25 over each career's title, summary, day-to-day work, skills,
subjects and routes (exams, degrees, steps). The title and the common names
Indian students use (MBBS, CA, UPSC, coding...) count extra. Small corpus,
no external services, fully deterministic, so it is easy to test.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from app.catalog import Career, Catalog, catalog

K1 = 1.5
B = 0.75
TITLE_WEIGHT = 3
MIN_SCORE = 2.0

_STOPWORD_TEXT = """
a an and are as at be become becoming best can could do does for from get go have how i if in into is it
its me my of on or should so than that the then there this to want what when where which who why will
with would you your after about also any job jobs career careers work india indian like
take taking make need needs know tell much many more really think help please
"""
# Class numbers say where a student is, not what the question is about.
_CLASS_TEXT = "class std standard 9 9th 10 10th 11 11th 12 12th th"
# Words that appear in nearly every career's routes, so they can't tell careers apart.
_GENERIC_TEXT = """
entrance exam exams test tests stream streams fit course courses college colleges degree degrees
scope option options good best right choose choosing
"""
STOPWORDS = frozenset((_STOPWORD_TEXT + _CLASS_TEXT + _GENERIC_TEXT).split())

# Words students use that the catalog text may not contain.
ALIASES: dict[str, str] = {
    "software-engineer": "coding coder programmer programming developer it computer computers cse tech",
    "data-scientist": "ai ml machine learning artificial intelligence statistics",
    "data-analyst": "excel sql dashboards analytics",
    "ux-designer": "ui ux app design product design figma",
    "graphic-designer": "drawing art sketching painting logo posters branding illustration",
    "animator": "drawing art animation vfx cartoons gaming visual effects",
    "fashion-designer": "drawing art fashion clothes clothing garments",
    "architect": "architecture buildings nata barch",
    "civil-engineer": "construction buildings bridges roads",
    "mechanical-engineer": "machines cars automobile robotics",
    "electronics-engineer": "electronics electrical circuits ece eee chips",
    "electrician": "iti wiring electrical trade",
    "doctor": "mbbs medicine medical neet physician aiims",
    "dentist": "bds dental teeth",
    "nurse": "nursing hospital",
    "physiotherapist": "physio bpt sports injury",
    "pharmacist": "pharmacy bpharm medicines",
    "veterinarian": "vet bvsc animals pets",
    "clinical-psychologist": "psychology psychologist therapy counselling mental health",
    "research-scientist": "research scientist phd iiser isro physics chemistry",
    "biotechnologist": "biotech biotechnology genetics",
    "agricultural-scientist": "agriculture farming icar crops",
    "economist": "economics economy policy",
    "chartered-accountant": "ca icai accounting accountancy audit tax",
    "financial-analyst": "finance investment banking stock market cfa",
    "bank-officer": "bank banking po ibps sbi",
    "marketing-manager": "marketing advertising brand mba",
    "product-manager": "product management pm startup",
    "entrepreneur": "business startup founder own business",
    "hotel-manager": "hotel hospitality ihm tourism",
    "chef": "cooking cook kitchen culinary food",
    "lawyer": "law llb clat advocate court legal",
    "civil-servant": "upsc ias ips ifs civil services government officer collector",
    "armed-forces-officer": "army navy air force nda defence military",
    "commercial-pilot": "pilot aviation flying airline cpl",
    "teacher": "teaching school bed ctet tutor",
    "journalist": "journalism media news reporter mass communication",
    "social-worker": "ngo social work msw community",
}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def career_text(career: Career) -> str:
    parts = [career.summary, " ".join(career.work), " ".join(career.skills), " ".join(career.subjects)]
    for route in career.routes:
        parts.append(route.label)
        parts.extend(f"{s.label} {s.detail or ''}" for s in route.steps)
        if route.requirement:
            parts.append(route.requirement)
    if career.graduate_route:
        parts.extend(s.label for s in career.graduate_route.steps)
    if career.note:
        parts.append(career.note)
    return " ".join(parts)


@dataclass(frozen=True)
class Hit:
    career: Career
    score: float


class CareerSearch:
    def __init__(self, catalog: Catalog):
        self.careers = catalog.careers
        self.docs: list[Counter[str]] = []
        for c in self.careers:
            weighted = (tokens(c.title) + tokens(ALIASES.get(c.slug, ""))) * TITLE_WEIGHT + tokens(career_text(c))
            self.docs.append(Counter(weighted))
        self.avg_len = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = MIN_SCORE) -> list[Hit]:
        terms = set(tokens(query))
        hits: list[Hit] = []
        for career, doc in zip(self.careers, self.docs, strict=True):
            length = sum(doc.values())
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_len))
            if score >= min_score:
                hits.append(Hit(career, score))
        hits.sort(key=lambda h: (-h.score, h.career.title))
        return hits[:k]


@lru_cache
def career_search() -> CareerSearch:
    return CareerSearch(catalog())


def career_note(career: Career) -> str:
    """A compact, factual summary of a career for the model to cite."""
    lines = [f"{career.title}: {career.summary}"]
    for route in career.routes:
        streams = "any stream" if "any" in route.streams else ", ".join(route.streams)
        steps = " -> ".join(
            s.label + (f" ({s.detail})" if s.detail else "") + (f" [about {s.years:g} years]" if s.years else "")
            for s in route.steps
        )
        after = {"class_10": "after Class 10", "class_12": "after Class 12", "graduation": "after graduation"}
        lines.append(f"Route '{route.label}' ({after[route.after]}, {streams}): {steps}.")
        if route.requirement:
            lines.append(f"Requirement: {route.requirement}")
    if career.graduate_route:
        steps = " -> ".join(s.label for s in career.graduate_route.steps)
        lines.append(f"{career.graduate_route.label}: {steps}.")
    if career.note:
        lines.append(f"Note: {career.note}")
    if career.check_with:
        lines.append("Official sources: " + "; ".join(b.name for b in career.check_with) + ".")
    return " ".join(lines)


CITATION = re.compile(r"\[(\d{1,2})\]")


def clean_citations(text: str, count: int) -> tuple[str, list[int]]:
    """Drop citation markers that don't point at a note, and list the ones used, in order of first use."""
    used: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= count:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    cleaned = CITATION.sub(keep, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), used
