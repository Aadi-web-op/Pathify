"""Training data for Pathfinder: six interest scores -> the field a person studied.

Source: Open Psychometrics RIASEC data (2015-2018). Respondents rated 48
activities from 1 (dislike) to 5 (enjoy) and typed their field of study as free
text. Majors are mapped to twelve broad fields with ordered keyword rules
(first match wins); anything vague ("general", "idk", "none") is dropped
rather than guessed.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np

from .paths import DIMS, RIASEC_CSV

FIELDS: dict[str, str] = {
    "computing": "Computing and IT",
    "engineering": "Engineering",
    "health": "Health and medicine",
    "psychology_social": "Psychology and social sciences",
    "education": "Education and teaching",
    "law_politics": "Law and politics",
    "business": "Business, finance and economics",
    "life_sciences": "Life sciences",
    "physical_sciences": "Maths and physical sciences",
    "arts_media": "Arts, design and media",
    "humanities": "Languages and humanities",
    "hospitality_aviation": "Hospitality, aviation and trades",
}

# Pathify career clusters each field leads into most often (used to link results to careers).
FIELD_TO_CLUSTERS: dict[str, list[str]] = {
    "computing": ["technology"],
    "engineering": ["engineering", "technology"],
    "health": ["health"],
    "psychology_social": ["education-social", "health"],
    "education": ["education-social"],
    "law_politics": ["law-government"],
    "business": ["business"],
    "life_sciences": ["science", "health"],
    "physical_sciences": ["science", "technology"],
    "arts_media": ["design", "media"],
    "humanities": ["media", "education-social", "law-government"],
    "hospitality_aviation": ["hospitality", "defence-aviation", "skilled-trades"],
}

_RULES: list[tuple[str, str]] = [
    ("humanities", r"liberal (arts|studies)|art history|history of art"),
    # Indian degree names first: B.Tech CSE / BCA are computing, B.Com / CA are business.
    ("computing", r"\bcse\b|\bbca\b|\bmca\b|b\.?\s?tech.*(cs|computer|it)\b"),
    ("business", r"\bb\.?\s?com\b|\bm\.?\s?com\b|\bca\b|chartered account|\bcfa\b"),
    ("engineering", r"\bb\.?\s?tech\b|\bb\.?\s?e\.?\b"),
    (
        "computing",
        r"comput|software|information (tech|systems)|\bit\b|\bict\b|cyber|data science|informatics|programming",
    ),
    ("engineering", r"engineer|mechatronic|electronics?\b|aerospace|aeronautic|construction management"),
    (
        "health",
        r"nurs|medic|doctor|\bmbbs\b|pharm|dent|physio|physical therapy|occupational therapy|kinesiology|exercise science|"
        r"nutrition|dietetic|public health|health|radiolog|paramedic|speech (path|therapy)|veterinar|optometr|midwif|"
        r"athletic training|surgery|anatomy|pre-?med",
    ),
    (
        "psychology_social",
        r"ps[yi]c|pyscholog|psycolog|sociolog|social work|counsel|human (services|development)|child development|"
        r"anthropolog|criminolog|social sciences?|behavio|rehabilitation|family studies|gender studies",
    ),
    ("education", r"educat|teach|pedagog|early childhood|montessori"),
    (
        "law_politics",
        r"\blaw\b|legal|paralegal|criminal justice|politic|international (relations|studies)|public (administration|policy)|government|diplomac",
    ),
    (
        "business",
        r"busi?ness?|buisness|bussiness|commerce|account|financ|econom|marketing|management|\bmba\b|\bbba\b|\bhrm?\b|human resource|"
        r"administration|actuar|banking|entrepreneur|supply chain|logistic|real estate|\bsales\b|retail",
    ),
    (
        "life_sciences",
        r"biolog|biochem|microbio|zoolog|biotech|neuroscience|marine|ecolog|environment|agricultur|animal science|botany|"
        r"genetic|physiolog|food science|forestry|biomedical science|life science",
    ),
    ("physical_sciences", r"\bmaths?\b|mathemat|physics|chemistry|statistic|geolog|astronom|geograph|applied science"),
    (
        "arts_media",
        r"\barts?\b|design|music|film|photograph|theat|drama|danc|animation|fashion|journalism|communication|media|"
        r"advertising|public relations|creative writing|graphic|architect|interior|broadcast|illustrat",
    ),
    (
        "humanities",
        r"english|literature|history|philosoph|linguistic|language|spanish|french|german|japanese|chinese|korean|"
        r"classics|theolog|religio|humanities|translation|writing",
    ),
    (
        "hospitality_aviation",
        r"hospitality|hotel|touris|culinary|chef|cook|aviation|pilot|mechanic|electrician|welding|automotive|"
        r"carpent|military|navy|army|air force",
    ),
]
_COMPILED = [(field, re.compile(rx)) for field, rx in _RULES]
_VAGUE = re.compile(
    r"^(no|none|nil|na|n/?a|-+|\.|idk|i dont know|not sure|undecided|undeclared|general( studies| education)?|nothing|yes|x+|other)$"
)
_FAKE_WORDS = ("VCL6", "VCL9", "VCL12")  # not real words: ticking them marks a careless respondent


def map_major(text: str) -> str | None:
    text = text.strip().lower()
    if not text or _VAGUE.match(text):
        return None
    for field, rx in _COMPILED:
        if rx.search(text):
            return field
    return None


@dataclass
class Dataset:
    x: np.ndarray  # n x 6 interest scores, 0-100 (mean of eight 1-5 ratings, rescaled)
    y: np.ndarray  # n field indices into FIELD_KEYS
    country: np.ndarray
    age: np.ndarray
    stats: dict


FIELD_KEYS = list(FIELDS)


def load(min_education: int = 2) -> Dataset:
    xs, ys, countries, ages = [], [], [], []
    stats = {"rows": 0, "careless": 0, "no_major": 0, "unmapped": 0, "low_education": 0, "kept": 0}
    with open(RIASEC_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stats["rows"] += 1
            if any(row.get(k) == "1" for k in _FAKE_WORDS):
                stats["careless"] += 1
                continue
            if not row["major"].strip():
                stats["no_major"] += 1
                continue
            try:
                education = int(row["education"])
            except ValueError:
                education = 0
            if education < min_education:
                stats["low_education"] += 1
                continue
            field = map_major(row["major"])
            if field is None:
                stats["unmapped"] += 1
                continue
            try:
                ratings = [[int(row[f"{d}{i}"]) for i in range(1, 9)] for d in DIMS]
            except (KeyError, ValueError):
                continue
            if any(v < 1 or v > 5 for r in ratings for v in r):
                continue
            xs.append([(np.mean(r) - 1) / 4 * 100 for r in ratings])
            ys.append(FIELD_KEYS.index(field))
            countries.append(row["country"])
            try:
                ages.append(int(row["age"]))
            except ValueError:
                ages.append(-1)
            stats["kept"] += 1
    return Dataset(np.array(xs, dtype=np.float32), np.array(ys), np.array(countries), np.array(ages), stats)
