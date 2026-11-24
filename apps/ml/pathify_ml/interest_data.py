"""Training data for the Interest Reader: text -> six RIASEC interest scores.

Labels come from O*NET 31.0. Every occupation has an interest profile (1-7 on
each of the six Holland types); each of its task statements and job titles is
labelled with that profile, scaled to 0-1. Hand-labelled activity sentences
(O*NET illustrative activities, the Open Psychometrics items and Pathify's own
item bank) add short first-person-style examples for a single type.

Splits are by occupation, so the test set holds occupations the model never saw.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from .paths import CATALOG, DIMS, ONET_DIR, ONET_NAMES, RIASEC_CSV

ONE_TYPE_HIGH = 0.95
ONE_TYPE_LOW = 0.15


@dataclass
class Example:
    text: str
    target: list[float]  # six values in [0, 1], order R I A S E C
    source: str
    group: str  # occupation code, or "activity"


def _read(name: str) -> list[dict[str, str]]:
    with open(ONET_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def occupation_profiles() -> dict[str, list[float]]:
    """O*NET-SOC code -> six interest scores scaled from 1-7 to 0-1."""
    raw: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read("Career Interest Types.txt"):
        dim = ONET_NAMES.get(row["Element Name"])
        if dim and row["Scale ID"] == "OI":
            raw[row["O*NET-SOC Code"]][dim] = (float(row["Data Value"]) - 1) / 6
    return {code: [v[d] for d in DIMS] for code, v in raw.items() if len(v) == 6}


def one_type(dim: str) -> list[float]:
    return [ONE_TYPE_HIGH if d == dim else ONE_TYPE_LOW for d in DIMS]


def _lower_first(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].lower() + text[1:] if text else text


def activity_examples() -> list[Example]:
    out: list[Example] = []
    area_to_type = {
        r["Specific Interest Areas Element ID"]: ONET_NAMES[r["Career Interest Types Element Name"]]
        for r in _read("Specific Interest Areas to Career Interest Types.txt")
    }
    type_ids = {
        r["Element ID"]: ONET_NAMES[r["Element Name"]]
        for r in _read("Interests Illustrative Activities.txt")
        if r["Element Name"] in ONET_NAMES
    }
    for row in _read("Interests Illustrative Activities.txt"):
        dim = type_ids.get(row["Element ID"]) or area_to_type.get(row["Element ID"])
        if dim:
            out.append(Example(row["Activity"], one_type(dim), "onet_activity", "activity"))

    # Open Psychometrics items, from the dataset's codebook ("R1<TAB>Test the quality of parts...").
    codebook = RIASEC_CSV.parent / "codebook.txt"
    for line in codebook.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and len(parts[0]) == 2 and parts[0][0] in DIMS and parts[0][1].isdigit():
            out.append(Example(parts[-1].strip(), one_type(parts[0][0]), "openpsych_item", "activity"))

    bank = json.loads((CATALOG / "assessment.json").read_text(encoding="utf-8"))
    for item in bank["items"]:
        out.append(Example(item["text"], one_type(item["dimension"]), "pathify_item", "activity"))

    # Each activity also in a student's voice.
    voiced = [
        Example(f"I enjoy doing this: {_lower_first(e.text)}", e.target, e.source + "_voiced", e.group) for e in out
    ]
    return out + voiced


def occupation_examples(per_occupation: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    profiles = occupation_profiles()
    tasks: dict[str, list[str]] = defaultdict(list)
    for row in _read("Task Statements.txt"):
        tasks[row["O*NET-SOC Code"]].append(row["Task"])
    titles: dict[str, list[str]] = defaultdict(list)
    for row in _read("Job Titles.txt"):
        titles[row["O*NET-SOC Code"]].append(row["Job Title"])
    names = {r["O*NET-SOC Code"]: r["Title"] for r in _read("Occupation Data.txt")}

    out: list[Example] = []
    for code, profile in profiles.items():
        picked_tasks = rng.sample(tasks[code], min(per_occupation, len(tasks[code])))
        picked_titles = rng.sample(titles[code], min(per_occupation // 2, len(titles[code])))
        for t in picked_tasks:
            out.append(Example(t, profile, "onet_task", code))
            # The same task as a student might put it.
            out.append(Example(f"I would like to {_lower_first(t)}", profile, "onet_task_voiced", code))
        for t in picked_titles:
            out.append(Example(f"I want to become a {t.lower()}", profile, "onet_title_voiced", code))
        if code in names:
            out.append(Example(names[code], profile, "onet_occupation", code))
    return out


def build(per_occupation: int = 10, seed: int = 13) -> dict[str, list[Example]]:
    """Train/val/test split by occupation; activity sentences all go to train."""
    occ = occupation_examples(per_occupation, seed)
    codes = sorted({e.group for e in occ})
    random.Random(seed).shuffle(codes)
    n = len(codes)
    test_codes = set(codes[: n // 10])
    val_codes = set(codes[n // 10 : n // 5])
    split: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for e in occ:
        key = "test" if e.group in test_codes else "val" if e.group in val_codes else "train"
        split[key].append(e)
    split["train"] += activity_examples()
    random.Random(seed).shuffle(split["train"])
    return split


def student_eval() -> list[dict]:
    from .paths import EVALS

    return [
        json.loads(line) for line in (EVALS / "interest_eval.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
