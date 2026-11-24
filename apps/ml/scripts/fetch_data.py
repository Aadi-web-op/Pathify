"""Download the public datasets the models are trained on.

- O*NET 31.0 database (US Department of Labor, CC BY 4.0): occupation interest
  profiles, task statements and job titles.
- Open Psychometrics RIASEC data (2015-2018, 145,828 respondents): answers to a
  48-item Holland interest test, with self-reported field of study.

Run from apps/ml:  .venv/bin/python -m scripts.fetch_data
"""

import hashlib
import io
import urllib.request
import zipfile

from pathify_ml.paths import RAW

SOURCES = {
    "onet": "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip",
    "riasec": "https://openpsychometrics.org/_rawdata/RIASEC_data12Dec2018.zip",
}


def fetch(name: str, url: str) -> None:
    target = RAW / name
    if target.exists() and any(target.iterdir()):
        print(f"{name}: already present in {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"{name}: downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 (fixed, trusted URLs)
        blob = response.read()
    print(f"{name}: {len(blob) / 1e6:.1f} MB, sha256 {hashlib.sha256(blob).hexdigest()[:16]}…")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            # Refuse paths that would escape the target folder.
            if member.startswith("/") or ".." in member.split("/"):
                raise ValueError(f"unsafe path in archive: {member}")
        archive.extractall(target)


if __name__ == "__main__":
    for name, url in SOURCES.items():
        fetch(name, url)
