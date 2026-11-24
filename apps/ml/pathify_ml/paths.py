from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parents[1]
RAW = ML_ROOT / "data" / "raw"
PROCESSED = ML_ROOT / "data" / "processed"
ARTIFACTS = ML_ROOT / "artifacts"
REPORTS = ML_ROOT / "reports"
EVALS = ML_ROOT / "evals"
CATALOG = REPO_ROOT / "data"
# Where the API looks for exported models.
SERVING = REPO_ROOT / "apps" / "api" / "models"

ONET_DIR = RAW / "onet" / "db_31_0_text"
RIASEC_CSV = RAW / "riasec" / "RIASEC_data12Dec2018" / "data.csv"

DIMS = ["R", "I", "A", "S", "E", "C"]
ONET_NAMES = {"Realistic": "R", "Investigative": "I", "Artistic": "A", "Social": "S", "Enterprising": "E", "Conventional": "C"}
