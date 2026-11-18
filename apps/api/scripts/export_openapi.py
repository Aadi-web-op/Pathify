"""Write the API's OpenAPI schema to a file so the web app can generate typed clients.

Usage: python -m scripts.export_openapi [output-path]
"""

import json
import sys
from pathlib import Path

from app.config import Settings
from app.main import create_app


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "web/openapi.json"
    schema = create_app(Settings(_env_file=None, environment="development")).openapi()
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
