"""Small JSON records; field arrays should be stored separately with named units."""

from pathlib import Path
import json


def write_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
