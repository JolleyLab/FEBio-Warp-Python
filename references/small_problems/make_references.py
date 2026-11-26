"""Generate pinned FEBio references for the small problems.

Usage::

    FEBIO_BIN=/path/to/febio4 python references/small_problems/make_references.py \
        --build "official 4.x release, Linux, Pardiso"

For each input file in ``inputs/`` this runs FEBio once and writes
``febio/<case>.json``: the data records FEBio wrote to its log file, with the
FEBio version, the build description given on the command line, and the hash
of the input file the reference came from. A reference is valid only for the
input with that hash.

This script only records what FEBio produced. It does not check FEBio's
results against anything. The comparison the harness makes is FEBio-Warp
against these files (see docs/design/verification.md).

FEBio writes log data with 12 significant digits, so a comparison against a
reference can be no tighter than about 1e-11 relative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

RECORD = re.compile(
    r"Data Record #(\d+)\n=+\nStep = \d+\nTime = ([^\n]+)\nData = ([^\n]+)\n((?:[^\n]+\n)*)"
)


def read_log(text: str) -> list[dict]:
    """Collect FEBio's log data records as {data, items, times, values}.

    ``values[t][i]`` holds the row for time ``times[t]`` and item ``items[i]``
    (a node or element id), one number per name in ``data``.
    """
    records: dict[int, dict] = {}
    for m in RECORD.finditer(text):
        rows = [
            [float(x) for x in re.split(r"[,\s]+", line.strip())]
            for line in m.group(4).splitlines()
            if line.strip()
        ]
        rec = records.setdefault(
            int(m.group(1)),
            {"data": m.group(3).split(";"), "items": None, "times": [], "values": []},
        )
        rec["items"] = [int(r[0]) for r in rows]
        rec["times"].append(float(m.group(2)))
        rec["values"].append([r[1:] for r in rows])
    return [records[k] for k in sorted(records)]


def make_reference(febio: str, deck: Path, work: Path, build: str) -> dict | None:
    work.mkdir(parents=True, exist_ok=True)
    local = work / deck.name
    shutil.copy(deck, local)
    subprocess.run(
        [febio, "-i", local.name, "-silent"],
        cwd=work,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    text = local.with_suffix(".log").read_text(errors="replace")
    if "N O R M A L   T E R M I N A T I O N" not in text:
        return None
    version = re.search(r"version (\d+\.\d+\.\d+)", text)
    return {
        "case": deck.stem,
        "input": f"inputs/{deck.name}",
        "input_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
        "febio": {"version": version.group(1) if version else None, "build": build},
        "records": read_log(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", required=True, help="description of the FEBio build")
    parser.add_argument("--work", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    febio = os.environ.get("FEBIO_BIN")
    if not febio:
        sys.exit("Set FEBIO_BIN to a FEBio executable.")

    out = HERE / "febio"
    out.mkdir(exist_ok=True)
    for deck in sorted((HERE / "inputs").glob("*.feb")):
        ref = make_reference(febio, deck, args.work, args.build)
        if ref is None:
            print(
                f"{deck.stem}: FEBio did not terminate normally; no reference written"
            )
            continue
        (out / f"{deck.stem}.json").write_text(
            json.dumps(ref, separators=(",", ":")) + "\n"
        )
        print(
            f"{deck.stem}: {len(ref['records'])} records, {len(ref['records'][0]['times'])} states"
        )


if __name__ == "__main__":
    main()
