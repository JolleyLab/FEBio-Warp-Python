"""Survey a folder of FEBio inputs: which read, round-trip, and can run.

    python tools/survey_inputs.py <folder or files...> [--json out.json]

For every .feb found: the spec version; for spec 4.0, whether it reads, whether
a write and re-read gives an equal model, and the coverage report. Prints
totals and the features that most often stand between an input and a run.
Nothing is solved here.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import tempfile
import time


def spec_version(path):
    with open(path, "rb") as f:
        head = f.read(4000).decode("latin-1", "replace")
    m = re.search(r'<febio_spec\s+version\s*=\s*"([^"]+)"', head)
    return m.group(1) if m else None


def survey(paths, materials="minimal"):
    from warp_febio.compiled import inspect
    from warp_febio.io import read_feb, write_feb

    rows = []
    for path in paths:
        row = {"file": path, "spec": spec_version(path)}
        rows.append(row)
        if row["spec"] != "4.0":
            row["status"] = "other spec"
            continue
        try:
            t = time.time()
            model = read_feb(path)
            row["read_seconds"] = round(time.time() - t, 3)
        except Exception as err:  # the survey must not stop on one file
            row["status"] = "read failed"
            row["error"] = f"{type(err).__name__}: {err}"[:300]
            continue
        row["module"] = model.module
        row["nodes"] = len(model.mesh.nodes)
        row["elements"] = sum(b.count for b in model.mesh.blocks)
        row["element_types"] = sorted({b.febio_type for b in model.mesh.blocks})
        row["issues"] = [i.where for i in model.issues]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "o.feb")
                write_feb(model, out)
                row["round_trip"] = bool(model.equals(read_feb(out)))
        except Exception as err:
            row["round_trip"] = False
            row["error"] = f"{type(err).__name__}: {err}"[:300]
        try:
            report = inspect(model, materials)
            row["missing"] = sorted({f"{e.category}: {e.febio_name}" for e in report.missing()})
            row["status"] = "runnable" if report.runnable else "not yet runnable"
        except Exception as err:
            row["status"] = "inspect failed"
            row["error"] = f"{type(err).__name__}: {err}"[:300]
    return rows


def summarize(rows, top=25):
    n = len(rows)
    by = collections.Counter(r["status"] for r in rows)
    print(f"inputs: {n}")
    print(
        "spec versions:",
        dict(collections.Counter(r["spec"] for r in rows).most_common()),
    )
    for k, v in by.most_common():
        print(f"  {k}: {v}")
    v4 = [r for r in rows if r["spec"] == "4.0" and "module" in r]
    print(
        "spec 4.0 read:",
        len(v4),
        "| round trip equal:",
        sum(r.get("round_trip", False) for r in v4),
        "| with reader issues:",
        sum(bool(r["issues"]) for r in v4),
    )
    print("modules:", dict(collections.Counter(r["module"] for r in v4).most_common()))
    print(
        "element types:",
        dict(collections.Counter(t for r in v4 for t in r["element_types"]).most_common()),
    )
    missing = collections.Counter(m for r in v4 for m in r.get("missing", []))
    print(f"\nmost common missing features (of {len(v4)} inputs):")
    for k, v in missing.most_common(top):
        print(f"  {v:4d}  {k}")
    for label in ("read failed", "inspect failed"):
        bad = [r for r in rows if r["status"] == label]
        if bad:
            print(f"\n{label}:")
            for r in bad:
                print("  ", os.path.basename(r["file"]), "-", r.get("error"))
    bad = [r for r in v4 if not r.get("round_trip", False)]
    if bad:
        print("\nround trip not equal:")
        for r in bad:
            print("  ", os.path.basename(r["file"]), "-", r.get("error", ""))
    print(
        "\nrunnable:",
        [os.path.basename(r["file"]) for r in rows if r["status"] == "runnable"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--json")
    p.add_argument("--materials", default="minimal")
    a = p.parse_args()
    files = []
    for x in a.paths:
        files += sorted(glob.glob(os.path.join(x, "**", "*.feb"), recursive=True)) if os.path.isdir(x) else [x]
    rows = survey(files, a.materials)
    summarize(rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
