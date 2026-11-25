"""Run FEBio inputs in FEBio and in FEBio-Warp and compare the plot files.

    python tools/compare_with_febio.py --febio /path/to/febio4 --work DIR inputs...

For each input: FEBio is run on a copy in DIR, FEBio-Warp runs the same file,
and displacement and stress are compared at the output times both codes wrote.
Plot files hold single precision, so agreement is reported relative to the
largest value. An input the coverage report refuses is listed with what is
missing; nothing is run for it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def compare(path, febio, work, linear_solver, materials, timeout):
    from warp_febio.analysis.static import SolveError, run
    from warp_febio.compiled import compile
    from warp_febio.coverage import CoverageError
    from warp_febio.io import read_feb, read_xplt, write_xplt
    from warp_febio.io.xplt import plot_file_from_results

    name = os.path.splitext(os.path.basename(path))[0]
    row = {"input": name}
    model = read_feb(path)
    try:
        compiled = compile(model, device="cpu", material_backend=materials)
    except CoverageError as err:
        row["status"] = "not yet runnable"
        row["missing"] = sorted(
            {f"{e.category}: {e.febio_name}" for e in err.report.missing()}
        )
        return row
    shutil.copy(path, os.path.join(work, name + ".feb"))
    try:
        subprocess.run(
            [febio, "-i", name + ".feb", "-silent"],
            cwd=work,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "FEBio timed out"
        return row
    log = os.path.join(work, name + ".log")
    if not (
        os.path.exists(log) and "N O R M A L" in open(log, errors="replace").read()
    ):
        row["status"] = "FEBio did not finish"
        return row
    ref = read_xplt(os.path.join(work, name + ".xplt"))
    t0 = time.time()
    try:
        results = run(compiled, linear_solver=linear_solver)
    except SolveError as err:
        row["status"] = f"FEBio-Warp failed: {err}"
        return row
    row["seconds"] = round(time.time() - t0, 1)
    write_xplt(model, results, os.path.join(work, name + "_warp.xplt"))
    ours = plot_file_from_results(model, results)
    row["states"] = (len(ref.states), len(ours.states))
    row["same_times"] = len(ref.states) == len(ours.states) and bool(
        np.allclose(ref.times, ours.times, rtol=1e-6, atol=1e-9)
    )
    # Compare at the times both wrote.
    pairs = [(i, int(np.argmin(abs(ours.times - t)))) for i, t in enumerate(ref.times)]
    pairs = [
        (i, j)
        for i, j in pairs
        if abs(ours.times[j] - ref.times[i]) <= 1e-6 * max(1.0, abs(ref.times[i]))
    ]
    row["states_compared"] = len(pairs)
    if pairs:
        i, j = map(list, zip(*pairs))
        ua, ub = ref.variable("displacement")[i], ours.variable("displacement")[j]
        row["max_displacement"] = float(abs(ua).max())
        row["displacement"] = float(abs(ua - ub).max() / max(abs(ua).max(), 1e-30))
        sa = np.concatenate(
            [d.reshape(len(i), -1) for d in [x[i] for x in ref.variable("stress")]],
            axis=1,
        )
        sb = np.concatenate(
            [d.reshape(len(j), -1) for d in [x[j] for x in ours.variable("stress")]],
            axis=1,
        )
        row["max_stress"] = float(abs(sa).max())
        # Differences are relative to the largest value. A stress field that is
        # zero in both codes (below 1e-10) is reported as agreeing.
        row["stress"] = (
            float(abs(sa - sb).max() / abs(sa).max()) if abs(sa).max() > 1e-10 else 0.0
        )
    row["status"] = "compared"
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--febio", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--linear-solver", default="cg")
    p.add_argument("--materials", default="minimal")
    p.add_argument("--timeout", type=float, default=300)
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for path in a.inputs:
        row = compare(path, a.febio, a.work, a.linear_solver, a.materials, a.timeout)
        print(row, flush=True)


if __name__ == "__main__":
    sys.exit(main())
