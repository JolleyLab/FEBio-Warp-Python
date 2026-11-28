"""Closure of the open mitral input, compared with FEBio in time.

The input is references/febio/originals/cardiofem/valve_pressurization/
mitral_p2prolapse.feb (FEBio model repository, project 134,
P2Prolapse_Mitral.feb; Aslan et al. 2025): 7,200 quad4 shells with an
isotropic Lee-Sacks material, 100 chord springs in 8 discrete sets, a
pressure of 13.3 kPa ramped on, mass damping, leaflet self-contact by the
contact potential, dynamic.

The FEBio results kept beside it (FEBio 4.9.0) are, per output time, the
first principal stress of every element of six leaflet regions and the
contact area. That run ends in an error termination at time 0.0135 of 0.02,
so the comparison covers the times up to there. The two codes take
different time steps (FEBio used BFGS, and the step size follows the
iteration count), so this run is interpolated linearly in time to FEBio's
output times.

    uv run examples/valve/mitral_closure.py [device] [linear solver] [w_tol]

On a GPU with the "cudss" solver (optional, see solvers/linear/cudss.py) the
run takes about 8 minutes; with "cg" much longer. Passing w_tol 0 runs the
variant without the normal test, whose energy is smooth in the positions.
What was measured is in docs/design/contact_in_run.md.
"""

import pathlib
import sys
import time

import numpy as np
import warp as wp

from warp_febio.analysis.steps import build_system, run
from warp_febio.io import read_feb

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASE = ROOT / "references" / "febio" / "originals" / "cardiofem" / "valve_pressurization"
REGIONS = ("A1", "A2", "A3", "P1", "P2", "P3")


def febio_log_data(path):
    """[(time, rows)] of a FEBio log data file with *Step, *Time, *Data headers."""
    out = []
    for part in path.read_text().split("*Step")[1:]:
        lines = part.strip().split("\n")
        rows = np.array([line.replace(",", " ").split() for line in lines[3:] if line.strip()], dtype=float)
        out.append((float(lines[1].split("=")[1]), rows))
    return out


def first_principal(stress):
    """Largest eigenvalue of stresses given in FEBio's order xx, yy, zz, xy, yz, xz."""
    m = np.zeros(stress.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))):
        m[..., i, j] = m[..., j, i] = stress[..., k]
    return np.linalg.eigvalsh(m)[..., -1]


def closure(device=None, linear_solver="cudss", w_tol=None, log=print):
    wp.config.quiet = True
    model = read_feb(CASE / "mitral_p2prolapse.feb")
    if w_tol is not None:
        next(r for r in model.contact if r.type == "contact potential").set("w_tol", float(w_tol))
    system = build_system(model, device=device)
    start = time.perf_counter()
    results = run(system, linear_solver=linear_solver)
    wall = time.perf_counter() - start
    times = results.times
    part = model.mesh.blocks[0]
    (interface,) = results.states[-1].contact
    area = np.array([s.contact[interface]["contact area"] for s in results.states])
    iterations = sum(s.iterations for s in results.states[1:])
    log(f"{len(times) - 1} steps to time {times[-1]:g}, {iterations} Newton iterations, {wall:.0f} s")
    log(f"final contact area {area[-1]:.2f}, {results.states[-1].contact[interface]['pairs']} active pairs")

    def at(t, values):
        j = int(np.searchsorted(times, t))
        if j == 0:
            return values(0)
        w = (t - times[j - 1]) / (times[j] - times[j - 1])
        return (1.0 - w) * values(j - 1) + w * values(j)

    # First principal stress per element, by region.
    row = {int(e): k for k, e in enumerate(part.ids)}
    squares, reference, largest = {}, {}, 0.0
    for region in REGIONS:
        for t, rows in febio_log_data(CASE / "febio_results" / f"{region}_p2Prolapse.csv"):
            if t == 0.0 or t > times[-1]:
                continue
            elements = np.array([row[int(i)] for i in rows[:, 0]])
            mine = first_principal(at(t, lambda j: results.states[j].stress[part.name][elements]))
            squares[t] = squares.get(t, 0.0) + float(((mine - rows[:, 4]) ** 2).sum())
            reference[t] = reference.get(t, 0.0) + float((rows[:, 4] ** 2).sum())
            largest = max(largest, float(abs(rows[:, 4]).max()))
    relative = {t: np.sqrt(squares[t] / reference[t]) for t in squares}
    log(
        f"first principal stress, {len(relative)} FEBio output times up to {max(relative):.4f} (largest value {largest:.0f}):"
    )
    log(
        f"  rms difference over rms value: median {np.median(list(relative.values())):.2e}, largest {max(relative.values()):.2e}"
    )

    # Contact area.
    febio_area = [
        (t, rows[0, 1]) for t, rows in febio_log_data(CASE / "febio_results" / "contactarea_p2Prolapse_sev.csv")
    ]
    t_febio = np.array([t for t, _ in febio_area])
    a_febio = np.array([a for _, a in febio_area])
    a_mine = np.interp(t_febio, times, area)
    onset = (float(t_febio[np.argmax(a_febio > 0)]), float(times[np.argmax(area > 0)]))
    log(f"contact starts at time {onset[0]:.5f} in FEBio and {onset[1]:.5f} here")
    log(f"contact area at FEBio's last time ({t_febio[-1]:.4f}): FEBio {a_febio[-1]:.2f}, here {a_mine[-1]:.2f}")
    return dict(
        times=times,
        area=area,
        stress_error=relative,
        febio_area=(t_febio, a_febio),
        area_at_febio_times=a_mine,
        onset=onset,
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    closure(
        device=args[0] if args else None,
        linear_solver=args[1] if len(args) > 1 else "cudss",
        w_tol=args[2] if len(args) > 2 else None,
    )
