"""Run the small problems in FEBio and check what can be checked today.

Usage::

    FEBIO_BIN=/path/to/febio4 python validation/cases/check.py [--work DIR]

For each input file in validation/cases/febio/ this runs FEBio, reads the
data records from the log file, and applies the check that fits the case:

- ``mp_*``  The logged deformation gradient against the one requested, then
            FEBio's Cauchy stress and spatial tangent against the FEBio-Warp
            material model of the same FEBio type, if this checkout has the
            material interface (``warp_febio.materials.model``). Otherwise
            only the deformation gradient is checked.
- ``ch_*``  Spring force against 60 * max(0, elongation - 4).
- ``bk_*``  Closed-form or consistency checks, described per case below.

Results go to validation/cases/results.json. FEBio writes log data with 12
significant digits, so agreement near 1e-11 is the best this can show.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import generate  # noqa: E402


# --------------------------------------------------------------------------
# FEBio log reader
# --------------------------------------------------------------------------
def read_log(path: Path) -> dict:
    """Return {'status', 'febio_version', 'records': {n: [(time, names, rows)]}}.

    rows is an array with one row per item (node or element); column 0 is the
    item id.
    """
    text = path.read_text(errors="replace")
    records: dict[int, list] = {}
    pattern = re.compile(
        r"Data Record #(\d+)\n=+\nStep = \d+\nTime = ([^\n]+)\nData = ([^\n]+)\n((?:[^\n]+\n)*)"
    )
    for m in pattern.finditer(text):
        rows = [
            [float(x) for x in re.split(r"[,\s]+", line.strip())]
            for line in m.group(4).splitlines()
            if line.strip()
        ]
        records.setdefault(int(m.group(1)), []).append(
            (float(m.group(2)), m.group(3).split(";"), np.array(rows))
        )
    version = re.search(r"version (\d+\.\d+\.\d+)", text)
    return {
        "status": (
            "normal" if "N O R M A L   T E R M I N A T I O N" in text else "error"
        ),
        "febio_version": version.group(1) if version else None,
        "records": records,
    }


def run_febio(febio: str, deck: Path, work: Path) -> dict:
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
    return read_log(local.with_suffix(".log"))


def rel(value, reference) -> float:
    reference = np.asarray(reference, dtype=float)
    scale = np.abs(reference).max()
    return float(
        np.abs(np.asarray(value) - reference).max() / (scale if scale else 1.0)
    )


# --------------------------------------------------------------------------
# FEBio-Warp side of the material point check
# --------------------------------------------------------------------------
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
# FEBio's 21 tangent log values are the first 21 entries of the 6x6 stored by
# column (see generate.C_VARS): value n is entry (n % 6, n // 6).
C_INDEX = [(n % 6, n // 6) for n in range(21)]


def warp_material(febio_type: str, params: dict, F: np.ndarray):
    """Cauchy stress (n,3,3) and spatial tangent (n,6,6) from FEBio-Warp."""
    try:
        from warp_febio.materials import model as api
        import warp_febio.materials.models  # noqa: F401  (registers the models)
    except ImportError:
        return None
    try:
        model = api.by_febio_type(febio_type)
    except (KeyError, LookupError, ValueError):
        return None
    if model is None:
        return None
    values = {k: params[k] for k in model.param_names}
    geometry = None
    if "HGO" in febio_type:
        # mat_axis a = (1,0,0), d = (0,1,0): the local frame is the global one.
        geometry = {"mat_axis": np.tile(np.eye(3).ravel(), (len(F), 1))}
    out = api.evaluate(model, F, values, geometry=geometry)
    c = np.zeros((len(F), 6, 6))
    for b, (k, l) in enumerate(VOIGT):
        d = np.zeros((3, 3))
        d[k, l] += 0.5
        d[l, k] += 0.5
        H = np.einsum("ij,njk->nik", d, F)  # so that sym(H F^-1) = d
        cd = api.evaluate(model, F, values, H=H, geometry=geometry)[
            "spatial_tangent_action"
        ]
        for a, (i, j) in enumerate(VOIGT):
            c[:, a, b] = cd[:, i, j]
    return out["cauchy"], c


def check_material_point(name: str, log: dict) -> dict:
    mat, path = name[len("mp_") :].rsplit("_", 1)  # e.g. mp_lee_sacks_shear
    febio_type, params, _ = generate.MATERIALS[mat]
    target = np.array(generate.target(mat, path))
    recs = log["records"]
    # Skip the state at time 0 (F = I, zero stress).
    times = [t for t, _, _ in recs[1]][1:]
    F = np.array([r[0, 1:].reshape(3, 3) for _, _, r in recs[1]][1:])
    s6 = np.array([r[0, 1:] for _, _, r in recs[2]][1:])
    c21 = np.array([r[0, 1:] for _, _, r in recs[3]][1:])
    wanted = np.array([np.eye(3) + t * (target - np.eye(3)) for t in times])
    result = {"states": len(times), "F_vs_requested": rel(F, wanted)}

    warp = warp_material(febio_type, params, wanted)
    if warp is None:
        result["febio_warp"] = "no FEBio-Warp model of this type in this checkout"
        return result
    sigma, c = warp
    s_warp = np.array([[s[i, j] for i, j in VOIGT] for s in sigma])
    c_warp = np.array([[m[a, b] for a, b in C_INDEX] for m in c])
    result["stress_febio_vs_febio_warp"] = rel(s_warp, s6)
    result["tangent_febio_vs_febio_warp"] = rel(c_warp, c21)
    result["max_abs_stress"] = float(np.abs(s6).max())
    return result


# --------------------------------------------------------------------------
# Other cases
# --------------------------------------------------------------------------
def check_chord(log: dict) -> dict:
    data = np.array([r[0, 1:] for _, _, r in log["records"][2]])
    elongation, force = data[:, 0], np.abs(data[:, 1])
    expected = 60.0 * np.maximum(0.0, elongation - 4.0)
    reaction = np.array([r[0, 2] for _, _, r in log["records"][1]])
    return {
        "states": len(data),
        "slack_states": int((elongation <= 4.0).sum()),
        "max_elongation": float(elongation.max()),
        "force_vs_closed_form": rel(force, expected),
        "reaction_vs_spring_force": rel(np.abs(reaction), force),
    }


def _neo_hookean_uniaxial_stretch(E, v, sigma):
    """Axial stretch of a neo-Hookean bar (FEBio's form) under Cauchy stress sigma.

    FEBio: sigma = mu/J (b - I) + lambda ln(J)/J I, with lateral stress zero.
    """
    lam = v * E / ((1 + v) * (1 - 2 * v))
    mu = 0.5 * E / (1 + v)

    def residual(x):
        a, t = x  # axial and lateral stretch
        J = a * t * t
        return np.array(
            [
                mu / J * (a * a - 1) + lam * np.log(J) / J - sigma,
                mu / J * (t * t - 1) + lam * np.log(J) / J,
            ]
        )

    x = np.array([1.0, 1.0])
    for _ in range(50):
        r = residual(x)
        Jm = np.empty((2, 2))
        for k in range(2):
            dx = np.zeros(2)
            dx[k] = 1e-7
            Jm[:, k] = (residual(x + dx) - residual(x - dx)) / 2e-7
        step = np.linalg.solve(Jm, r)
        x = x - step
        if np.abs(step).max() < 1e-14:
            break
    return x


def check_modulus(log: dict) -> dict:
    recs = log["records"]
    times = np.array([t for t, _, _ in recs[2]])
    Fzz = np.array([r[0, 9] for _, _, r in recs[2]])
    sz = np.array([r[0, 3] for _, _, r in recs[3]])
    load = np.interp(times, [0, 0.2, 1], [0, 1, 1]) * 0.05
    E = np.interp(times, [0, 0.5, 0.6, 1], [1, 1, 0.25, 0.25])
    expected = np.array(
        [_neo_hookean_uniaxial_stretch(e, 0.3, p)[0] for e, p in zip(E, load)]
    )
    return {
        "states": len(times),
        "axial_stress_vs_applied": rel(sz[1:], load[1:]),
        "stretch_vs_closed_form": rel(Fzz[1:], expected[1:]),
        "stretch_before_and_after_step": [float(Fzz[5]), float(Fzz[-1])],
    }


def check_damping(log: dict) -> dict:
    recs = log["records"]
    uz = np.array([r[0, 1] for _, _, r in recs[1]])
    vz = np.array([r[0, 2] for _, _, r in recs[1]])
    static = _neo_hookean_uniaxial_stretch(1.0, 0.3, 0.05)[0] - 1.0
    return {
        "states": len(uz),
        "peak_over_static": float(uz.max() / static),
        "final_vs_static": rel(uz[-1], static),
        "final_speed": float(abs(vz[-1])),
    }


def check_hgo_block(log: dict) -> dict:
    recs = log["records"]
    _, _, last = recs[2][-1]  # stress and J for all 8 elements at the end
    stress, J = last[:, 1:7], last[:, 7]
    return {
        "elements": int(len(last)),
        "stress_spread_across_elements": rel(
            stress, np.tile(stress.mean(0), (len(stress), 1))
        ),
        "max_volume_change": float(np.abs(J - 1).max()),
        "final_sx": float(stress[:, 0].mean()),
    }


def check_contact(log: dict) -> dict:
    recs = log["records"][1]
    total = np.array([r[:, 2].sum() for _, _, r in recs])
    uz = np.array([r[0, 1] for _, _, r in recs])
    return {
        "states": len(total),
        "final_displacement": float(uz[-1]),
        "final_reaction": float(total[-1]),
        "reaction_monotonic": bool(np.all(np.diff(np.abs(total)) >= -1e-12)),
        "note": "No independent reference yet; recorded for the FEBio-Warp comparison.",
    }


def check_shell_pressure(log: dict) -> dict:
    uz = np.array([r[0, 3] for _, _, r in log["records"][1]])
    last = log["records"][2][-1][2][:, 1:]
    return {
        "states": len(uz),
        "final_center_deflection": float(uz[-1]),
        "peak_over_final": float(np.abs(uz).max() / abs(uz[-1])),
        "change_over_last_tenth": rel(uz[-1], uz[-len(uz) // 10]),
        "max_abs_stress": float(np.abs(last).max()),
        "note": "No independent reference yet; recorded for the FEBio-Warp comparison.",
    }


def check_self_contact(log: dict) -> dict:
    recs = log["records"][1]
    close = -np.array([r[0, 1] for _, _, r in recs])
    force = np.array([r[:, 2].sum() for _, _, r in recs])
    engaged = close > 1.5  # flaps start 2 mm apart; the potential reaches 0.5 mm
    return {
        "states": len(close),
        "states_within_potential_range": int(engaged.sum()),
        "closing_force_before_range": float(force[~engaged][-1]),
        "closing_force_final": float(force[-1]),
        "final_edge_gap": float(2.0 - close[-1]),
        "note": "No independent reference yet; recorded for the FEBio-Warp comparison.",
    }


def check_ring(log: dict) -> dict:
    recs = log["records"]
    times = np.array([t for t, _, _ in recs[1]])
    ux = np.array([r[0, 1] for _, _, r in recs[1]])
    hoop = np.array([r[:, 1].sum() for _, _, r in recs[2]])

    def at(t):
        return int(np.abs(times - t).argmin())

    return {
        "states": len(times),
        "ring_outer_radius_crimped_soft": float(10 + ux[at(1.0)]),
        "ring_outer_radius_crimped_stiff": float(10 + ux[at(1.5)]),
        "hoop_force_crimped_soft": float(abs(hoop[at(1.0)])),
        "hoop_force_crimped_stiff": float(abs(hoop[at(1.5)])),
        "hoop_force_after_opening": float(abs(hoop[at(2.5)])),
        "note": "No independent reference yet; recorded for the FEBio-Warp comparison.",
    }


CHECKS = {
    "sh_lee_sacks_pressure": check_shell_pressure,
    "sh_self_contact_fold": check_self_contact,
    "bk_ring_in_tube": check_ring,
    "ch_chord_slack_tension": check_chord,
    "bk_modulus_load_curve": check_modulus,
    "bk_mass_damping": check_damping,
    "bk_hgo_three_field": check_hgo_block,
    "bk_contact_potential_patch": check_contact,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    febio = os.environ.get("FEBIO_BIN")
    if not febio:
        sys.exit("Set FEBIO_BIN to a FEBio executable.")

    results = {}
    version = None
    for deck in sorted((HERE / "febio").glob("*.feb")):
        name = deck.stem
        log = run_febio(febio, deck, args.work)
        version = version or log["febio_version"]
        entry = {"febio_status": log["status"]}
        if log["status"] == "normal":
            if name.startswith("mp_"):
                entry.update(check_material_point(name, log))
            else:
                entry.update(CHECKS[name](log))
        results[name] = entry
        print(name, json.dumps(entry))
    (HERE / "results.json").write_text(
        json.dumps({"febio_version": version, "cases": results}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
