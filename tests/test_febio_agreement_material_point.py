"""FEBio-Warp material models against pinned FEBio references.

Each ``mp_*`` case in references/small_problems is one hex8 element driven by
FEBio's "prescribed deformation" boundary condition, so the deformation is
uniform and FEBio's logged stress and tangent are point values. The test
reads the material and the deformation from the same FEBio input file the
reference was generated from, evaluates the FEBio-Warp material model of
that FEBio type, and compares with the stored FEBio values.

This is the one comparison of FEBio-Warp with FEBio that needs no input
reader or solver. It is skipped where the material interface
(``warp_febio.materials.model``) is not present, and for FEBio material
types FEBio-Warp does not implement.
"""

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CASES = Path(__file__).parent.parent / "references" / "small_problems"

# FEBio writes log data with 12 significant digits.
TOLERANCE = 1e-10

VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
# FEBio's 21 tangent log values are the first 21 entries of the 6x6 stored by
# column, in the order xx, yy, zz, xy, yz, xz: value n is entry (n % 6, n // 6).
C_INDEX = [(n % 6, n // 6) for n in range(21)]

try:
    from warp_febio.materials import model as api
    import warp_febio.materials.models  # noqa: F401  (registers the models)
except ImportError:
    api = None


def read_input(path):
    """Material type, parameters and target deformation gradient of an mp case."""
    root = ET.parse(path).getroot()
    mat = root.find("Material/material")
    params = {p.tag: float(p.text) for p in mat if len(p) == 0 and p.text}
    bc = root.find("Boundary/bc[@type='prescribed deformation']")
    F = np.array([float(x) for x in bc.find("F").text.split(",")]).reshape(3, 3)
    points = [
        tuple(float(x) for x in pt.text.split(","))
        for pt in root.find("LoadData/load_controller/points")
    ]
    return mat.get("type"), params, F, np.array(points)


def febio_warp(febio_type, params, F):
    """Cauchy stress (n, 6) and the 21 logged tangent entries (n, 21)."""
    try:
        model = api.by_febio_type(febio_type)
    except (KeyError, LookupError, ValueError):
        model = None
    if model is None:
        return None
    values = {k: params[k] for k in model.param_names}
    geometry = None
    if "HGO" in febio_type:
        # The inputs set mat_axis a = (1,0,0), d = (0,1,0): the global frame.
        geometry = {"mat_axis": np.tile(np.eye(3).ravel(), (len(F), 1))}
    sigma = api.evaluate(model, F, values, geometry=geometry)["cauchy"]
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
    return (
        np.array([[s[i, j] for i, j in VOIGT] for s in sigma]),
        np.array([[m[a, b] for a, b in C_INDEX] for m in c]),
    )


def relative(value, reference):
    return np.abs(value - reference).max() / np.abs(reference).max()


@unittest.skipIf(api is None, "FEBio-Warp material interface not present")
class MaterialPointAgreementTest(unittest.TestCase):
    def test_stress_and_tangent_match_febio(self):
        compared = 0
        for ref_path in sorted((CASES / "febio").glob("mp_*.json")):
            ref = json.loads(ref_path.read_text())
            deck = CASES / ref["input"]
            with self.subTest(case=ref["case"]):
                self.assertEqual(
                    hashlib.sha256(deck.read_bytes()).hexdigest(),
                    ref["input_sha256"],
                    "reference was not generated from this input file",
                )
                febio_type, params, target, curve = read_input(deck)
                f_rec, s_rec, c_rec = ref["records"]
                times = np.array(f_rec["times"])[1:]  # skip the undeformed state
                scale = np.interp(times, curve[:, 0], curve[:, 1])
                F = np.eye(3) + scale[:, None, None] * (target - np.eye(3))
                result = febio_warp(febio_type, params, F)
                if result is None:
                    self.skipTest(f"no FEBio-Warp model of type {febio_type!r}")
                stress, tangent = result
                s_febio = np.array(s_rec["values"])[1:, 0, :]
                c_febio = np.array(c_rec["values"])[1:, 0, :]
                self.assertLess(relative(stress, s_febio), TOLERANCE)
                self.assertLess(relative(tangent, c_febio), TOLERANCE)
                compared += 1
        self.assertGreater(compared, 0)


if __name__ == "__main__":
    unittest.main()
