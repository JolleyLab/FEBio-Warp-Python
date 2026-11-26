"""FEBio-Warp material models against pinned FEBio references.

Each ``mp_*`` case in references/small_problems is one hex8 element driven by
FEBio's "prescribed deformation" boundary condition, so the deformation is
uniform and FEBio's logged stress and tangent are point values. The test
reads the material and the deformation from the same FEBio input file the
reference was generated from, evaluates the FEBio-Warp material model of
that FEBio type, and compares with the stored FEBio values.

Uncoupled materials (``formulation == "uncoupled"``) supply only their
deviatoric part. FEBio logs the total, so for those cases the test adds the
volumetric terms exactly as FEUncoupledMaterial::Stress and ::Tangent do, with
the model's pressure model. That sum is made here for the comparison only; in
FEBio-Warp the two parts are meant to be combined by a multi-field element.

The chord case (``ch_chord_slack_tension``) is compared the same way: FEBio
logged the elongation and force of one nonlinear spring through slack and
tension, and the FEBio-Warp spring is evaluated at the logged elongations.

These are the comparisons of FEBio-Warp with FEBio that need no input
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
    from warp_febio.materials.geometry import axes_from_vectors
    import warp_febio.materials.models  # noqa: F401  (registers the models)
except ImportError:
    api = None

# U'(J) and U''(J) of FEBio's pressure models (FEBioMech/FEUncoupledMaterial.cpp).
PRESSURE = {
    "default": (
        lambda J, k: k * np.log(J) / J,
        lambda J, k: k * (1 - np.log(J)) / J**2,
    ),
    "NIKE3D": (
        lambda J, k: 0.5 * k * (J - 1 / J),
        lambda J, k: 0.5 * k * (1 + 1 / J**2),
    ),
    "Abaqus": (lambda J, k: k * (J - 1), lambda J, k: k + 0 * J),
    "Abaqus (GOH)": (
        lambda J, k: 0.5 * k * (J - 1 / J),
        lambda J, k: 0.5 * k * (1 + 1 / J**2),
    ),
}
# dyad1s(I) and dyad4s(I) as 6x6, in the order xx, yy, zz, xy, yz, xz.
IxI = np.zeros((6, 6))
IxI[:3, :3] = 1.0
I4 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


def read_input(path):
    """Material type, parameters and target deformation gradient of an mp case."""
    root = ET.parse(path).getroot()
    mat = root.find("Material/material")
    params = {}
    for p in mat:
        if len(p) == 0 and p.text:
            try:
                params[p.tag] = float(p.text)
            except ValueError:
                params[p.tag] = p.text.strip()  # an option, for example pressure_model
    axis = mat.find("mat_axis")
    if axis is not None:
        params["mat_axis"] = axes_from_vectors(
            [float(x) for x in axis.findtext("a").split(",")],
            [float(x) for x in axis.findtext("d").split(",")],
        )
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
    if any(f.name == "mat_axis" for f in model.geometry):
        # Material axes as FEBio builds them from the input's a and d vectors.
        geometry = {"mat_axis": np.tile(params["mat_axis"].ravel(), (len(F), 1))}
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
    if model.formulation == "uncoupled":
        # FEUncoupledMaterial: sigma = dev + p I,
        # c = c_dev + p (dyad1s(I) - 2 dyad4s(I)) + J U''(J) dyad1s(I), p = U'(J).
        name = params.get("pressure_model", dict(model.options)["pressure_model"])
        dU, d2U = PRESSURE[name]
        J = np.linalg.det(F)
        p = dU(J, values["k"])
        sigma = sigma + p[:, None, None] * np.eye(3)
        c = (
            c
            + p[:, None, None] * (IxI - 2 * I4)
            + (J * d2U(J, values["k"]))[:, None, None] * IxI
        )
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


@unittest.skipIf(api is None, "FEBio-Warp material interface not present")
class ChordAgreementTest(unittest.TestCase):
    def test_spring_force_matches_febio(self):
        try:
            import warp as wp
            from warp_febio.materials.models.nonlinear_spring import nonlinear_spring
            from warp_febio.materials.uniaxial import make_update_kernel
        except ImportError:
            self.skipTest("no FEBio-Warp model of type 'nonlinear spring'")
        ref = json.loads((CASES / "febio" / "ch_chord_slack_tension.json").read_text())
        deck = CASES / ref["input"]
        self.assertEqual(
            hashlib.sha256(deck.read_bytes()).hexdigest(),
            ref["input_sha256"],
            "reference was not generated from this input file",
        )
        mat = ET.parse(deck).getroot().find("Discrete/discrete_material")
        self.assertEqual(mat.get("type"), "nonlinear spring")
        points = [
            tuple(float(x) for x in pt.text.split(","))
            for pt in mat.find("force/points")
        ]
        model = nonlinear_spring(
            len(points), mat.findtext("measure"), mat.findtext("force/extend")
        )
        values = {"scale": float(mat.findtext("scale"))}
        for k, (x, y) in enumerate(points):
            values[f"x{k}"] = x
            values[f"y{k}"] = y
        record = [r for r in ref["records"] if "discrete element force" in r["data"]][0]
        logged = np.array(record["values"])[:, 0, :]
        elongation = logged[:, record["data"].index("discrete element elongation")]
        force = logged[:, record["data"].index("discrete element force")]
        # The reference must cover both branches of the spring.
        self.assertGreater((force == 0.0).sum(), 3)
        self.assertGreater((force > 0.0).sum(), 3)

        n = len(elongation)
        pvec, svec = model.param_vector(), model.state_vector()
        out = wp.zeros(n, dtype=wp.float64)
        wp.launch(
            make_update_kernel(model),
            dim=n,
            inputs=[
                wp.array(elongation, dtype=wp.float64),
                wp.zeros(n, dtype=int),
                wp.array([model.pack(**values)], dtype=pvec),
                wp.zeros(n, dtype=svec),
            ],
            outputs=[out, wp.zeros(n, dtype=wp.float64), wp.zeros(n, dtype=svec)],
        )
        self.assertLess(np.abs(out.numpy() - force).max() / force.max(), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
