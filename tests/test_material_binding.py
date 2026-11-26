"""The binding between the central model and the material layer.

`warp_febio.materials.febio.bind_material` is the one place a FEBio
<material> record is interpreted. These tests cover what the reader, the
coverage report and the element kernel rely on: FEBio-style solid mixtures, a
parameter on a load curve, range checks, and reporting of what is not
interpreted.
"""

import json
import unittest
from pathlib import Path

import numpy as np

try:
    import warp as wp

    from warp_febio.io import read_feb
    from warp_febio.materials import model as api
    from warp_febio.materials import status
    from warp_febio.materials.febio import NOT_SUPPORTED, bind_material
    from warp_febio.model import Record
except ImportError:
    api = None

CASES = Path(__file__).parent.parent / "references" / "small_problems"
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

F = np.array(
    [
        [[1.10, 0.05, 0.00], [0.02, 0.95, 0.03], [0.00, 0.04, 1.02]],
        [[0.90, 0.00, 0.10], [0.00, 1.20, 0.00], [0.05, 0.00, 1.00]],
    ]
)


def material(type_, name="m", **params):
    return Record(
        "material",
        attrs={"id": "1", "name": name, "type": type_},
        children=[Record(k, value=v) for k, v in params.items()],
    )


_KERNELS = {}


def one_pass_at_points(bound, F):
    """Cauchy stress (n, 3, 3) and tangent (n, 6, 6) as an element kernel gets them."""
    model = bound.model
    if model.name not in _KERNELS:
        func = bound.stress_and_tangent()
        pvec = model.param_vector(wp.float64)
        gvec = model.geometry_vector(wp.float64)
        mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)

        def point(
            F: wp.array(dtype=wp.mat33d),
            g: wp.array(dtype=gvec),
            p: wp.array(dtype=pvec),
            sigma: wp.array(dtype=wp.mat33d),
            c: wp.array(dtype=mat66),
        ):
            i = wp.tid()
            s, t = func(F[i], g[i], p[0])
            sigma[i] = s
            c[i] = t

        _KERNELS[model.name] = (wp.kernel(point, module="unique"), pvec, gvec, mat66)
    kernel, pvec, gvec, mat66 = _KERNELS[model.name]
    n = len(F)
    sigma = wp.zeros(n, dtype=wp.mat33d, device="cpu")
    c = wp.zeros(n, dtype=mat66, device="cpu")
    wp.launch(
        kernel,
        dim=n,
        inputs=[
            wp.array(F, dtype=wp.mat33d, device="cpu"),
            wp.array(bound.geometry_array(n), dtype=gvec, device="cpu"),
            wp.array([pvec(*bound.param_values())], dtype=pvec, device="cpu"),
        ],
        outputs=[sigma, c],
        device="cpu",
    )
    return sigma.numpy(), c.numpy()


@unittest.skipIf(api is None, "Warp or the material layer is not installed")
class MaterialBinding(unittest.TestCase):
    def test_solid_mixture_is_the_sum_of_its_solids(self):
        """<material type="solid mixture"> with two <solid> children, as
        FEElasticMixture: stress and tangent are the sums of the children's."""
        a = dict(E=2.0, v=0.3)
        b = dict(c0=1.5, c1=3.0, c2=0.4, k=50.0)
        mix = Record(
            "material",
            attrs={"id": "1", "name": "mix", "type": "solid mixture"},
            children=[
                Record("density", value=1.0),
                Record(
                    "solid",
                    attrs={"type": "neo-Hookean"},
                    children=[Record(k, value=v) for k, v in a.items()],
                ),
                Record(
                    "solid",
                    attrs={"type": "isotropic Lee-Sacks"},
                    children=[Record(k, value=v) for k, v in b.items()],
                ),
            ],
        )
        bound, issues = bind_material(mix)
        self.assertIsNotNone(bound, issues)
        self.assertEqual(
            bound.model.param_names,
            (
                "solid[0].E",
                "solid[0].v",
                "solid[1].c0",
                "solid[1].c1",
                "solid[1].c2",
                "solid[1].k",
            ),
        )
        self.assertEqual(bound.density, 1.0)
        total = api.evaluate(bound.model, F, bound.material.params_at(0.0))["cauchy"]
        parts = sum(
            api.evaluate(api.by_febio_type(t), F, p)["cauchy"]
            for t, p in (("neo-Hookean", a), ("isotropic Lee-Sacks", b))
        )
        self.assertLess(abs(total - parts).max() / abs(parts).max(), 1e-14)

    def test_parameter_on_a_load_curve_matches_febio(self):
        """A modulus on a load curve (the published device model's mechanism).

        FEBio multiplies the parameter by the curve value at the time of the
        step. The reference logs F and the stress of one element at each
        time; the stress from the bound material at that F and time is
        compared with FEBio's. F is read from the log at 12 digits, which
        limits the comparison to about 1e-10; the tolerance is 1e-9.
        """
        model = read_feb(str(CASES / "inputs" / "bk_modulus_load_curve.feb"))
        bound, issues = bind_material(model.materials[0], model)
        self.assertIsNotNone(bound, issues)
        self.assertTrue(bound.time_dependent)
        records = json.loads(
            (CASES / "febio" / "bk_modulus_load_curve.json").read_text()
        )["records"]
        times = np.array(records[1]["times"])
        Fs = np.array(records[1]["values"])[:, 0, :].reshape(-1, 3, 3)
        stress = np.array(records[2]["values"])[:, 0, :]
        moduli = set()
        for k in range(1, len(times)):
            values = bound.material.params_at(times[k])
            moduli.add(round(values["E"], 12))
            s = api.evaluate(bound.model, Fs[k : k + 1], values)["cauchy"][0]
            mine = np.array([s[i, j] for i, j in VOIGT])
            self.assertLess(abs(mine - stress[k]).max() / abs(stress[k]).max(), 1e-9)
        self.assertEqual(moduli, {1.0, 0.25})  # the curve did change the modulus

    def test_material_point_cases_through_the_reader_match_febio(self):
        """The coupled material-point cases of the verification draft, taken
        the way a run takes them: read_feb, bind_material, and the one-pass
        stress and tangent an element kernel calls. Compared with the stored
        FEBio stress at 1e-10 (FEBio logs 12 digits). The uncoupled cases
        need the volumetric part added and are covered by
        test_febio_agreement_material_point.py.
        """
        compared = 0
        for ref_path in sorted((CASES / "febio").glob("mp_*.json")):
            ref = json.loads(ref_path.read_text())
            model = read_feb(str(CASES / ref["input"]))
            bound, issues = bind_material(model.materials[0], model)
            if bound is None or bound.model.formulation != "coupled":
                continue
            with self.subTest(case=ref["case"]):
                bc = next(
                    r for r in model.boundary if r.type == "prescribed deformation"
                )
                target = np.asarray(bc.get("F"), dtype=float).reshape(3, 3)
                curve = model.load_controller(int(bc.child("scale").attrs["lc"]))
                f_rec, s_rec, c_rec = ref["records"]
                times = np.array(f_rec["times"])[1:]  # skip the undeformed state
                scale = np.array([curve.value(t) for t in times])
                Fs = np.eye(3) + scale[:, None, None] * (target - np.eye(3))
                sigma, c = one_pass_at_points(bound, Fs)
                mine = np.array([[s[i, j] for i, j in VOIGT] for s in sigma])
                febio = np.array(s_rec["values"])[1:, 0, :]
                self.assertLess(abs(mine - febio).max() / abs(febio).max(), 1e-10)
                # FEBio logs the first 21 entries of the 6x6 by column.
                mine_c = np.array([[m[n % 6, n // 6] for n in range(21)] for m in c])
                febio_c = np.array(c_rec["values"])[1:, 0, :]
                self.assertLess(abs(mine_c - febio_c).max() / abs(febio_c).max(), 1e-10)
                compared += 1
        self.assertEqual(compared, 9)

    def test_what_is_not_interpreted_is_reported(self):
        for record, expected in (
            (material("no such material", E=1.0), "no such material"),
            (material("neo-Hookean", E=1.0), "neo-Hookean.v"),  # missing
            (material("neo-Hookean", E=1.0, v=0.3, extra=2.0), "neo-Hookean.extra"),
        ):
            bound, issues = bind_material(record)
            self.assertIsNone(bound)
            self.assertIn(
                expected, {i.febio_name for i in issues if i.status == NOT_SUPPORTED}
            )

    def test_range_checks_apply_as_in_febio(self):
        """FEBio rejects v outside [-1, 0.5) for neo-Hookean at initialization."""
        bound, issues = bind_material(material("neo-Hookean", E=1.0, v=0.5))
        self.assertIsNone(bound)
        self.assertTrue(any("out of range" in i.note for i in issues))

    def test_element_and_material_share_status_codes(self):
        from warp_febio.elements import brick8

        self.assertIs(brick8.NEGATIVE_JACOBIAN, status.NEGATIVE_JACOBIAN)
        self.assertIs(brick8.NON_FINITE, status.NON_FINITE)
        self.assertEqual(brick8.OK, status.OK)


if __name__ == "__main__":
    unittest.main()
