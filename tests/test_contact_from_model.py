"""The contact potential block as the reader delivers it.

`from_febio_block` takes the contact record of the central model directly, so
a FEBio `contact potential` block has one route from a .feb file to the
contact module's options.
"""

import unittest
from pathlib import Path

try:
    from warp_febio.contact import from_febio_block
    from warp_febio.io import read_feb
except ImportError:
    from_febio_block = None

ROOT = Path(__file__).parent.parent / "references"


@unittest.skipIf(from_febio_block is None, "Warp or the contact module is not installed")
class ContactBlockFromModel(unittest.TestCase):
    def test_record_and_dict_give_the_same_options(self):
        model = read_feb(str(ROOT / "small_problems" / "inputs" / "bk_contact_potential_patch.feb"))
        record = next(r for r in model.contact if r.type == "contact potential")
        params, kwargs, rule = from_febio_block(record)
        # The values written in the input file.
        self.assertEqual((params.kc, params.p, params.r_in, params.r_out), (1e-2, 4.0, 0.3, 0.5))
        as_dict = dict(kc=1e-2, p=4, R_in=0.3, R_out=0.5, R0_min=0, w_tol=0)
        params_d, kwargs_d, rule_d = from_febio_block(as_dict)
        for name in ("kc", "p", "r_in", "r_out", "r_self", "scale"):
            self.assertEqual(getattr(params, name), getattr(params_d, name))
        self.assertEqual((kwargs, rule), (kwargs_d, rule_d))

    def test_open_mitral_input(self):
        """The leaflet contact block of the mitral reference input translates."""
        model = read_feb(
            str(ROOT / "febio" / "originals" / "cardiofem" / "valve_pressurization" / "mitral_p2prolapse.feb")
        )
        record = next(r for r in model.contact if r.type == "contact potential")
        params, kwargs, _ = from_febio_block(record)
        self.assertEqual(params.kc, float(record.get("kc")))
        self.assertEqual(params.r_out, float(record.get("R_out")))
        self.assertEqual(kwargs["weight_mode"], "current")


if __name__ == "__main__":
    unittest.main()
