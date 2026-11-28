# SPDX-FileCopyrightText: Copyright (c) 2026 The FEBio-Warp-Python Authors
# SPDX-License-Identifier: Apache-2.0
"""The contact potential in a run (contact/in_run.py), against FEBio.

DRAFT (contact in a run, November 2026).

    bk_contact_potential_patch   two hex8 blocks with surface meshes that do
                                 not match, pressed together (the setup of
                                 suite problem co62 on a small mesh).
    sh_self_contact_fold         a quad4 shell surface folded onto itself,
                                 with the contact settings of the open mitral
                                 input (kc 2, R_out 0.5, w_tol 0.6).

The FEBio results of both were stored before this code existed.
"""

import hashlib
import json
import pathlib
import unittest

import numpy as np
import warp as wp
from support import DEVICE, setUpModule, tearDownModule  # noqa: F401  (device for the module)

from warp_febio.analysis.steps import build_system, inspect_model, run
from warp_febio.assembly.system import SolveError
from warp_febio.io import read_feb
from warp_febio.solvers.nonlinear.newton import NewtonSettings, solve_step

CASES = pathlib.Path(__file__).resolve().parents[1] / "references" / "small_problems"


def load(case):
    ref = json.loads((CASES / "febio" / f"{case}.json").read_text())
    deck = CASES / ref["input"]
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == ref["input_sha256"]
    return ref, read_feb(deck)


def record(ref, name):
    rec = next(r for r in ref["records"] if name in r["data"])
    return rec["items"], np.array(rec["values"]), rec["data"]


def contact_terms(system):
    return [t for t in system.terms if getattr(t, "element_type", "") == "contact"]


class FebioAgreementTests(unittest.TestCase):
    def nodal(self, model, results, ref, name, field):
        items, logged, names = record(ref, name)
        nodes = model.mesh.node_index(items)
        column = "xyz".index(name[-1])
        mine = np.stack([getattr(s, field)[nodes, column] for s in results.states])
        return mine, logged[:, :, names.index(name)]

    def test_two_blocks_pressed_together(self):
        ref, model = load("bk_contact_potential_patch")
        system = build_system(model, device=DEVICE)
        results = run(system)
        mine, logged = self.nodal(model, results, ref, "Rz", "reaction_force")
        self.assertLess(abs(mine - logged).max() / abs(logged).max(), 1e-10)
        items, logged, _ = record(ref, "sx")
        stress = np.stack([np.concatenate([s.stress[b.name] for b in model.mesh.blocks]) for s in results.states])
        self.assertLess(abs(stress[:, np.array(items) - 1] - logged).max() / abs(logged).max(), 1e-10)
        # The blocks start apart: no pairs, and one iteration per step, until they meet.
        self.assertEqual(results.states[1].iterations, 1)
        (contact,) = contact_terms(system)
        self.assertGreater(contact.pairs, 0)
        self.assertGreater(float(abs(logged[-1]).max()), 0.0)

    def test_shell_folded_onto_itself(self):
        """Self-contact of one surface, with w_tol above zero. FEBio applies
        that gate per element pair and the search here per point pair; on
        this case the two give the same result."""
        ref, model = load("sh_self_contact_fold")
        results = run(build_system(model, device=DEVICE))
        mine, logged = self.nodal(model, results, ref, "Rz", "reaction_force")
        self.assertLess(abs(mine - logged).max() / abs(logged).max(), 1e-8)
        for name in ("ux", "uy"):
            mine, logged = self.nodal(model, results, ref, name, "displacement")
            self.assertLess(abs(mine - logged).max(), 1e-9)
        entry = next(e for e in inspect_model(model).entries if e.category == "contact")
        self.assertIn("w_tol", entry.note)


class StepRuleTests(unittest.TestCase):
    def test_a_move_beyond_r_out_rejects_the_step(self):
        """No node may move farther than R_out in one iterate. Taking the
        whole approach of the two blocks in one step breaks the rule."""
        _, model = load("bk_contact_potential_patch")
        system = build_system(model, device=DEVICE)
        (contact,) = contact_terms(system)
        self.assertEqual(contact.max_move, 0.5)
        contact.r_out = 0.01  # far below the driven displacement of one step
        settings = NewtonSettings.from_record(model.steps[0].control.child("solver"))
        with self.assertRaises(SolveError) as caught:
            solve_step(system, system.state(), 1.0, settings)
        self.assertIn("R_out", str(caught.exception))


if __name__ == "__main__":
    wp.config.quiet = True
    unittest.main()
