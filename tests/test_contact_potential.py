"""Checks for the contact potential: law, derivatives, tape, exclusion, capsules.

Synthetic geometry, FP64. Runs on the device named by the environment
variable WARP_FEBIO_TEST_DEVICE (default "cpu"); set it to "cuda:0" to run
the same checks on a GPU. Tolerances are stated at each assertion.
"""

import os
import unittest

import numpy as np
import warp as wp

from warp_febio.contact import (
    ContactInterface,
    ContactPoints,
    ContactPotential,
    ContactPotentialTerm,
    ContactSet,
    PotentialParams,
    from_febio_block,
    node_points,
    quad_facet_points,
    segment_points,
)

wp.init()
DEVICE = os.environ.get("WARP_FEBIO_TEST_DEVICE", "cpu")
vec3d = wp.vec3d


def quad_plate(nx, ny, size, z, offset=(0.0, 0.0), node0=0):
    """Flat Quad4 plate in the plane z = const; returns nodes [n,3], quads [m,4]."""
    xs = np.linspace(-size / 2, size / 2, nx + 1) + offset[0]
    ys = np.linspace(-size / 2, size / 2, ny + 1) + offset[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    quads = []
    for i in range(nx):
        for j in range(ny):
            n00 = i * (ny + 1) + j
            quads.append([n00, n00 + (ny + 1), n00 + (ny + 1) + 1, n00 + 1])
    return nodes, np.asarray(quads, dtype=np.int32) + node0


def two_plates(gap, nx=4, size=1.0, n_per_edge=2, rule="gauss"):
    """Two plates facing each other, a gap apart, as two bodies."""
    n_lo, q_lo = quad_plate(nx, nx, size, 0.0)
    n_hi, q_hi = quad_plate(nx, nx, size, gap, offset=(0.013, -0.021), node0=len(n_lo))
    nodes = np.concatenate([n_lo, n_hi])
    pts = ContactPoints.concatenate(
        [
            quad_facet_points(q_lo, nodes, body=0, n_per_edge=n_per_edge, rule=rule),
            quad_facet_points(q_hi, nodes, body=1, n_per_edge=n_per_edge, rule=rule),
        ]
    )
    return nodes, pts


def wp_nodes(nodes, requires_grad=False):
    return wp.array(
        np.asarray(nodes, dtype=np.float64),
        dtype=vec3d,
        device=DEVICE,
        requires_grad=requires_grad,
    )


class ForceLawTests(unittest.TestCase):
    def test_force_is_c1_and_energy_vanishes_at_r_out(self):
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        e = 1e-7
        f_in = P.force(P.r_in[0] - e)
        f_out = P.force(P.r_in[0] + e)
        self.assertAlmostEqual(f_in / f_out, 1.0, places=5)
        slope_in = (P.force(P.r_in[0]) - P.force(P.r_in[0] - e)) / e
        slope_out = (P.force(P.r_in[0] + e) - P.force(P.r_in[0])) / e
        self.assertAlmostEqual(slope_in / slope_out, 1.0, places=4)
        self.assertEqual(P.phi(P.r_out[0]), 0.0)
        self.assertEqual(P.force(P.r_out[0]), 0.0)
        # phi' = -force by finite differences across the blend region
        r = 0.35
        dphi = (P.phi(r + e) - P.phi(r - e)) / (2 * e)
        self.assertAlmostEqual(-dphi / P.force(r), 1.0, places=6)

    def test_parameter_table_broadcasts(self):
        P = PotentialParams(kc=[1.0, 2.0], p=4.0, r_in=0.2, r_out=[0.5, 0.5])
        self.assertEqual(P.n_classes, 2)
        self.assertAlmostEqual(P.force(0.15, cls=1) / P.force(0.15, cls=0), 2.0, places=12)
        Q = PotentialParams(kc=1.0, p=4.0, r_in=[0.2, 0.3], r_out=0.5)
        # each class is continuous at its own R_in (the c1, c2 of that class)
        for cls in (0, 1):
            e = 1e-7
            self.assertAlmostEqual(
                Q.force(Q.r_in[cls] - e, cls) / Q.force(Q.r_in[cls] + e, cls),
                1.0,
                places=5,
            )


class PairKernelTests(unittest.TestCase):
    def setUp(self):
        self.P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        self.nodes, self.pts = two_plates(gap=0.3)
        self.n = len(self.nodes)
        self.cp = ContactPotential(
            self.pts,
            self.P,
            np.array([[0, 0], [0, 0]]),
            self.nodes,
            self.n,
            device=DEVICE,
        )
        self.x = wp_nodes(self.nodes)
        self.cp.build_pairs(self.x)
        self.assertGreater(self.cp.search.n_pairs, 0)

    def energy_and_force(self, nodes):
        f = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        self.cp.energy_force(wp_nodes(nodes), f)
        return self.cp.energy_value(), f.numpy()

    def test_force_matches_energy_gradient(self):
        E0, f0 = self.energy_and_force(self.nodes)
        self.assertGreater(E0, 0.0)
        rng = np.random.default_rng(0)
        direction = rng.standard_normal(self.nodes.shape)
        h = 1e-6
        Ep, _ = self.energy_and_force(self.nodes + h * direction)
        Em, _ = self.energy_and_force(self.nodes - h * direction)
        dE_fd = (Ep - Em) / (2 * h)
        dE_an = -np.sum(f0 * direction)
        self.assertAlmostEqual(dE_fd / dE_an, 1.0, places=8)

    def test_hessian_action_matches_force_derivative(self):
        rng = np.random.default_rng(1)
        v = rng.standard_normal(self.nodes.shape)
        h = 1e-6
        _, fp = self.energy_and_force(self.nodes + h * v)
        _, fm = self.energy_and_force(self.nodes - h * v)
        Kv_fd = -(fp - fm) / (2 * h)  # K = d(-f)/dx = d2E/dx2
        self.energy_and_force(self.nodes)
        out = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        self.cp.hessian_matvec(wp_nodes(v), out, project=False)
        Kv = out.numpy()
        rel = np.linalg.norm(Kv - Kv_fd) / np.linalg.norm(Kv_fd)
        self.assertLess(rel, 1e-7)
        # assembled matrix agrees with the action and is symmetric
        K = self.cp.hessian_dense(project=False)
        self.assertLess(np.linalg.norm(K @ v.ravel() - Kv.ravel()) / np.linalg.norm(Kv), 1e-12)
        self.assertLess(np.abs(K - K.T).max() / np.abs(K).max(), 1e-12)

    def test_projected_tangent_is_positive_semidefinite(self):
        self.energy_and_force(self.nodes)
        K_full = self.cp.hessian_dense(project=False)
        K_proj = self.cp.hessian_dense(project=True)
        lam_full = np.linalg.eigvalsh(K_full)
        lam_proj = np.linalg.eigvalsh(K_proj)
        self.assertLess(lam_full.min(), -1e-6 * np.abs(lam_full).max())
        self.assertGreater(lam_proj.min(), -1e-10 * np.abs(lam_proj).max())

    def test_tape_gradient_equals_force(self):
        x = wp_nodes(self.nodes, requires_grad=True)
        f = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        tape = wp.Tape()
        with tape:
            E = self.cp.energy_force(x, f)
        tape.backward(E)
        g = x.grad.numpy()
        f = f.numpy()
        self.assertLess(np.linalg.norm(g + f) / np.linalg.norm(f), 1e-12)

    def test_term_residual_sign_and_tangent(self):
        term = ContactPotentialTerm(
            self.pts,
            self.P,
            np.array([[0, 0], [0, 0]]),
            self.nodes,
            self.n,
            device=DEVICE,
        )
        term.update_pairs(self.x)
        R = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        term.residual(self.x, None, R)
        _, f = self.energy_and_force(self.nodes)
        self.assertLess(np.linalg.norm(R.numpy() + f) / np.linalg.norm(f), 1e-12)
        T = term.linearize(self.x, None)
        v = np.random.default_rng(2).standard_normal(self.nodes.shape)
        out = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        T.apply(wp_nodes(v), out)
        ref = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        self.cp.hessian_matvec(wp_nodes(v), ref, project=False)
        self.assertLess(
            np.linalg.norm(out.numpy() - ref.numpy()) / np.linalg.norm(ref.numpy()),
            1e-12,
        )
        # tape gradient through the term equals the residual (custom-op building block)
        xg = wp_nodes(self.nodes, requires_grad=True)
        E, g = term.energy_and_gradient(xg)
        self.assertLess(np.linalg.norm(g.numpy() - R.numpy()) / np.linalg.norm(R.numpy()), 1e-12)
        self.assertAlmostEqual(float(E.numpy()[0]) / term.energy(), 1.0, places=12)
        # the term satisfies the generic interface and composes in a ContactSet
        self.assertIsInstance(term, ContactInterface)
        cs = ContactSet([term])
        cs.update_pairs(self.x)
        R2 = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        cs.residual(self.x, None, R2)
        self.assertLess(np.linalg.norm(R2.numpy() - R.numpy()), 1e-14)
        self.assertIn("contact_potential/contact_area", cs.results())


class CapsuleDerivativeTests(unittest.TestCase):
    """Force and Hessian consistency with nonzero radius offsets (the capsule path)."""

    def test_capsule_force_and_hessian_match_finite_differences(self):
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        nodes, pts = two_plates(gap=0.45)
        pts.radius[pts.body == 1] = 0.08
        pts.radius[pts.body == 0] = 0.03
        n = len(nodes)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, n, device=DEVICE)
        cp.build_pairs(wp_nodes(nodes))
        self.assertGreater(cp.search.n_pairs, 0)

        def ef(x_np):
            f = wp.zeros(n, dtype=vec3d, device=DEVICE)
            cp.energy_force(wp_nodes(x_np), f)
            return cp.energy_value(), f.numpy()

        rng = np.random.default_rng(5)
        d = rng.standard_normal(nodes.shape)
        h = 1e-6
        Ep, fp = ef(nodes + h * d)
        Em, fm = ef(nodes - h * d)
        E0, f0 = ef(nodes)
        self.assertAlmostEqual(((Ep - Em) / (2 * h)) / (-np.sum(f0 * d)), 1.0, places=8)
        out = wp.zeros(n, dtype=vec3d, device=DEVICE)
        cp.hessian_matvec(wp_nodes(d), out, project=False)
        Kd_fd = -(fp - fm) / (2 * h)
        self.assertLess(np.linalg.norm(out.numpy() - Kd_fd) / np.linalg.norm(Kd_fd), 1e-7)


class CoreBranchTests(unittest.TestCase):
    """Derivative checks with pairs inside the stiff core (r < R_in)."""

    def test_force_and_hessian_in_the_core(self):
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        nodes, pts = two_plates(gap=0.15)
        n = len(nodes)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, n, device=DEVICE)
        cp.build_pairs(wp_nodes(nodes))
        self.assertLess(cp.min_distance(wp_nodes(nodes)), P.r_in[0])

        def ef(x_np):
            f = wp.zeros(n, dtype=vec3d, device=DEVICE)
            cp.energy_force(wp_nodes(x_np), f)
            return cp.energy_value(), f.numpy()

        d = np.random.default_rng(11).standard_normal(nodes.shape)
        h = 1e-7
        Ep, fp = ef(nodes + h * d)
        Em, fm = ef(nodes - h * d)
        _, f0 = ef(nodes)
        # central differences: relative agreement to 1e-7 (force) and 1e-6 (tangent action)
        self.assertLess(abs(((Ep - Em) / (2 * h)) / (-np.sum(f0 * d)) - 1.0), 1e-7)
        out = wp.zeros(n, dtype=vec3d, device=DEVICE)
        cp.hessian_matvec(wp_nodes(d), out, project=False)
        Kd_fd = -(fp - fm) / (2 * h)
        self.assertLess(np.linalg.norm(out.numpy() - Kd_fd) / np.linalg.norm(Kd_fd), 1e-6)


class GuardTests(unittest.TestCase):
    def test_coincident_and_overlapping_points_do_not_poison_the_sum(self):
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        # nodes 0 and 1 coincide (different bodies); node 2 overlaps node 0's capsule
        nodes = np.array([[0.0, 0, 0], [0.0, 0, 0], [0.05, 0, 0], [0.3, 0, 0]])
        pts = ContactPoints.concatenate([node_points([0], body=0, radius=0.1), node_points([1, 2, 3], body=1)])
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, 4, device=DEVICE)
        cp.build_pairs(wp_nodes(nodes))
        f = wp.zeros(4, dtype=vec3d, device=DEVICE)
        cp.energy_force(wp_nodes(nodes), f)
        self.assertTrue(np.all(np.isfinite(f.numpy())))
        self.assertTrue(np.isfinite(cp.energy_value()))
        self.assertGreater(f.numpy()[3, 0], 0.0)  # the one valid pair (0, 3) still repels

    def test_asymmetric_pair_class_is_rejected(self):
        P = PotentialParams(kc=[1.0, 1.0], p=4.0, r_in=0.2, r_out=0.5)
        nodes = np.array([[0.0, 0, 0], [0.3, 0, 0]])
        pts = ContactPoints.concatenate([node_points([0], body=0), node_points([1], body=1)])
        with self.assertRaises(ValueError):
            ContactPotential(pts, P, np.array([[0, 1], [0, 0]]), nodes, 2, device=DEVICE)

    def test_febio_block_requires_kc(self):
        with self.assertRaises(ValueError):
            from_febio_block({"R_in": 0.2, "R_out": 0.5})


class ExclusionTests(unittest.TestCase):
    def pairs_for(self, nodes, ref_nodes, body, exclusion, pt_nodes=None, r_self=0.75):
        n = len(nodes)
        if pt_nodes is None:
            pts = ContactPoints.concatenate([node_points(np.where(body == b)[0], body=b) for b in np.unique(body)])
        else:
            pts = pt_nodes
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5, r_self=r_self)
        nb = int(pts.body.max()) + 1
        cp = ContactPotential(
            pts,
            P,
            np.zeros((nb, nb), dtype=int),
            ref_nodes,
            n,
            exclusion=exclusion,
            device=DEVICE,
        )
        cp.build_pairs(wp_nodes(nodes))
        return cp.search.pair_array()

    def test_same_body_close_in_reference_is_excluded(self):
        ref = np.array([[0.0, 0, 0], [0.3, 0, 0]])
        self.assertEqual(len(self.pairs_for(ref, ref, np.array([0, 0]), "none")), 0)

    def test_different_bodies_always_interact(self):
        ref = np.array([[0.0, 0, 0], [0.3, 0, 0]])
        self.assertEqual(len(self.pairs_for(ref, ref, np.array([0, 1]), "none")), 1)

    def test_same_body_far_in_reference_interacts_when_folded(self):
        ref = np.array([[0.0, 0, 0], [5.0, 0, 0]])
        cur = np.array([[0.0, 0, 0], [0.3, 0, 0]])
        self.assertEqual(len(self.pairs_for(cur, ref, np.array([0, 0]), "none")), 1)

    def test_shared_node_points_are_excluded_by_topology(self):
        # two points on facets that share node 1, far apart in the reference,
        # brought within r_out in the current configuration
        ref = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
        cur = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]])
        pt_nodes = np.array([[0, 1], [1, 2]])
        pt_w = np.array([[0.5, 0.5], [0.5, 0.5]])
        pts = ContactPoints(
            pt_nodes,
            pt_w,
            np.ones(2),
            np.ones(2),
            np.zeros(2),
            np.zeros(2, dtype=np.int32),
            np.arange(2, dtype=np.int32),
        )
        self.assertEqual(
            len(self.pairs_for(cur, ref, None, "neighbors", pt_nodes=pts, r_self=0.01)),
            0,
        )
        # two points of one facet, 0.4 apart in the current configuration
        same = ContactPoints(
            np.array([[0, 1], [0, 1]]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.ones(2),
            np.ones(2),
            np.zeros(2),
            np.zeros(2, dtype=np.int32),
            np.zeros(2, dtype=np.int32),
        )
        cur2 = np.array([[0.0, 0, 0], [0.5, 0, 0], [10.0, 0, 0]])
        self.assertEqual(
            len(self.pairs_for(cur2, ref, None, "element", pt_nodes=same, r_self=0.01)),
            0,
        )
        self.assertEqual(len(self.pairs_for(cur2, ref, None, "none", pt_nodes=same, r_self=0.01)), 1)
        self.assertEqual(len(self.pairs_for(cur, ref, None, "element", pt_nodes=pts, r_self=0.01)), 1)

    def test_pair_class_minus_one_disables_a_body_pair(self):
        ref = np.array([[0.0, 0, 0], [0.3, 0, 0]])
        pts = ContactPoints.concatenate([node_points([0], body=0), node_points([1], body=1)])
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        cp = ContactPotential(pts, P, np.array([[0, -1], [-1, 0]]), ref, 2, device=DEVICE)
        cp.build_pairs(wp_nodes(ref))
        self.assertEqual(len(cp.search.pair_array()), 0)


class CapsuleAndClassTests(unittest.TestCase):
    def point_pair_force(self, d, radius_a, radius_b, kc_table=(1.0,), pair_class=((0, 0), (0, 0))):
        nodes = np.array([[0.0, 0, 0], [0.0, 0, d]])
        pts = ContactPoints.concatenate(
            [
                node_points([0], body=0, radius=radius_a),
                node_points([1], body=1, radius=radius_b),
            ]
        )
        P = PotentialParams(kc=np.array(kc_table), p=4.0, r_in=0.2, r_out=0.5)
        cp = ContactPotential(pts, P, np.array(pair_class), nodes, 2, device=DEVICE)
        cp.build_pairs(wp_nodes(nodes))
        f = wp.zeros(2, dtype=vec3d, device=DEVICE)
        cp.energy_force(wp_nodes(nodes), f)
        return f.numpy()[1, 2], cp.energy_value()

    def test_radius_offsets_shift_the_law_exactly(self):
        f0, E0 = self.point_pair_force(0.30, 0.0, 0.0)
        f1, E1 = self.point_pair_force(0.30 + 0.1 + 0.15, 0.1, 0.15)
        self.assertAlmostEqual(f1 / f0, 1.0, places=12)
        self.assertAlmostEqual(E1 / E0, 1.0, places=12)
        self.assertGreater(f0, 0.0)

    def test_pair_class_selects_parameters(self):
        f_a, _ = self.point_pair_force(0.30, 0.0, 0.0, kc_table=(1.0, 3.0), pair_class=((0, 0), (0, 0)))
        f_b, _ = self.point_pair_force(0.30, 0.0, 0.0, kc_table=(1.0, 3.0), pair_class=((0, 1), (1, 0)))
        self.assertAlmostEqual(f_b / f_a, 3.0, places=12)

    def test_crossing_wires_repel_through_their_radii(self):
        """Two perpendicular wires of radius r_w: the law acts on the surface gap."""
        r_w, L = 0.05, 2.0
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.1, r_out=0.25)

        # wire 0 along x at z = 0, wire 1 along y at height h (nodes 0,1 and 2,3)
        def nodes_at(h):
            return np.array([[-L / 2, 0, 0], [L / 2, 0, 0], [0, -L / 2, h], [0, L / 2, h]])

        ref = nodes_at(1.0)
        pts = ContactPoints.concatenate(
            [
                segment_points(np.array([[0, 1]]), ref, body=0, spacing=P.r_out[0] / 4, radius=r_w),
                segment_points(np.array([[2, 3]]), ref, body=1, spacing=P.r_out[0] / 4, radius=r_w),
            ]
        )
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), ref, 4, device=DEVICE)

        def vertical_force(h):
            x = wp_nodes(nodes_at(h))
            cp.build_pairs(x)
            f = wp.zeros(4, dtype=vec3d, device=DEVICE)
            cp.energy_force(x, f)
            return f.numpy()[2:, 2].sum()

        # the force is repulsive and grows monotonically as the wires approach
        gaps = np.array([0.9, 0.7, 0.5, 0.3]) * P.r_out[0]
        forces = np.array([vertical_force(2 * r_w + g) for g in gaps])
        self.assertTrue(np.all(forces > 0.0))
        self.assertTrue(np.all(np.diff(forces) > 0.0))
        # no interaction beyond 2 r_w + R_out, and the reported gap excludes the radii
        self.assertEqual(vertical_force(2 * r_w + 1.01 * P.r_out[0]), 0.0)
        h = 2 * r_w + 0.5 * P.r_out[0]
        vertical_force(h)
        dmin = cp.min_distance(wp_nodes(nodes_at(h)))  # sampled points straddle the crossing
        self.assertGreaterEqual(dmin, 0.5 * P.r_out[0])
        self.assertLess(dmin, 0.5 * P.r_out[0] + 0.01)


class SegmentRuleTests(unittest.TestCase):
    def test_nodal_rule_matches_febio_line_element(self):
        nodes = np.array([[0.0, 0, 0], [2.0, 0, 0]])
        pts = segment_points(np.array([[0, 1]]), nodes, body=0, rule="nodal")
        self.assertEqual(pts.count, 2)
        np.testing.assert_allclose(pts.positions(nodes), nodes)
        np.testing.assert_allclose(pts.weight, [1.0, 1.0])  # half the length each
        lat = segment_points(np.array([[0, 1]]), nodes, body=0, spacing=0.5)
        self.assertEqual(lat.count, 4)
        self.assertAlmostEqual(lat.weight.sum(), 2.0)


class MetricTests(unittest.TestCase):
    def test_contact_area_and_fraction(self):
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        nodes, pts = two_plates(gap=0.3, nx=4, size=1.0)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, len(nodes), device=DEVICE)
        x = wp_nodes(nodes)
        cp.build_pairs(x)
        f = wp.zeros(len(nodes), dtype=vec3d, device=DEVICE)
        cp.energy_force(x, f)
        area0 = cp.contact_area(body=0)
        self.assertGreater(area0, 0.5)  # most of a unit plate is within r_out of the other
        self.assertLessEqual(area0, 1.0 + 1e-12)
        self.assertAlmostEqual(cp.contact_fraction(0, 0.5, 1, nodes), area0, places=12)
        self.assertEqual(cp.contact_fraction(0, 0.1, 1, nodes), 0.0)


class ReferenceTierTests(unittest.TestCase):
    """FEBio semantics that the reference tier switches on."""

    def setUp(self):
        self.P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        self.nodes, self.pts = two_plates(gap=0.3)
        self.n = len(self.nodes)

    def test_w_tol_drops_oblique_pairs(self):
        cp = ContactPotential(
            self.pts,
            self.P,
            np.zeros((2, 2), dtype=int),
            self.nodes,
            self.n,
            w_tol=0.6,
            device=DEVICE,
        )
        n06 = cp.build_pairs(wp_nodes(self.nodes))
        counts = []
        for w_tol in (0.0, 0.8, 0.99):
            cp = ContactPotential(
                self.pts,
                self.P,
                np.zeros((2, 2), dtype=int),
                self.nodes,
                self.n,
                w_tol=w_tol,
                device=DEVICE,
            )
            counts.append(cp.build_pairs(wp_nodes(self.nodes)))
        # w_tol = 0.6 (the Aslan value) removes the in-plane same-plate pairs and keeps
        # every plate-to-plate pair; larger values start dropping oblique plate-to-plate pairs
        self.assertGreater(counts[0], n06)
        self.assertGreater(n06, counts[1])
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        self.assertGreater(counts[2], 0)

    def test_current_weights_track_the_deformed_area(self):
        cp_ref = ContactPotential(
            self.pts,
            self.P,
            np.zeros((2, 2), dtype=int),
            self.nodes,
            self.n,
            device=DEVICE,
        )
        cp_cur = ContactPotential(
            self.pts,
            self.P,
            np.zeros((2, 2), dtype=int),
            self.nodes,
            self.n,
            weight_mode="current",
            device=DEVICE,
        )
        # undeformed: identical
        f_ref, f_cur = wp.zeros(self.n, dtype=vec3d, device=DEVICE), wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        x = wp_nodes(self.nodes)
        cp_ref.build_pairs(x)
        cp_ref.energy_force(x, f_ref)
        cp_cur.build_pairs(x)
        cp_cur.energy_force(x, f_cur)
        self.assertLess(np.linalg.norm(f_ref.numpy() - f_cur.numpy()), 1e-12)
        # stretch both plates in-plane by 10%: current weights scale by 1.21 per side
        stretched = self.nodes.copy()
        stretched[:, :2] *= 1.1
        x2 = wp_nodes(stretched)
        f_ref.zero_()
        f_cur.zero_()
        cp_ref.build_pairs(x2)
        cp_ref.energy_force(x2, f_ref)
        cp_cur.build_pairs(x2)
        cp_cur.energy_force(x2, f_cur)
        ratio = np.linalg.norm(f_cur.numpy()) / np.linalg.norm(f_ref.numpy())
        self.assertAlmostEqual(ratio, 1.21 * 1.21, places=6)

    def test_scale_two_doubles_the_pair_force(self):
        P2 = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5, scale=2.0)
        f1, f2 = wp.zeros(self.n, dtype=vec3d, device=DEVICE), wp.zeros(self.n, dtype=vec3d, device=DEVICE)
        x = wp_nodes(self.nodes)
        for P, f in ((self.P, f1), (P2, f2)):
            cp = ContactPotential(
                self.pts,
                P,
                np.zeros((2, 2), dtype=int),
                self.nodes,
                self.n,
                device=DEVICE,
            )
            cp.build_pairs(x)
            cp.energy_force(x, f)
        self.assertAlmostEqual(np.linalg.norm(f2.numpy()) / np.linalg.norm(f1.numpy()), 2.0, places=12)


class FebioBlockTests(unittest.TestCase):
    def test_cardiofem_block_round_trips(self):
        # the archived CardioFEM contact block (examples/cardiofem_valve/febio_cases.json)
        block = {
            "kc": 2.0,
            "p": 4.0,
            "R_in": 0.2,
            "R_out": 0.5,
            "R0_min": 0.0,
            "w_tol": 0.6,
        }
        P, kw, rule = from_febio_block(block, scale=2.0)  # primary == secondary surface
        self.assertEqual(P.n_classes, 1)
        self.assertAlmostEqual(P.kc[0], 2.0)
        self.assertAlmostEqual(P.r_in[0], 0.2)
        self.assertAlmostEqual(P.r_out[0], 0.5)
        self.assertAlmostEqual(P.scale[0], 2.0)
        self.assertEqual(kw, {"w_tol": 0.6, "exclusion": "element", "weight_mode": "current"})
        self.assertEqual(rule, {"quad_n_per_edge": 2, "tri_order": "3"})
        nodes, pts = two_plates(gap=0.3, n_per_edge=rule["quad_n_per_edge"])
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, len(nodes), device=DEVICE, **kw)
        self.assertGreater(cp.build_pairs(wp_nodes(nodes)), 0)
        _, kw2, rule2 = from_febio_block({**block, "integration_rule": "higher-order", "exclude_neighbors": 1})
        self.assertEqual(kw2["exclusion"], "neighbors")
        self.assertEqual(rule2, {"quad_n_per_edge": 4, "tri_order": "7"})


class AssemblyTests(unittest.TestCase):
    def test_bsr_assembly_matches_matrix_free_action(self):
        import warp.sparse as wps

        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        nodes, pts = two_plates(gap=0.3)
        n = len(nodes)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, n, device=DEVICE)
        x = wp_nodes(nodes)
        cp.build_pairs(x)
        f = wp.zeros(n, dtype=vec3d, device=DEVICE)
        cp.energy_force(x, f)
        A = cp.hessian_bsr(project=False)
        v = np.random.default_rng(3).standard_normal(nodes.shape)
        y = wps.bsr_mv(A, wp_nodes(v))
        ref = wp.zeros(n, dtype=vec3d, device=DEVICE)
        cp.hessian_matvec(wp_nodes(v), ref, project=False)
        self.assertLess(np.linalg.norm(y.numpy() - ref.numpy()) / np.linalg.norm(ref.numpy()), 1e-12)
        # reuse of the destination matrix
        A2 = cp.hessian_bsr(project=True, dest=A)
        y2 = wps.bsr_mv(A2, wp_nodes(v))
        ref.zero_()
        cp.hessian_matvec(wp_nodes(v), ref, project=True)
        self.assertLess(
            np.linalg.norm(y2.numpy() - ref.numpy()) / np.linalg.norm(ref.numpy()),
            1e-12,
        )

    @unittest.skipUnless(wp.get_cuda_device_count() > 0, "CUDA graph capture needs a GPU")
    def test_newton_body_is_graph_capturable(self):
        device = "cuda:0"
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.2, r_out=0.5)
        nodes, pts = two_plates(gap=0.3)
        n = len(nodes)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, n, device=device)
        x = wp.array(nodes, dtype=vec3d, device=device)
        f = wp.zeros(n, dtype=vec3d, device=device)
        cp.build_pairs(x)  # warm up and size the buffers
        with wp.ScopedCapture(device=device) as capture:
            cp.build_pairs(x, sync=False)
            cp.energy_force(x, f)
            cp.hessian_blocks(project=False)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        self.assertFalse(cp.search.check_overflow())
        self.assertGreater(np.linalg.norm(f.numpy()), 0.0)


class FebioDiagnosticTests(unittest.TestCase):
    """Same check as FEBio's contact diagnostic (FEBioTest/FEContactDiagnostic.cpp).

    FEBio builds two unit hex elements whose facing surfaces overlap by eps = 0.5,
    assembles the contact stiffness of a sliding interface, differentiates the
    contact residual by finite differences node by node, and passes when the
    largest entry-wise difference is below 1e-3 percent. Here the geometry,
    the two surfaces and the pass criterion are FEBio's; the law is the
    contact potential on 2x2 Gauss points, and the differences are central.
    """

    def test_two_hex_contact_stiffness_matches_residual_derivative(self):
        eps = 0.5
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 1 - eps],
                [1, 0, 1 - eps],
                [1, 1, 1 - eps],
                [0, 1, 1 - eps],
                [0, 0, 2 - eps],
                [1, 0, 2 - eps],
                [1, 1, 2 - eps],
                [0, 1, 2 - eps],
            ],
            dtype=np.float64,
        )
        primary = np.array([[4, 5, 6, 7]])  # top face of element 1 (FEBio ms)
        secondary = np.array([[11, 10, 9, 8]])  # bottom face of element 2 (FEBio ss)
        pts = ContactPoints.concatenate(
            [
                quad_facet_points(primary, nodes, body=0, n_per_edge=2),
                quad_facet_points(secondary, nodes, body=1, n_per_edge=2),
            ]
        )
        P = PotentialParams(kc=1.0, p=4.0, r_in=0.4, r_out=1.0)
        n = len(nodes)
        cp = ContactPotential(pts, P, np.zeros((2, 2), dtype=int), nodes, n, device=DEVICE)

        def residual(x_np):
            f = wp.zeros(n, dtype=vec3d, device=DEVICE)
            cp.energy_force(wp_nodes(x_np), f)
            return -f.numpy().ravel()  # contact residual = dE/dx

        cp.build_pairs(wp_nodes(nodes))
        self.assertGreater(cp.search.n_pairs, 0)
        K0 = cp.hessian_dense(project=False)
        K1 = np.zeros_like(K0)
        dx = 1e-6
        for j in range(3 * n):
            xp = nodes.copy().ravel()
            xp[j] += dx
            xm = nodes.copy().ravel()
            xm[j] -= dx
            K1[:, j] = (residual(xp.reshape(-1, 3)) - residual(xm.reshape(-1, 3))) / (2 * dx)
        denom = np.abs(K0)
        denom[denom < 1e-15] = 1.0
        kmax_percent = 100.0 * np.max(np.abs(K1 - K0) / denom)
        self.assertLess(kmax_percent, 1e-3)


class DifferentiabilityTests(unittest.TestCase):
    """What differentiates, in both weight modes, and the parameter adjoint."""

    def setUp(self):
        self.P = PotentialParams(kc=[1.0, 2.0], p=4.0, r_in=0.2, r_out=0.5)
        self.nodes, self.pts = two_plates(gap=0.3)
        self.n = len(self.nodes)
        self.pair_class = np.array([[0, 1], [1, 0]])

    def term(self, weight_mode):
        t = ContactPotentialTerm(
            self.pts,
            self.P,
            self.pair_class,
            self.nodes,
            self.n,
            weight_mode=weight_mode,
            device=DEVICE,
        )
        t.update_pairs(wp_nodes(self.nodes))
        return t

    def test_tape_gradient_equals_residual_in_both_weight_modes(self):
        for mode in ("reference", "current"):
            t = self.term(mode)
            R = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
            t.residual(wp_nodes(self.nodes), None, R)
            _, g = t.energy_and_gradient(wp_nodes(self.nodes, requires_grad=True))
            self.assertLess(
                np.linalg.norm(g.numpy() - R.numpy()) / np.linalg.norm(R.numpy()),
                1e-12,
                mode,
            )

    def test_parameter_adjoint_matches_finite_differences(self):
        t = self.term("reference")
        rng = np.random.default_rng(7)
        lam = rng.standard_normal(self.nodes.shape)
        g = t.residual_parameter_vjp(wp_nodes(self.nodes), wp_nodes(lam))["kc"]
        self.assertEqual(g.shape, (2,))
        # finite differences over kc of class 1 (the plate-to-plate class)
        h = 1e-6

        def lam_R(kc1):
            P = PotentialParams(kc=[1.0, kc1], p=4.0, r_in=0.2, r_out=0.5)
            tt = ContactPotentialTerm(self.pts, P, self.pair_class, self.nodes, self.n, device=DEVICE)
            tt.update_pairs(wp_nodes(self.nodes))
            R = wp.zeros(self.n, dtype=vec3d, device=DEVICE)
            tt.residual(wp_nodes(self.nodes), None, R)
            return np.sum(lam * R.numpy())

        fd = (lam_R(2.0 + h) - lam_R(2.0 - h)) / (2 * h)
        self.assertAlmostEqual(g[1] / fd, 1.0, places=8)

    def test_energy_is_linear_in_kc_through_the_tape(self):
        t = self.term("reference")
        x = wp_nodes(self.nodes, requires_grad=True)
        E, _ = t.energy_and_gradient(x)
        pot = t.potential
        tape = wp.Tape()
        f = wp.zeros(self.n, dtype=vec3d, device=DEVICE, requires_grad=True)
        with tape:
            E2 = pot.energy_force(x, f)
        tape.backward(E2)
        g_kc = pot.kc_gradient_from_tables()
        # E is linear in kc per class: sum_c kc_c dE/dkc_c = E (kc = [1, 2])
        self.assertAlmostEqual((g_kc[0] * 1.0 + g_kc[1] * 2.0) / float(E2.numpy()[0]), 1.0, places=10)
        tape.zero()


if __name__ == "__main__":
    unittest.main()
