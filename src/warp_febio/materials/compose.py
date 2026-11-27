"""Composition of material models when the kernel is built.

FEBio's "solid mixture" holds a list of child materials. Every child sees the
same deformation gradient, and the mixture returns the sum of their stresses,
tangents and energies (FEElasticMixture::Stress,
https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticMixture.cpp).
FEBio does this with virtual calls at run time. Warp kernels cannot dispatch at
run time, so the sum is built here once, as ordinary functions, and compiled
into a single kernel. The result is itself a MaterialModel, so mixtures nest.

Each child reads its own slice of the parameter vector and of the geometry
vector. Parameter and geometry names are prefixed with the child's label, the
way FEBio XML nests <solid> elements under the mixture.

Not carried over yet: FEBio's per-child mixture weights (they default to 1) and
children with history. Both fit this structure and are left for when a use case
needs them.
"""

from typing import Any

import warp as wp

from dataclasses import replace

from .model import GeometryField, MaterialModel, from_spatial


def _pair(label_a, a: MaterialModel, label_b, b: MaterialModel, name):
    """Sum of two models. `solid_mixture` folds a list through this."""
    na, nb = len(a.param_names), len(b.param_names)
    nab = na + nb
    ga = sum(f.size for f in a.geometry)
    gb = sum(f.size for f in b.geometry)
    gab = ga + gb
    ea, sa, ta = a.energy, a.cauchy, a.spatial_tangent_action
    eb, sb, tb = b.energy, b.cauchy, b.spatial_tangent_action

    # Slices use Python integers captured from this scope, which Warp treats as
    # compile-time constants.
    @wp.func
    def params_a(p: Any):
        return p[0:na]

    @wp.func
    def params_b(p: Any):
        return p[na:nab]

    # A child with no geometry ignores g, so it is handed the whole vector
    # (a zero-length slice is not a valid Warp type).
    if ga > 0:

        @wp.func
        def geometry_a(g: Any):
            return g[0:ga]

    else:

        @wp.func
        def geometry_a(g: Any):
            return g

    if gb > 0:

        @wp.func
        def geometry_b(g: Any):
            return g[ga:gab]

    else:

        @wp.func
        def geometry_b(g: Any):
            return g

    @wp.func
    def energy(F: Any, g: Any, p: Any):
        return ea(F, geometry_a(g), params_a(p)) + eb(F, geometry_b(g), params_b(p))

    @wp.func
    def cauchy(F: Any, g: Any, p: Any):
        return sa(F, geometry_a(g), params_a(p)) + sb(F, geometry_b(g), params_b(p))

    @wp.func
    def spatial_tangent_action(F: Any, d: Any, g: Any, p: Any):
        return ta(F, d, geometry_a(g), params_a(p)) + tb(F, d, geometry_b(g), params_b(p))

    # The one-pass stress and tangent of a mixture is the sum of its children's,
    # available when every child provides one.
    combined = None
    if a.stress_and_tangent is not None and b.stress_and_tangent is not None:
        ca, cb = a.stress_and_tangent, b.stress_and_tangent

        @wp.func
        def stress_and_tangent(F: Any, g: Any, p: Any):
            sigma_a, tangent_a = ca(F, geometry_a(g), params_a(p))
            sigma_b, tangent_b = cb(F, geometry_b(g), params_b(p))
            return sigma_a + sigma_b, tangent_a + tangent_b

        combined = stress_and_tangent

    def prefixed(label, names):
        return tuple(f"{label}.{n}" if label else n for n in names)

    return from_spatial(
        name,
        "solid mixture",
        prefixed(label_a, a.param_names) + prefixed(label_b, b.param_names),
        energy,
        cauchy,
        spatial_tangent_action,
        geometry=tuple(GeometryField(f"{label_a}.{f.name}" if label_a else f.name, f.size) for f in a.geometry)
        + tuple(GeometryField(f"{label_b}.{f.name}" if label_b else f.name, f.size) for f in b.geometry),
        # Each child's parameter ranges carry over under its prefixed name.
        param_ranges=tuple(replace(r, name=f"{label_a}.{r.name}" if label_a else r.name) for r in a.param_ranges)
        + tuple(replace(r, name=f"{label_b}.{r.name}" if label_b else r.name) for r in b.param_ranges),
        stress_and_tangent=combined,
        authored="composed",
    )


def solid_mixture(name, components):
    """FEBio-style solid mixture.

    components: list of (label, MaterialModel). Labels prefix the parameter and
    geometry names, for example "ground.E" and "fibers.mat_axis".
    """
    if len(components) < 2:
        raise ValueError("a solid mixture needs at least two components")
    if any(m.formulation != "coupled" for _, m in components):
        raise ValueError(
            "solid_mixture takes coupled models; uncoupled mixtures need their own "
            "operator so the deviatoric parts are summed before U(J) is added"
        )
    label, model = components[0]
    # Fold left. After the first pair the accumulated names are already
    # prefixed, so the accumulated side takes an empty label.
    for i, (next_label, next_model) in enumerate(components[1:]):
        last = i == len(components) - 2
        model = _pair(label, model, next_label, next_model, name if last else f"{name}__{i}")
        label = ""
    return model
