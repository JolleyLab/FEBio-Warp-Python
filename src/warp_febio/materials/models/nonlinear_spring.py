"""FEBio's nonlinear spring, as used for chordae in the valve models.

FEBio material type : "nonlinear spring"  (a discrete material)
FEBio source        : https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENonlinearSpring.cpp
Point function      : https://github.com/febiosoftware/FEBio/blob/develop/FECore/PointCurve.cpp

The force along the spring is

    F = scale * f(s)

where f is a function given in the input file and s is a deformation measure
chosen by the `measure` parameter:

    elongation  s = L - L0          (default)
    strain      s = (L - L0) / L0
    stretch     s = L / L0

In the published valve models (Wu et al. 2022; Aslan et al., arXiv 2509.22865)
the chordae are these springs with a piecewise-linear f that is zero up to a
slack elongation and rises linearly beyond it, so the chord carries tension
only. The repository's mitral input defines eight such materials, for example

    <scale>60</scale> <measure>elongation</measure>
    <force type="point"> <interpolate>linear</interpolate> <extend>extrapolate</extend>
        <points> <pt>0,0</pt> <pt>4,0</pt> <pt>5,1</pt> </points> </force>

What is implemented
-------------------
- f as a point function with linear interpolation, and extension "constant" or
  "extrapolate". FEBio's other interpolation and extension modes, and
  `type="math"` functions, are not implemented and raise.
- The three measures.
- The spring force vector and 3x3 stiffness block of
  FENonlinearSpringMaterial::Force and ::Stiffness.

Layout
------
The number of points, the measure and the extension are fixed when the model is
built, because they select code. The scale and the point coordinates are float
parameters:

    p = (scale, x0, y0, x1, y1, ...)

so materials with the same number of points share one kernel and differ by
their row of the parameter array (the mitral input needs one kernel for its
eight chord groups), and gradients with respect to the scale and to the slack
length are available from the tape.

Two differences from FEBio, both deliberate
-------------------------------------------
- The slope df/ds is the exact slope of the active segment. FEBio evaluates it
  by a central difference of the point curve (PointCurve::derive), which agrees
  to about 1e-7 away from a knot and returns the mean of the two slopes within
  1e-9 of the curve's span of a knot.
- At a knot the segment to the right is used, as FEBio's value() does.

FEBio does not implement a strain energy for this material
(FENonlinearSpringMaterial::StrainEnergy asserts), and none is provided here.
"""

from functools import cache
from typing import Any

import warp as wp

from ..uniaxial import UniaxialModel

MEASURES = ("elongation", "strain", "stretch")
EXTENSIONS = ("constant", "extrapolate")

SOURCE = "FEBioMech/FENonlinearSpring.cpp, FECore/PointCurve.cpp (FEBio develop)"


def make_point_function(n_points: int, extend: str = "extrapolate", offset: int = 0):
    """f(s) and df/ds for a piecewise-linear function through `n_points` points.

    The points are read from the parameter vector as p[offset + 2k] (abscissa)
    and p[offset + 2k + 1] (ordinate), k = 0 .. n_points - 1, with abscissae
    increasing. Mirrors PointCurve::value for LINEAR interpolation with the
    CONSTANT or EXTRAPOLATE extension.
    """
    if n_points < 2:
        raise ValueError("a point function needs at least two points")
    if extend not in EXTENSIONS:
        raise ValueError(f"extend must be one of {EXTENSIONS}, got {extend!r}")
    last = offset + 2 * (n_points - 1)
    constant = extend == "constant"

    @wp.func
    def value_and_slope(s: Any, p: Any):
        zero = s.dtype(0.0)
        # Active segment: the last one whose left end is not beyond s, or the
        # first segment when s lies before the first point. Chosen with
        # wp.where so the function has no data-dependent branch.
        x0 = p[offset]
        y0 = p[offset + 1]
        x1 = p[offset + 2]
        y1 = p[offset + 3]
        for k in range(2, n_points):
            xa = p[offset + 2 * (k - 1)]
            use = s >= xa
            x0 = wp.where(use, xa, x0)
            y0 = wp.where(use, p[offset + 2 * (k - 1) + 1], y0)
            x1 = wp.where(use, p[offset + 2 * k], x1)
            y1 = wp.where(use, p[offset + 2 * k + 1], y1)
        slope = (y1 - y0) / (x1 - x0)
        value = y0 + slope * (s - x0)
        if wp.static(constant):
            below = s < p[offset]
            above = s > p[last]
            value = wp.where(below, p[offset + 1], wp.where(above, p[last + 1], value))
            slope = wp.where(below or above, zero, slope)
        return value, slope

    return value_and_slope


@cache
def nonlinear_spring(n_points: int, measure: str = "elongation", extend: str = "extrapolate") -> UniaxialModel:
    """The FEBio nonlinear spring with a linear point function of `n_points` points.

    As a uniaxial model its "strain" is FEBio's deformation measure s and its
    "stress" is the spring force:

        update(s, p, committed) -> (F, dF/ds, committed)

    It has no history; the state vector has one unused entry.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
    f = make_point_function(n_points, extend, offset=1)

    @wp.func
    def update(s: Any, p: Any, committed: Any):
        value, slope = f(s, p)
        return p[0] * value, p[0] * slope, committed

    names = ["scale"]
    for k in range(n_points):
        names += [f"x{k}", f"y{k}"]
    return UniaxialModel(
        name=f"nonlinear_spring_{measure}_{extend}_{n_points}",
        param_names=tuple(names),
        state_names=("unused",),
        update=update,
        initial_state=(0.0,),
        source=SOURCE,
        febio_type="nonlinear spring",
        options=(("measure", measure), ("extend", extend), ("n_points", n_points)),
    )


def make_spring_function(model: UniaxialModel):
    """Force vector and 3x3 stiffness of one spring, as FEBio's discrete domain uses them.

        force_and_stiffness(r0, rt, p) -> (force along the spring, stiffness block, s)

    r0 and rt are the vectors from the first to the second node in the reference
    and current configurations. Mirrors FENonlinearSpringMaterial::Force and
    ::Stiffness, including the guard for a spring of zero current length.
    Assembly into nodal forces is the element's job.
    """
    update = model.update
    measure = model.option("measure", "elongation")
    is_strain = measure == "strain"
    is_stretch = measure == "stretch"

    @wp.func
    def force_and_stiffness(r0: Any, rt: Any, p: Any):
        zero = r0.dtype(0.0)
        one = r0.dtype(1.0)
        L0 = wp.length(r0)
        Lt = wp.length(rt)
        s = Lt - L0
        ds = one
        if wp.static(is_strain):
            s = (Lt - L0) / L0
            ds = one / L0
        if wp.static(is_stretch):
            s = Lt / L0
            ds = one / L0
        F, dF, unused = update(s, p, wp.vector(zero, length=1))
        E = dF * ds
        collapsed = Lt == zero
        length = wp.where(collapsed, one, Lt)
        e = wp.where(collapsed, wp.vector(one, one, one), rt / length)
        F = wp.where(collapsed, zero, F)
        ratio = F / length
        K = (E - ratio) * wp.outer(e, e) + ratio * wp.identity(n=3, dtype=r0.dtype)
        return F * e, K, s

    return force_and_stiffness


_KERNELS = {}


def make_spring_kernel(model: UniaxialModel, dtype=wp.float64):
    """Kernel that evaluates a batch of springs from nodal positions."""
    key = (model, dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    pvec = model.param_vector(dtype)
    vec3 = wp.types.vector(length=3, dtype=dtype)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=dtype)
    fn = make_spring_function(model)

    def springs(
        X: wp.array(dtype=vec3),
        x: wp.array(dtype=vec3),
        nodes: wp.array2d(dtype=int),
        region: wp.array(dtype=int),
        params: wp.array(dtype=pvec),
        force: wp.array(dtype=vec3),
        stiffness: wp.array(dtype=mat33),
        measure: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        a = nodes[i, 0]
        b = nodes[i, 1]
        f, K, s = fn(X[b] - X[a], x[b] - x[a], params[region[i]])
        force[i] = f
        stiffness[i] = K
        measure[i] = s

    kernel = wp.kernel(springs, module="unique")
    _KERNELS[key] = kernel
    return kernel
