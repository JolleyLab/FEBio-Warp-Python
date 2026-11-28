"""From a FEBio <material> in the central model to a material of this layer.

This is the one place where the model (warp_febio.model, FEBio names and
records as read from a .feb file) meets the material layer (model.py,
compose.py). Compile, the coverage report and the verification tests all go
through `bind_material`, so a FEBio material type is interpreted in one way.

FEBio counterpart: the material factory and FEMaterial::Init, which build a
material object from the input, validate its parameters and evaluate its
material axes
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEMaterial.cpp,
 https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticMixture.cpp).

What is interpreted
    type                the registered model with that FEBio type string
    "solid mixture"     its <solid> children, composed with compose.solid_mixture;
                        children are labelled solid[0], solid[1], ... in file
                        order, so a parameter is named "solid[0].E"
    float parameters    by FEBio tag; range-checked as FEBio does at Init
    lc="id"             the parameter follows that load controller of the model
                        (FEBio: the curve value multiplies the parameter)
    density             kept on the material; no constitutive kernel reads it
    mat_axis            type "vector" (a, d), built as FEMat3dVectorMap does
    options             int / bool / enum parameters such as pressure_model must
                        match the option the model was built with

Anything else is returned as an issue under its FEBio name and is not
dropped, in keeping with the coverage report: a material with an issue of
status "not yet supported" does not run.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from . import model as material_model
from . import models  # noqa: F401  (registers the models)
from .compose import solid_mixture
from .geometry import axes_from_vectors
from .voigt import VI, VJ, zeros66

NOT_SUPPORTED = "not yet supported"
NO_EFFECT = "no effect here"
IMPLEMENTED = "implemented"

# Tags a FEBio material may carry that are not constitutive parameters.
_STRUCTURAL = {"density", "mat_axis", "solid"}


@dataclass
class MaterialIssue:
    """One finding about a material, for the coverage report."""

    category: str  # "material" or "material parameter"
    febio_name: str
    status: str
    note: str = ""


@dataclass
class BoundMaterial:
    """A FEBio material resolved against the material layer.

    `material` is the layer's Material (model, parameter values, density and
    load curves). `geometry` holds the fields the model declares, as read from
    the input (for example {"mat_axis": 3x3}).
    """

    name: str
    febio_type: str
    material: material_model.Material
    geometry: dict = field(default_factory=dict)

    @property
    def model(self) -> material_model.MaterialModel:
        return self.material.model

    @property
    def density(self) -> float:
        return self.material.density

    @property
    def time_dependent(self) -> bool:
        return bool(self.material.schedules)

    def param_values(self, time: float = 0.0) -> list:
        """Parameter values at `time`, in the order of the model's vector."""
        values = self.material.params_at(time)
        return [float(values[k]) for k in self.model.param_names]

    def geometry_array(self, count: int) -> np.ndarray:
        """(count, geometry_size) geometry rows, one per element or point."""
        return self.model.pack_geometry(count, **{k: np.asarray(v).ravel() for k, v in self.geometry.items()})

    def stress_and_tangent(self) -> Callable:
        """Cauchy stress and 6x6 spatial tangent in one call, for an element."""
        return one_pass(self.model)


_ONE_PASS = {}


def one_pass(model: material_model.MaterialModel) -> Callable:
    """`stress_and_tangent(F, g, p)` for any solid model.

    The model's own one-pass function where it has one. Otherwise the 6x6 is
    built from six tangent actions, as the exchange kernel does; the
    conformance checks compare the two routes for models that have both.
    """
    if model.stress_and_tangent is not None:
        return model.stress_and_tangent
    if model.name in _ONE_PASS:
        return _ONE_PASS[model.name]
    cauchy, action = model.cauchy, model.spatial_tangent_action

    def stress_and_tangent(F: Any, g: Any, p: Any):
        sigma = cauchy(F, g, p)
        half = F.dtype(0.5)
        c = zeros66(F)
        # Shear columns use (e_k e_l + e_l e_k) / 2, so the entry is c_ijkl
        # itself (FEBio's tens4ds convention), not engineering shear.
        for b in range(6):
            d = wp.matrix(shape=(3, 3), dtype=F.dtype)
            d[VI[b], VJ[b]] = d[VI[b], VJ[b]] + half
            d[VJ[b], VI[b]] = d[VJ[b], VI[b]] + half
            cd = action(F, d, g, p)
            for a in range(6):
                c[a, b] = cd[VI[a], VJ[a]]
        return sigma, c

    func = wp.func(stress_and_tangent, name=f"one_pass_{model.name}")
    _ONE_PASS[model.name] = func
    return func


def _resolve(record, issues, path=""):
    """Model, values, load-curve ids, geometry and density of one material record.

    `record` is a model Record with attrs["type"] and the parameters as
    children. Returns None for the model when the type cannot be resolved.
    """
    febio_type = record.attrs.get("type")
    where = f"{path}{febio_type}"
    values, curves, geometry, density = {}, {}, {}, 0.0

    if febio_type == "solid mixture":
        solids = record.all("solid")
        parts = []
        for i, child in enumerate(solids):
            label = f"solid[{i}]"
            sub = _resolve(child, issues, path=f"{label}.")
            if sub[0] is None:
                return None, values, curves, geometry, density
            cm, cv, cc, cg, _ = sub
            parts.append((label, cm))
            values.update({f"{label}.{k}": v for k, v in cv.items()})
            curves.update({f"{label}.{k}": v for k, v in cc.items()})
            geometry.update({f"{label}.{k}": v for k, v in cg.items()})
        for c in record.children:
            if c.tag == "density":
                density = float(c.value)
            elif c.tag != "solid":
                # A mixture's own mat_axis composes with its children's in
                # FEBio (FEElasticMixture); that is not done here yet.
                issues.append(MaterialIssue("material parameter", f"{where}.{c.tag}", NOT_SUPPORTED))
        if len(parts) < 2:
            issues.append(
                MaterialIssue(
                    "material",
                    where,
                    NOT_SUPPORTED,
                    "a solid mixture needs two or more solids",
                )
            )
            return None, values, curves, geometry, density
        try:
            model = solid_mixture("mixture(" + ", ".join(m.name for _, m in parts) + ")", parts)
        except ValueError as err:
            issues.append(MaterialIssue("material", where, NOT_SUPPORTED, str(err)))
            return None, values, curves, geometry, density
        return model, values, curves, geometry, density

    try:
        model = material_model.by_febio_type(febio_type)
    except KeyError:
        issues.append(MaterialIssue("material", where, NOT_SUPPORTED))
        return None, values, curves, geometry, density

    options = dict(model.options)
    declares_axes = any(f.name == "mat_axis" for f in model.geometry)
    for c in record.children:
        tag = c.tag
        if tag in model.param_names:
            values[tag] = float(c.value)
            if "lc" in c.attrs:
                curves[tag] = int(c.attrs["lc"])
        elif tag == "density":
            density = float(c.value)
        elif tag == "mat_axis":
            kind = c.attrs.get("type")
            if kind != "vector" or c.child("a") is None or c.child("d") is None:
                issues.append(
                    MaterialIssue(
                        "material parameter",
                        f"{where}.mat_axis type {kind!r}",
                        NOT_SUPPORTED,
                    )
                )
            elif declares_axes:
                geometry["mat_axis"] = axes_from_vectors(
                    np.asarray(c.child("a").value, dtype=float),
                    np.asarray(c.child("d").value, dtype=float),
                )
            else:
                issues.append(
                    MaterialIssue(
                        "material parameter",
                        f"{where}.mat_axis",
                        NO_EFFECT,
                        "isotropic model",
                    )
                )
        elif tag in options:
            if str(c.value).strip() != str(options[tag]):
                issues.append(
                    MaterialIssue(
                        "material parameter",
                        f"{where}.{tag}",
                        NOT_SUPPORTED,
                        f"{c.value!r}; this model is built with {options[tag]!r}",
                    )
                )
        else:
            issues.append(MaterialIssue("material parameter", f"{where}.{tag}", NOT_SUPPORTED))
    for name in model.param_names:
        if name not in values:
            issues.append(
                MaterialIssue(
                    "material parameter",
                    f"{where}.{name}",
                    NOT_SUPPORTED,
                    "missing; FEBio defaults are not applied yet",
                )
            )
    return model, values, curves, geometry, density


def bind_material(material, model=None):
    """Resolve one material of the central model.

    material   a warp_febio.model Material (its `params` record is read), or a
               Record with a `type` attribute
    model      the central Model, needed only when a parameter has an `lc`
               attribute, to find the load controller

    Returns (BoundMaterial or None, issues). The bound material is None when
    the type, a parameter or a range check stops it from being evaluated; the
    issues say why, by FEBio name.
    """
    record = getattr(material, "params", material)
    name = getattr(material, "name", None) or record.attrs.get("name", "")
    febio_type = record.attrs.get("type")
    issues: list = []
    resolved, values, curves, geometry, density = _resolve(record, issues)
    if resolved is None:
        return None, issues
    issues.insert(0, MaterialIssue("material", febio_type, IMPLEMENTED))

    schedules = {}
    for tag, lc_id in curves.items():
        controller = None
        if model is not None:
            try:
                controller = model.load_controller(lc_id)
            except KeyError:
                controller = None
        if controller is None or controller.type != "loadcurve":
            issues.append(
                MaterialIssue(
                    "material parameter",
                    f"{febio_type}.{tag}",
                    NOT_SUPPORTED,
                    f"load controller {lc_id} not found or not a load curve",
                )
            )
        else:
            # The model's own evaluation of FEBio's PointCurve; there is one
            # implementation of the curve in the run path.
            schedules[tag] = controller.value

    if any(i.status == NOT_SUPPORTED for i in issues):
        return None, issues
    try:
        resolved.validate(values)
    except ValueError as err:
        issues.append(MaterialIssue("material parameter", febio_type, NOT_SUPPORTED, str(err)))
        return None, issues
    if resolved.formulation != "coupled":
        issues.append(
            MaterialIssue(
                "material",
                febio_type,
                NOT_SUPPORTED,
                "uncoupled; needs a multi-field element",
            )
        )
    bound = BoundMaterial(
        name=name,
        febio_type=febio_type,
        material=material_model.Material(model=resolved, params=values, density=density, schedules=schedules),
        geometry=geometry,
    )
    return bound, issues


def runnable(issues) -> bool:
    return not any(i.status == NOT_SUPPORTED for i in issues)
