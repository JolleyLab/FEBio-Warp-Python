"""Truss and discrete domains from a Model, and their coverage entries.

DRAFT (chordae elements, November 2026). Structure for discussion, not final.

FEBio counterpart
    FESolidDomainFactory picks "linear-truss" for a truss material on two-node
    line elements, and "discrete" for a discrete elastic material; the
    Discrete section pairs each <discrete> entry with a discrete set of the
    mesh and a discrete material by its position in the file.
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidDomainFactory.cpp
    https://github.com/febiosoftware/FEBio/blob/develop/FEBioXML/FEBioDiscreteSection.cpp

Both builders are called by analysis/steps.py. The coverage functions return
rows for the coverage report as (category, FEBio name, status, where, note).
"""

from __future__ import annotations

import numpy as np

from ..coverage import IMPLEMENTED, NO_EFFECT, NOT_SUPPORTED
from ..materials.discrete import resolve_discrete_material, resolve_truss_material
from .discrete_domain import DiscreteDomain
from .truss_domain import TrussDomain

TRUSS_ELEMENTS = ("line2", "truss2")
_TRUSS_SETTINGS = ("cross_sectional_area",)


def is_truss(block, dom) -> bool:
    """A part FEBio would run as a "linear-truss" domain."""
    return (
        dom.kind == "BeamDomain"
        and block.febio_type in TRUSS_ELEMENTS
        and dom.params.attrs.get("type") in (None, "linear-truss")
    )


def truss_coverage(model, block, dom) -> list:
    where = f"part {block.name}"
    rows = [("domain", f"linear-truss, {block.febio_type}", IMPLEMENTED, where, "")]
    for c in dom.params.children:
        ok = c.tag in _TRUSS_SETTINGS
        rows.append(("domain setting", c.tag, IMPLEMENTED if ok else NOT_SUPPORTED, where, ""))
    area = dom.params.get("cross_sectional_area", 0.0)
    if not float(area) > 0.0:
        # FELinearTrussDomain::Init refuses a domain without a positive area.
        rows.append(("domain setting", "cross_sectional_area", NOT_SUPPORTED, where, "must be positive"))
    try:
        material = model.material(dom.material)
    except KeyError:
        rows.append(("material", dom.material, NOT_SUPPORTED, where, "material not found"))
        return rows
    _, _, _, issues = resolve_truss_material(material)
    for i in issues:
        rows.append((i.category, i.febio_name, i.status, f"material {material.name}", i.note))
    if material.params.child("density") is not None:
        dynamic = any(str(s.control.get("analysis", "STATIC")).upper() == "DYNAMIC" for s in model.steps)
        rows.append(
            (
                "material parameter",
                "density",
                IMPLEMENTED if dynamic else NO_EFFECT,
                f"material {material.name}",
                "" if dynamic else "static analysis",
            )
        )
    return rows


def build_truss_domain(model, block, fields, device) -> TrussDomain:
    dom = model.domain_of(block.name)
    law, values, density, _ = resolve_truss_material(model.material(dom.material))
    conn = block.connectivity
    return TrussDomain(
        block.name,
        model.mesh.nodes[conn],
        fields.entries_of("displacement", conn),
        float(dom.params.get("cross_sectional_area")),
        [values],
        fields.count,
        device,
        model=law,
        poisson_ratio=values[1],
        density=density,
    )


def _discrete_entries(model):
    """(set name, node pairs, material record) for every <discrete> entry, in
    file order, or an issue string where an entry cannot be resolved."""
    section = next((s for s in model.other_sections if s.tag == "Discrete"), None)
    if section is None:
        return []
    materials = section.all("discrete_material")
    sets = {r.attrs.get("name"): r for r in model.mesh.other if r.tag == "DiscreteSet"}
    out = []
    for link in section.all("discrete"):
        name = link.attrs.get("discrete_set")
        # FEBio indexes discrete materials by their position in the file.
        index = int(link.attrs.get("dmat", 0))
        # FEBioDiscreteSection: type "spring" (or none) is a discrete domain;
        # type "wire" is a deformable spring domain, which is not built here.
        # Settings inside <discrete> are not read either.
        unsupported = link.attrs.get("type") not in (None, "spring") or bool(link.children)
        if name not in sets or not 1 <= index <= len(materials) or unsupported:
            out.append((name, None, None))
            continue
        ids = np.array([np.atleast_1d(c.value) for c in sets[name].all("delem")], dtype=np.int64).reshape(-1, 2)
        out.append((name, model.mesh.node_index(ids), materials[index - 1]))
    return out


def discrete_coverage(model) -> list:
    rows = []
    section = next((s for s in model.other_sections if s.tag == "Discrete"), None)
    if section is None:
        return rows
    for c in section.children:
        if c.tag not in ("discrete_material", "discrete"):
            rows.append(("Discrete", c.type or c.tag, NOT_SUPPORTED, c.name or "", ""))
    for name, pairs, material in _discrete_entries(model):
        if pairs is None:
            rows.append(
                (
                    "Discrete",
                    "discrete",
                    NOT_SUPPORTED,
                    name or "",
                    'discrete set or material not found, type other than "spring", or settings inside <discrete>',
                )
            )
            continue
        rows.append(("domain", "discrete", IMPLEMENTED, f"discrete set {name}", f"{len(pairs)} spring(s)"))
        _, _, issues = resolve_discrete_material(material)
        for i in issues:
            rows.append((i.category, i.febio_name, i.status, f"discrete material {material.name}", i.note))
    return rows


def build_discrete_domains(model, fields, device) -> list:
    """One DiscreteDomain per spring law in use, each holding every discrete
    set with that law. Sets keep their file order within a domain."""
    batches = {}
    for name, pairs, material in _discrete_entries(model):
        law, values, _ = resolve_discrete_material(material)
        batch = batches.setdefault(law, dict(pairs=[], region=[], params=[], names=[]))
        batch["region"].append(np.full(len(pairs), len(batch["names"]), np.int32))
        batch["pairs"].append(pairs)
        batch["params"].append(values)
        batch["names"].append(name)
    terms = []
    for law, b in batches.items():
        pairs = np.concatenate(b["pairs"])
        terms.append(
            DiscreteDomain(
                f"discrete ({law.febio_type})" if len(batches) == 1 else f"discrete ({law.name})",
                law,
                model.mesh.nodes[pairs],
                fields.entries_of("displacement", pairs),
                b["params"],
                fields.count,
                device,
                region=np.concatenate(b["region"]),
                set_names=b["names"],
            )
        )
    return terms
