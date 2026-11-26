"""The central model: what a FEBio input file means, held on the host.

FEBio counterpart: FEModel and FEMesh
(https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEModel.h,
 https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEMesh.h).

Three objects carry a simulation (see docs/design/model.md):

    Model          host, NumPy only. Populated by the .feb reader or by code.
                   Frozen once built. This file.
    CompiledModel  device, Warp. Built once from a Model (compiled.py).
    State          device, Warp. One per time point (compiled.py).

What stays true to FEBio
------------------------
Section structure, type strings and parameter names are FEBio's own, as
written in a spec 4.0 input file. Node and element IDs from the file are kept
and written back, so results line up item for item with FEBio's.

What differs, for Warp
----------------------
The mesh is stored as flat arrays with zero-based indices in place of
per-element objects, and the solution state is held apart from the model
(as Newton's Model and State are), which is also what differentiation needs.

Everything that is not mesh data is kept as a `Record`: a small tree with
the tag, attributes, value and children exactly as in the file. A Record is
lossless, so a feature this code does not yet interpret is still carried,
reported by the coverage report, and written back unchanged.

This module imports only NumPy and the standard library, so it loads in
3D Slicer's Python and in tests without Warp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Nodes per element for the FEBio element type strings read so far.
# FEBio's list: FEModelBuilder::ElementSpec in
# https://github.com/febiosoftware/FEBio/blob/develop/FEBioXML/FEModelBuilder.cpp
ELEMENT_NODES = {
    "hex8": 8,
    "penta6": 6,
    "tet4": 4,
    "quad4": 4,
    "tri3": 3,
    "line2": 2,
}

# Facet type strings used inside <Surface>.
FACET_NODES = {"quad4": 4, "tri3": 3, "quad8": 8, "quad9": 9, "tri6": 6, "tri7": 7}


@dataclass
class Record:
    """One XML element of a FEBio input, kept as written.

    tag       the XML tag, for example "material", "bc", "E", "solver"
    attrs     its attributes, for example {"type": "neo-Hookean", "id": "1"}
    value     its text, parsed: None, bool is never inferred, int, float, str,
              or a 1-D float array for comma-separated numbers
    children  nested Records, in file order
    """

    tag: str
    attrs: dict = field(default_factory=dict)
    value: Any = None
    children: list = field(default_factory=list)
    # The text as it stood in the file. The writer emits it again while
    # `value` is unchanged, so formatting such as "0.10" or "007" survives.
    # Not part of equality.
    text: Optional[str] = None

    @property
    def type(self) -> Optional[str]:
        return self.attrs.get("type")

    @property
    def name(self) -> Optional[str]:
        return self.attrs.get("name")

    def child(self, tag: str) -> Optional["Record"]:
        for c in self.children:
            if c.tag == tag:
                return c
        return None

    def all(self, tag: str) -> list:
        return [c for c in self.children if c.tag == tag]

    def get(self, tag: str, default=None):
        """Value of the first child with this tag, or `default` if absent."""
        c = self.child(tag)
        return default if c is None or c.value is None else c.value

    def set(self, tag: str, value, **attrs) -> "Record":
        c = self.child(tag)
        if c is None:
            c = Record(tag)
            self.children.append(c)
        c.value = value
        c.text = None
        c.attrs.update({k: str(v) for k, v in attrs.items()})
        return c

    def equals(self, other: "Record") -> bool:
        """Deep equality, with arrays compared by value."""
        if (
            self.tag != other.tag
            or self.attrs != other.attrs
            or len(self.children) != len(other.children)
        ):
            return False
        a, b = self.value, other.value
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (
                isinstance(a, np.ndarray)
                and isinstance(b, np.ndarray)
                and a.shape == b.shape
                and np.array_equal(a, b)
            ):
                return False
        elif a != b:
            return False
        return all(x.equals(y) for x, y in zip(self.children, other.children))


@dataclass
class ElementBlock:
    """One <Elements> block: a part, with one element type.

    connectivity holds zero-based node indices (rows of Mesh.nodes), in FEBio's
    node order for the element type. ids holds the element IDs from the file.
    """

    name: str
    febio_type: str
    connectivity: np.ndarray  # (E, k) int32
    ids: np.ndarray  # (E,) int32
    # Attributes of <Elements> other than type and name, kept as written.
    attrs: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.ids)


@dataclass
class Surface:
    """One <Surface>: facets as zero-based node indices.

    Facets may mix types, so each facet has a type string and its own node
    list. `ids` are the facet IDs from the file.
    """

    name: str
    facet_types: list  # str per facet
    facets: list  # int32 array per facet
    ids: np.ndarray  # (F,) int32


@dataclass
class Mesh:
    """Nodes, element blocks and named sets. FEBio counterpart: FEMesh."""

    nodes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )  # (N, 3) float64
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # (N,)
    # The <Nodes name="..."> blocks, as (name, first index, count), so the
    # writer can reproduce them.
    node_blocks: list = field(default_factory=list)
    blocks: list = field(default_factory=list)  # ElementBlock
    node_sets: dict = field(default_factory=dict)  # name -> (n,) int32 node indices
    element_sets: dict = field(default_factory=dict)  # name -> (n,) int32 element IDs
    surfaces: dict = field(default_factory=dict)  # name -> Surface
    # Mesh items not interpreted yet (SurfacePair, DiscreteSet, Edge, ...),
    # carried as Records.
    other: list = field(default_factory=list)

    def block(self, name: str) -> ElementBlock:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"no element block named {name!r}")

    def node_index(self, node_ids) -> np.ndarray:
        """Zero-based indices for FEBio node IDs."""
        # The sorted order is cached; it is rebuilt if the ID array is replaced.
        cache = self.__dict__.get("_id_order")
        if cache is None or cache[0] is not self.node_ids:
            order = np.argsort(self.node_ids, kind="stable")
            cache = (self.node_ids, order, self.node_ids[order])
            self.__dict__["_id_order"] = cache
        _, order, sorted_ids = cache
        node_ids = np.asarray(node_ids)
        if len(order) == 0:
            if node_ids.size:
                raise KeyError("node ID not present in the mesh")
            return np.zeros(node_ids.shape, np.int32)
        pos = np.clip(np.searchsorted(sorted_ids, node_ids), 0, len(order) - 1)
        idx = order[pos]
        if not np.array_equal(self.node_ids[idx], node_ids):
            raise KeyError("node ID not present in the mesh")
        return idx.astype(np.int32)

    def node_set_indices(self, name: str) -> np.ndarray:
        """Node indices of a node set.

        FEBio also accepts "@surface:Name", "@edge:Name", "@elem_set:Name"
        and "@part_list:Name" where a node set is expected
        (FEModelBuilder / FEBioMeshSection). The first three are resolved here.
        """
        if name.startswith("@surface:"):
            s = self.surfaces[name[len("@surface:") :]]
            return np.unique(np.concatenate(s.facets)).astype(np.int32)
        if name.startswith("@elem_set:"):
            key = name[len("@elem_set:") :]
            for b in self.blocks:
                if b.name == key:
                    return np.unique(b.connectivity).astype(np.int32)
            raise KeyError(f"element set {key!r} as node set: only parts are resolved")
        if name.startswith("@edge:"):
            key = name[len("@edge:") :]
            for r in self.other:
                if r.tag == "Edge" and r.attrs.get("name") == key:
                    ids = np.concatenate([np.atleast_1d(c.value) for c in r.children])
                    return np.unique(self.node_index(ids)).astype(np.int32)
            raise KeyError(f"no edge named {key!r}")
        return self.node_sets[name]


@dataclass
class Material:
    """One <material>. `params` holds every child as written."""

    id: int
    name: str
    type: str
    params: Record

    def value(self, tag: str, default=None):
        return self.params.get(tag, default)


@dataclass
class Domain:
    """One entry of <MeshDomains>: assigns a material to a part.

    kind is the XML tag ("SolidDomain", "ShellDomain", "BeamDomain"), and
    `params` holds its attributes and children (shell_thickness, type, ...).
    """

    kind: str
    name: str  # the part (ElementBlock) name
    material: str  # material name
    params: Record


@dataclass
class LoadController:
    """One <load_controller>. For type "loadcurve", `points` is (n, 2).

    FEBio counterpart: FELoadCurve
    (https://github.com/febiosoftware/FEBio/blob/develop/FECore/FELoadCurve.h).
    """

    id: int
    type: str
    params: Record
    points: Optional[np.ndarray] = None

    def value(self, time: float) -> float:
        """Value at `time`, as PointCurve::value
        (https://github.com/febiosoftware/FEBio/blob/develop/FECore/PointCurve.cpp).

        Interpolation LINEAR, STEP and SMOOTH; extension CONSTANT. Other modes
        are reported by the coverage report as not yet supported.
        """
        if self.type != "loadcurve" or self.points is None:
            raise NotImplementedError(f"load controller type {self.type!r}")
        t, v = self.points[:, 0], self.points[:, 1]
        n = len(t)
        if n == 0:
            return 0.0
        if n == 1 or time <= t[0]:
            return float(v[0])  # CONSTANT extension below the first point
        if time >= t[-1]:
            return float(v[-1])  # CONSTANT extension above the last point
        interp = str(self.params.get("interpolate", "LINEAR")).upper()
        # FEBio's binarySearch: the first point whose time is greater than `time`.
        k = int(np.searchsorted(t, time, side="right"))
        if interp == "LINEAR":
            return _lerp(time, t[k - 1], v[k - 1], t[k], v[k])
        if interp == "STEP":
            return float(v[k])
        if interp == "SMOOTH":
            if n == 2:
                return _lerp(time, t[0], v[0], t[1], v[1])
            if n == 3 or k == 1:
                return _qerp(time, t[0:3], v[0:3])
            if k == n - 1:
                return _qerp(time, t[k - 2 : k + 1], v[k - 2 : k + 1])
            q1 = _qerp(time, t[k - 2 : k + 1], v[k - 2 : k + 1])
            q2 = _qerp(time, t[k - 1 : k + 2], v[k - 1 : k + 2])
            return _lerp(time, t[k - 1], q1, t[k], q2)
        raise NotImplementedError(f"load curve interpolation {interp!r}")


def _lerp(t, t0, f0, t1, f1):
    return float(f0 + (f1 - f0) * (t - t0) / (t1 - t0))


def _qerp(t, tt, ff):
    """Quadratic through three points (FEBio's qerp)."""
    t0, t1, t2 = tt
    f0, f1, f2 = ff
    return float(
        f0 * (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
        + f1 * (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
        + f2 * (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
    )


@dataclass
class Step:
    """One analysis step: its Control settings and the features it adds.

    A file with a top-level <Control> and no <Step> has one Step. Features
    defined at the top level of the file live on the Model and stay active in
    every step, as in FEBio.
    """

    id: int
    name: str
    control: Record
    boundary: list = field(default_factory=list)  # Records (tag "bc")
    loads: list = field(default_factory=list)
    contact: list = field(default_factory=list)
    other: list = field(default_factory=list)  # sections not interpreted yet
    # Attributes of <step> other than id and name, kept as written.
    attrs: dict = field(default_factory=dict)


@dataclass
class Issue:
    """Something the reader could not place. Never dropped silently."""

    where: str  # path in the file, for example "Mesh/Edge"
    message: str


@dataclass
class Model:
    """A FEBio model. FEBio counterpart: FEModel."""

    spec_version: str = "4.0"
    module: str = "solid"
    # The <Module> element as written (it may carry units), for the writer.
    module_record: Optional[Record] = None
    globals: Optional[Record] = None
    materials: list = field(default_factory=list)  # Material
    mesh: Mesh = field(default_factory=Mesh)
    domains: list = field(default_factory=list)  # Domain
    mesh_data: list = field(
        default_factory=list
    )  # Records (ElementData, NodeData, ...)
    boundary: list = field(default_factory=list)  # Records (tag "bc"), all steps
    loads: list = field(default_factory=list)  # Records, all steps
    contact: list = field(default_factory=list)  # Records, all steps
    load_controllers: list = field(default_factory=list)  # LoadController
    steps: list = field(default_factory=list)  # Step
    output: Optional[Record] = None
    # True when the file had a top-level <Control> (one implicit step) rather
    # than a <Step> section. Kept so the writer reproduces the layout.
    single_step_layout: bool = True
    # Top-level sections carried but not interpreted (Initial, Rigid,
    # Constraints, Discrete, ...), in file order.
    other_sections: list = field(default_factory=list)
    issues: list = field(default_factory=list)  # Issue

    # ---- queries used by compile and by the writers -------------------------

    def material(self, name: str) -> Material:
        for m in self.materials:
            if m.name == name:
                return m
        raise KeyError(f"no material named {name!r}")

    def domain_of(self, block_name: str) -> Domain:
        for d in self.domains:
            if d.name == block_name:
                return d
        raise KeyError(f"part {block_name!r} has no entry in MeshDomains")

    def load_controller(self, lc_id: int) -> LoadController:
        for lc in self.load_controllers:
            if lc.id == lc_id:
                return lc
        raise KeyError(f"no load controller with id {lc_id}")

    def plot_variables(self) -> list:
        """Plot variable names the file requests, in order.

        With no <plotfile>, FEBio writes displacement and stress
        (FEBioPlotFile::Dictionary::Defaults).
        """
        names = []
        if self.output is not None:
            for pf in self.output.all("plotfile"):
                names += [v.attrs.get("type") for v in pf.all("var")]
        return names or ["displacement", "stress"]

    def element_data(self, data_type: str, block_name: str) -> Optional[Record]:
        for r in self.mesh_data:
            if (
                r.tag == "ElementData"
                and r.attrs.get("type") == data_type
                and r.attrs.get("elem_set") == block_name
            ):
                return r
        return None

    def equals(self, other: "Model") -> bool:
        """Deep equality of everything the reader stores. Used by round-trip tests.

        Carried top-level sections are compared tag by tag, in file order
        within a tag. Their order relative to each other is not compared,
        because the writer places sections where FEBio Studio writes them
        and not where the input had them.
        """
        a, b = self.mesh, other.mesh
        if not (
            np.array_equal(a.nodes, b.nodes)
            and np.array_equal(a.node_ids, b.node_ids)
            and a.node_blocks == b.node_blocks
            and len(a.blocks) == len(b.blocks)
            and a.node_sets.keys() == b.node_sets.keys()
            and a.element_sets.keys() == b.element_sets.keys()
            and a.surfaces.keys() == b.surfaces.keys()
        ):
            return False
        for x, y in zip(a.blocks, b.blocks):
            if not (
                x.name == y.name
                and x.febio_type == y.febio_type
                and x.attrs == y.attrs
                and np.array_equal(x.connectivity, y.connectivity)
                and np.array_equal(x.ids, y.ids)
            ):
                return False
        for k in a.node_sets:
            if not np.array_equal(a.node_sets[k], b.node_sets[k]):
                return False
        for k in a.element_sets:
            if not np.array_equal(a.element_sets[k], b.element_sets[k]):
                return False
        for k in a.surfaces:
            s, t = a.surfaces[k], b.surfaces[k]
            if s.facet_types != t.facet_types or not all(
                np.array_equal(p, q) for p, q in zip(s.facets, t.facets)
            ):
                return False

        def recs(p, q):
            return len(p) == len(q) and all(x.equals(y) for x, y in zip(p, q))

        def by_tag(p, q):
            tags = [r.tag for r in p]
            return sorted(tags) == sorted(r.tag for r in q) and all(
                recs([r for r in p if r.tag == t], [r for r in q if r.tag == t])
                for t in set(tags)
            )

        def opt(p, q):
            return (p is None and q is None) or (
                p is not None and q is not None and p.equals(q)
            )

        if not (
            self.spec_version == other.spec_version
            and self.module == other.module
            and (
                self.module_record is None
                or other.module_record is None
                or self.module_record.equals(other.module_record)
            )
            and opt(self.globals, other.globals)
            and opt(self.output, other.output)
            and recs(a.other, b.other)
            and recs(self.mesh_data, other.mesh_data)
            and recs(self.boundary, other.boundary)
            and recs(self.loads, other.loads)
            and recs(self.contact, other.contact)
            and by_tag(self.other_sections, other.other_sections)
            and len(self.materials) == len(other.materials)
            and len(self.domains) == len(other.domains)
            and len(self.load_controllers) == len(other.load_controllers)
            and len(self.steps) == len(other.steps)
        ):
            return False
        for x, y in zip(self.materials, other.materials):
            if (x.id, x.name, x.type) != (y.id, y.name, y.type) or not x.params.equals(
                y.params
            ):
                return False
        for x, y in zip(self.domains, other.domains):
            if (x.kind, x.name, x.material) != (
                y.kind,
                y.name,
                y.material,
            ) or not x.params.equals(y.params):
                return False
        for x, y in zip(self.load_controllers, other.load_controllers):
            if (x.id, x.type) != (y.id, y.type) or not x.params.equals(y.params):
                return False
        for x, y in zip(self.steps, other.steps):
            if not (
                (x.id, x.name, x.attrs) == (y.id, y.name, y.attrs)
                and x.control.equals(y.control)
                and recs(x.boundary, y.boundary)
                and recs(x.loads, y.loads)
                and recs(x.contact, y.contact)
                and recs(x.other, y.other)
            ):
                return False
        return True
