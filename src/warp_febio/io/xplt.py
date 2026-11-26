"""Read and write FEBio plot files (.xplt), as FEBio writes them today.

FEBio counterpart: FEBioPlotFile and PltArchive, plot version 0x0035
(https://github.com/febiosoftware/FEBio/blob/develop/FEBioPlot/FEBioPlotFile.cpp,
 https://github.com/febiosoftware/FEBio/blob/develop/FEBioPlot/PltArchive.h).
There is no written specification; this follows the writer's source. The tag
values below are copied from FEBioPlotFile.h.

File layout
    uint32 0x00464542                         file signature
    chunk  PLT_ROOT   { header, dictionary }
    chunk  PLT_MESH   { nodes, domains, surfaces, node sets, element sets,
                        facet sets, parts }
    chunk  PLT_STATE  { header, mesh state, data }      one per output state
A chunk is  uint32 id, uint32 size, payload.  A branch chunk's payload is more
chunks; a leaf's payload is data. All values are little-endian, and results
are single precision, as in FEBio.

Uses only the standard library and NumPy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..model import ELEMENT_NODES, Model
from ..results import Results

SIGNATURE = 0x00464542
PLT_VERSION = 0x0035
STR_SIZE = 64

# --- tags (FEBioPlotFile.h) --------------------------------------------------
PLT_ROOT = 0x01000000
PLT_HEADER = 0x01010000
PLT_HDR_VERSION = 0x01010001
PLT_HDR_COMPRESSION = 0x01010004
PLT_HDR_AUTHOR = 0x01010005
PLT_HDR_SOFTWARE = 0x01010006
PLT_HDR_UNITS = 0x01010007
PLT_DICTIONARY = 0x01020000
PLT_DIC_ITEM = 0x01020001
PLT_DIC_ITEM_TYPE = 0x01020002
PLT_DIC_ITEM_FMT = 0x01020003
PLT_DIC_ITEM_NAME = 0x01020004
PLT_DIC_ITEM_ARRAYSIZE = 0x01020005
PLT_DIC_ITEM_ARRAYNAME = 0x01020006
PLT_DIC_ITEM_UNITS = 0x01020007
PLT_DIC_GLOBAL = 0x01021000
PLT_DIC_NODAL = 0x01023000
PLT_DIC_DOMAIN = 0x01024000
PLT_DIC_SURFACE = 0x01025000
PLT_DIC_EDGE = 0x01026000
PLT_MESH = 0x01040000
PLT_NODE_SECTION = 0x01041000
PLT_NODE_HEADER = 0x01041100
PLT_NODE_SIZE = 0x01041101
PLT_NODE_DIM = 0x01041102
PLT_NODE_NAME = 0x01041103
PLT_NODE_COORDS = 0x01041200
PLT_DOMAIN_SECTION = 0x01042000
PLT_DOMAIN = 0x01042100
PLT_DOMAIN_HDR = 0x01042101
PLT_DOM_ELEM_TYPE = 0x01042102
PLT_DOM_PART_ID = 0x01042103
PLT_DOM_ELEMS = 0x01032104
PLT_DOM_NAME = 0x01032105
PLT_DOM_ELEM_LIST = 0x01042200
PLT_ELEMENT = 0x01042201
PLT_SURFACE_SECTION = 0x01043000
PLT_SURFACE = 0x01043100
PLT_SURFACE_HDR = 0x01043101
PLT_SURFACE_ID = 0x01043102
PLT_SURFACE_FACES = 0x01043103
PLT_SURFACE_NAME = 0x01043104
PLT_SURFACE_MAX_FACET_NODES = 0x01043105
PLT_FACE_LIST = 0x01043200
PLT_FACE = 0x01043201
PLT_NODESET_SECTION = 0x01044000
PLT_NODESET = 0x01044100
PLT_NODESET_HDR = 0x01044101
PLT_NODESET_ID = 0x01044102
PLT_NODESET_NAME = 0x01044103
PLT_NODESET_SIZE = 0x01044104
PLT_NODESET_LIST = 0x01044200
PLT_PARTS_SECTION = 0x01045000
PLT_PART = 0x01045100
PLT_PART_ID = 0x01045101
PLT_PART_NAME = 0x01045102
PLT_ELEMENTSET_SECTION = 0x01046000
PLT_ELEMENTSET = 0x01046100
PLT_ELEMENTSET_HDR = 0x01046101
PLT_ELEMENTSET_ID = 0x01046102
PLT_ELEMENTSET_NAME = 0x01046103
PLT_ELEMENTSET_SIZE = 0x01046104
PLT_ELEMENTSET_LIST = 0x01046200
PLT_FACETSET_SECTION = 0x01047000
PLT_FACETSET = 0x01047100
PLT_FACETSET_HDR = 0x01047101
PLT_FACETSET_ID = 0x01047102
PLT_FACETSET_NAME = 0x01047103
PLT_FACETSET_SIZE = 0x01047104
PLT_FACETSET_MAXNODES = 0x01047105
PLT_FACETSET_LIST = 0x01047200
PLT_FACET = 0x01047201
PLT_EDGE_SECTION = 0x01048000
PLT_OBJECTS_SECTION = 0x01050000
PLT_STATE = 0x02000000
PLT_STATE_HEADER = 0x02010000
PLT_STATE_HDR_TIME = 0x02010002
PLT_STATE_STATUS = 0x02010003
PLT_STATE_DATA = 0x02020000
PLT_STATE_VARIABLE = 0x02020001
PLT_STATE_VAR_ID = 0x02020002
PLT_STATE_VAR_DATA = 0x02020003
PLT_GLOBAL_DATA = 0x02020100
PLT_NODE_DATA = 0x02020300
PLT_ELEMENT_DATA = 0x02020400
PLT_FACE_DATA = 0x02020500
PLT_EDGE_DATA = 0x02020600
PLT_MESH_STATE = 0x02030000
PLT_ELEMENT_STATE = 0x02030001
PLT_OBJECTS_STATE = 0x02040000

_BRANCHES = {
    PLT_ROOT, PLT_HEADER, PLT_DICTIONARY, PLT_DIC_ITEM, PLT_DIC_GLOBAL, PLT_DIC_NODAL,
    PLT_DIC_DOMAIN, PLT_DIC_SURFACE, PLT_DIC_EDGE, PLT_MESH, PLT_NODE_SECTION,
    PLT_NODE_HEADER, PLT_DOMAIN_SECTION, PLT_DOMAIN, PLT_DOMAIN_HDR, PLT_DOM_ELEM_LIST,
    PLT_SURFACE_SECTION, PLT_SURFACE, PLT_SURFACE_HDR, PLT_FACE_LIST, PLT_NODESET_SECTION,
    PLT_NODESET, PLT_NODESET_HDR, PLT_PARTS_SECTION, PLT_PART, PLT_ELEMENTSET_SECTION,
    PLT_ELEMENTSET, PLT_ELEMENTSET_HDR, PLT_FACETSET_SECTION, PLT_FACETSET,
    PLT_FACETSET_HDR, PLT_FACETSET_LIST, PLT_STATE, PLT_STATE_HEADER, PLT_STATE_DATA,
    PLT_STATE_VARIABLE, PLT_GLOBAL_DATA, PLT_NODE_DATA, PLT_ELEMENT_DATA, PLT_FACE_DATA,
    PLT_EDGE_DATA, PLT_MESH_STATE,
}  # fmt: skip

# Var_Type and Storage_Fmt (FECore/fecore_enum.h).
PLT_FLOAT, PLT_VEC3F, PLT_MAT3FS, PLT_MAT3FD, PLT_TENS4FS, PLT_MAT3F = range(6)
VAR_SIZE = {
    PLT_FLOAT: 1,
    PLT_VEC3F: 3,
    PLT_MAT3FS: 6,
    PLT_MAT3FD: 3,
    PLT_TENS4FS: 21,
    PLT_MAT3F: 9,
}
FMT_NODE, FMT_ITEM, FMT_MULT, FMT_REGION = range(4)

# Elem_Type (FEBioPlotFile.h) for the FEBio element type strings.
PLT_ELEM = {"hex8": 0, "penta6": 1, "tet4": 2, "quad4": 3, "tri3": 4, "line2": 5}
PLT_ELEM_NODES = {
    0: 8,
    1: 6,
    2: 4,
    3: 4,
    4: 3,
    5: 2,
    6: 20,
    7: 10,
    8: 15,
    9: 27,
    10: 6,
    11: 8,
    12: 9,
}
# FEFacetSet::FACET::MAX_NODES (FECore/FEFacetSet.h).
FACET_MAX_NODES = 10

# Plot variables this writer knows: name -> (dictionary category, type, format).
KNOWN_VARIABLES = {
    "displacement": ("nodal", PLT_VEC3F, FMT_NODE),
    "stress": ("domain", PLT_MAT3FS, FMT_ITEM),
}


# ---------------------------------------------------------------------------
# data model of a plot file


@dataclass
class DictionaryItem:
    name: str
    type: int
    fmt: int
    array_size: int = 0
    units: Optional[str] = None


@dataclass
class PlotDomain:
    elem_type: int
    part_id: int
    name: Optional[str]
    elem_ids: np.ndarray  # (E,)
    connectivity: np.ndarray  # (E, k) zero-based node indices


@dataclass
class PlotSurface:
    id: int
    name: str
    max_nodes: int
    faces: np.ndarray  # (F, max_nodes + 2): face id, node count, nodes


@dataclass
class PlotState:
    time: float
    status: int = 0
    element_state: Optional[np.ndarray] = None
    # category -> list (one per dictionary item) of {item id: float32 array}
    data: dict = field(default_factory=dict)


@dataclass
class PlotFile:
    version: int = PLT_VERSION
    compression: int = 0
    software: Optional[str] = None
    units: Optional[str] = None
    dictionary: dict = field(default_factory=dict)  # category -> [DictionaryItem]
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    domains: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)
    node_sets: list = field(default_factory=list)  # (id, name, indices)
    element_sets: list = field(default_factory=list)  # (id, name, items)
    facet_sets: list = field(default_factory=list)  # PlotSurface
    parts: list = field(default_factory=list)  # (id, name)
    states: list = field(default_factory=list)

    @property
    def times(self):
        return np.array([s.time for s in self.states])

    def variable(self, name: str):
        """All states of one variable.

        Nodal variables: array (states, nodes, components).
        Domain variables: list over domains of arrays (states, items, components).
        """
        for category, items in self.dictionary.items():
            for k, item in enumerate(items):
                if item.name != name:
                    continue
                size = VAR_SIZE[item.type]
                if category == "nodal":
                    return np.stack(
                        [s.data[category][k][0].reshape(-1, size) for s in self.states]
                    )
                if category == "domain":
                    return [
                        np.stack(
                            [
                                s.data[category][k][d + 1].reshape(-1, size)
                                for s in self.states
                            ]
                        )
                        for d in range(len(self.domains))
                        if (d + 1) in self.states[0].data[category][k]
                    ]
                raise NotImplementedError(f"reading {category} variables")
        raise KeyError(f"no plot variable named {name!r}")


# ---------------------------------------------------------------------------
# reading

_CATEGORY = {
    PLT_DIC_GLOBAL: "global", PLT_DIC_NODAL: "nodal", PLT_DIC_DOMAIN: "domain",
    PLT_DIC_SURFACE: "surface", PLT_DIC_EDGE: "edge",
}  # fmt: skip
_DATA_CATEGORY = {
    PLT_GLOBAL_DATA: "global", PLT_NODE_DATA: "nodal", PLT_ELEMENT_DATA: "domain",
    PLT_FACE_DATA: "surface", PLT_EDGE_DATA: "edge",
}  # fmt: skip


def _chunks(buf, start, end):
    """Yield (id, payload start, payload end) for the chunks in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        cid, size = struct.unpack_from("<II", buf, pos)
        if pos + 8 + size > end:
            raise ValueError(
                f"plot file is truncated or corrupt: chunk 0x{cid:08X} at byte "
                f"{pos} runs past the end of its section"
            )
        yield cid, pos + 8, pos + 8 + size
        pos += 8 + size


def _tree(buf, start, end):
    """Parse a branch into a list of (id, value); value is a list for branches
    and a bytes-like slice for leaves. Unknown ids are kept as leaves."""
    out = []
    for cid, a, b in _chunks(buf, start, end):
        out.append((cid, _tree(buf, a, b) if cid in _BRANCHES else buf[a:b]))
    return out


def _first(tree, cid, default=None):
    for c, v in tree:
        if c == cid:
            return v
    return default


def _all(tree, cid):
    return [v for c, v in tree if c == cid]


def _int(b):
    return struct.unpack_from("<i", b)[0]


def _fixed_string(b):
    return bytes(b).split(b"\0", 1)[0].decode("latin-1")


def _counted_string(b):
    n = _int(b)
    return bytes(b[4 : 4 + n]).decode("latin-1")


def read_xplt(path) -> PlotFile:
    buf = memoryview(Path(path).read_bytes())
    if len(buf) < 4 or struct.unpack_from("<I", buf)[0] != SIGNATURE:
        raise ValueError(f"{path}: not a FEBio plot file")
    top = _tree(buf, 4, len(buf))
    pf = PlotFile()
    root = _first(top, PLT_ROOT)
    header = _first(root, PLT_HEADER)
    pf.version = _int(_first(header, PLT_HDR_VERSION))
    pf.compression = _int(_first(header, PLT_HDR_COMPRESSION, struct.pack("<i", 0)))
    if pf.compression:
        raise NotImplementedError("compressed plot files are not read yet")
    sw = _first(header, PLT_HDR_SOFTWARE)
    pf.software = _counted_string(sw) if sw is not None else None
    un = _first(header, PLT_HDR_UNITS)
    pf.units = _counted_string(un) if un is not None else None
    for cid, sub in _first(root, PLT_DICTIONARY):
        items = []
        for item in _all(sub, PLT_DIC_ITEM):
            units = _first(item, PLT_DIC_ITEM_UNITS)
            items.append(
                DictionaryItem(
                    name=_fixed_string(_first(item, PLT_DIC_ITEM_NAME)),
                    type=_int(_first(item, PLT_DIC_ITEM_TYPE)),
                    fmt=_int(_first(item, PLT_DIC_ITEM_FMT)),
                    array_size=_int(
                        _first(item, PLT_DIC_ITEM_ARRAYSIZE, struct.pack("<i", 0))
                    ),
                    units=_fixed_string(units) if units is not None else None,
                )
            )
        pf.dictionary[_CATEGORY[cid]] = items

    mesh = _first(top, PLT_MESH)
    coords = np.frombuffer(
        _first(_first(mesh, PLT_NODE_SECTION), PLT_NODE_COORDS), dtype="<f4"
    ).reshape(-1, 4)
    pf.node_ids = coords[:, 0].copy().view("<i4")
    pf.nodes = coords[:, 1:].copy()
    for dom in _all(_first(mesh, PLT_DOMAIN_SECTION), PLT_DOMAIN):
        hdr = _first(dom, PLT_DOMAIN_HDR)
        name = _first(hdr, PLT_DOM_NAME)
        elems = [
            np.frombuffer(e, dtype="<i4")
            for e in _all(_first(dom, PLT_DOM_ELEM_LIST), PLT_ELEMENT)
        ]
        table = np.vstack(elems) if elems else np.zeros((0, 1), np.int32)
        pf.domains.append(
            PlotDomain(
                elem_type=_int(_first(hdr, PLT_DOM_ELEM_TYPE)),
                part_id=_int(_first(hdr, PLT_DOM_PART_ID)),
                name=_counted_string(name) if name is not None else None,
                elem_ids=table[:, 0].copy(),
                connectivity=table[:, 1:].copy(),
            )
        )

    def surfaces(
        section, item, hdr_tag, id_tag, size_tag, name_tag, max_tag, list_tag, face_tag
    ):
        out = []
        for s in _all(_first(mesh, section, []), item):
            hdr = _first(s, hdr_tag)
            faces = [
                np.frombuffer(f, dtype="<i4")
                for f in _all(_first(s, list_tag), face_tag)
            ]
            out.append(
                PlotSurface(
                    id=_int(_first(hdr, id_tag)),
                    name=_counted_string(_first(hdr, name_tag)),
                    max_nodes=_int(_first(hdr, max_tag)),
                    faces=np.vstack(faces) if faces else np.zeros((0, 2), np.int32),
                )
            )
        return out

    pf.surfaces = surfaces(PLT_SURFACE_SECTION, PLT_SURFACE, PLT_SURFACE_HDR, PLT_SURFACE_ID,
                           PLT_SURFACE_FACES, PLT_SURFACE_NAME, PLT_SURFACE_MAX_FACET_NODES,
                           PLT_FACE_LIST, PLT_FACE)  # fmt: skip
    pf.facet_sets = surfaces(PLT_FACETSET_SECTION, PLT_FACETSET, PLT_FACETSET_HDR, PLT_FACETSET_ID,
                             PLT_FACETSET_SIZE, PLT_FACETSET_NAME, PLT_FACETSET_MAXNODES,
                             PLT_FACETSET_LIST, PLT_FACET)  # fmt: skip
    for s in _all(_first(mesh, PLT_NODESET_SECTION, []), PLT_NODESET):
        hdr = _first(s, PLT_NODESET_HDR)
        pf.node_sets.append(
            (_int(_first(hdr, PLT_NODESET_ID)), _counted_string(_first(hdr, PLT_NODESET_NAME)),
             np.frombuffer(_first(s, PLT_NODESET_LIST, b""), dtype="<i4").copy())
        )  # fmt: skip
    for s in _all(_first(mesh, PLT_ELEMENTSET_SECTION, []), PLT_ELEMENTSET):
        hdr = _first(s, PLT_ELEMENTSET_HDR)
        pf.element_sets.append(
            (_int(_first(hdr, PLT_ELEMENTSET_ID)), _counted_string(_first(hdr, PLT_ELEMENTSET_NAME)),
             np.frombuffer(_first(s, PLT_ELEMENTSET_LIST, b""), dtype="<i4").copy())
        )  # fmt: skip
    for p in _all(_first(mesh, PLT_PARTS_SECTION, []), PLT_PART):
        pf.parts.append(
            (_int(_first(p, PLT_PART_ID)), _fixed_string(_first(p, PLT_PART_NAME)))
        )

    for st in _all(top, PLT_STATE):
        hdr = _first(st, PLT_STATE_HEADER)
        state = PlotState(
            time=struct.unpack_from("<f", _first(hdr, PLT_STATE_HDR_TIME))[0],
            status=_int(_first(hdr, PLT_STATE_STATUS, struct.pack("<i", 0))),
        )
        ms = _first(st, PLT_MESH_STATE)
        if ms is not None:
            state.element_state = np.frombuffer(
                _first(ms, PLT_ELEMENT_STATE, b""), dtype="<u4"
            ).copy()
        for cid, sub in _first(st, PLT_STATE_DATA, []):
            per_var = []
            for var in _all(sub, PLT_STATE_VARIABLE):
                raw = _first(var, PLT_STATE_VAR_DATA)
                # Inside VAR_DATA, each chunk id is the item number (0 for the
                # node set of all nodes, domain index + 1 for domains).
                items = {}
                for item_id, a in raw:
                    items[item_id] = np.frombuffer(a, dtype="<f4").copy()
                per_var.append(items)
            state.data[_DATA_CATEGORY[cid]] = per_var
        pf.states.append(state)
    return pf


# VAR_DATA holds chunks whose ids are item numbers, so it is parsed as a branch
# of leaves. Small integers never collide with the tag values above.
_BRANCHES.add(PLT_STATE_VAR_DATA)


# ---------------------------------------------------------------------------
# writing


class _Writer:
    """Builds the chunk tree in memory, as PltArchive does, then writes it."""

    def __init__(self):
        self.out = bytearray(struct.pack("<I", SIGNATURE))
        self.stack = []

    def begin(self, cid):
        self.stack.append((cid, bytearray()))

    def end(self):
        cid, body = self.stack.pop()
        chunk = struct.pack("<II", cid, len(body)) + body
        (self.stack[-1][1] if self.stack else self.out).extend(chunk)

    def leaf(self, cid, payload: bytes):
        self.stack[-1][1].extend(struct.pack("<II", cid, len(payload)) + payload)

    def int(self, cid, value):
        self.leaf(cid, struct.pack("<i", int(value)))

    def uint(self, cid, value):
        self.leaf(cid, struct.pack("<I", int(value)))

    def float(self, cid, value):
        self.leaf(cid, struct.pack("<f", float(value)))

    def counted_string(self, cid, s):
        b = (s or "").encode("latin-1")
        self.leaf(cid, struct.pack("<i", len(b)) + b)

    def fixed_string(self, cid, s):
        b = (s or "").encode("latin-1")[: STR_SIZE - 1]
        self.leaf(cid, b.ljust(STR_SIZE, b"\0"))

    def array(self, cid, a, dtype):
        self.leaf(cid, np.ascontiguousarray(a, dtype=dtype).tobytes())


def plot_file_from_results(model: Model, results: Results) -> PlotFile:
    """Arrange a Model and its Results as FEBio arranges a plot file.

    Follows FEBioPlotFile::WriteMeshSection: one domain per part with the
    material ID as its part ID, the surfaces of the mesh both as surfaces and
    as facet sets, node sets and element sets by name, and one "part" entry
    per material.
    """
    mesh = model.mesh
    pf = PlotFile(software=results.software)
    requested = model.plot_variables()
    for name in requested:
        if name not in KNOWN_VARIABLES:
            continue  # reported as "not written" by the coverage report
        category, vtype, fmt = KNOWN_VARIABLES[name]
        pf.dictionary.setdefault(category, []).append(DictionaryItem(name, vtype, fmt))
    pf.node_ids = mesh.node_ids.astype(np.int32)
    pf.nodes = mesh.nodes.astype(np.float32)
    for block in mesh.blocks:
        dom = model.domain_of(block.name)
        pf.domains.append(
            PlotDomain(
                elem_type=PLT_ELEM[block.febio_type],
                part_id=model.material(dom.material).id,
                # FEBio writes a name for solid domains only (WriteSolidDomain).
                name=block.name if dom.kind == "SolidDomain" else None,
                elem_ids=block.ids.astype(np.int32),
                connectivity=block.connectivity.astype(np.int32),
            )
        )
    for k, (name, s) in enumerate(mesh.surfaces.items()):
        for target, width in (
            (pf.surfaces, max(len(f) for f in s.facets)),
            (pf.facet_sets, FACET_MAX_NODES),
        ):
            faces = np.zeros((len(s.facets), width + 2), dtype=np.int32)
            for i, f in enumerate(s.facets):
                faces[i, 0] = i + 1
                faces[i, 1] = len(f)
                faces[i, 2 : 2 + len(f)] = f
            target.append(
                PlotSurface(id=k + 1, name=name, max_nodes=width, faces=faces)
            )
    # Node sets, in FEBio's order (checked against FEBio 4.9 and 4.13 files):
    # one per <Nodes> block, then the named node sets, then one per edge that
    # a boundary condition uses as "@edge:Name".
    sets = [(name, np.arange(start, start + n)) for name, start, n in mesh.node_blocks]
    sets += list(mesh.node_sets.items())
    used = []
    for r in model.boundary + [b for s in model.steps for b in s.boundary]:
        ref = r.attrs.get("node_set", "")
        if ref.startswith("@edge:") and ref not in used:
            used.append(ref)
            sets.append((ref[len("@edge:") :], mesh.node_set_indices(ref)))
    for k, (name, idx) in enumerate(sets):
        pf.node_sets.append((k + 1, name or "", np.asarray(idx, dtype=np.int32)))
    # Element sets: one per part, then the named element sets; items are element IDs.
    esets = [(b.name, b.ids) for b in mesh.blocks] + list(mesh.element_sets.items())
    for k, (name, ids) in enumerate(esets):
        pf.element_sets.append((k + 1, name, np.asarray(ids, dtype=np.int32)))
    pf.parts = [(m.id, m.name) for m in model.materials]

    # Element status flags: 1 is an active element (FEElement::status).
    count = sum(b.count for b in mesh.blocks)
    for s in results.states:
        state = PlotState(
            time=s.time, status=0, element_state=np.ones(count, np.uint32)
        )
        for category, items in pf.dictionary.items():
            per_var = []
            for item in items:
                if item.name == "displacement":
                    per_var.append({0: s.displacement.astype(np.float32).ravel()})
                elif item.name == "stress":
                    per_var.append(
                        {
                            d + 1: s.stress[b.name].astype(np.float32).ravel()
                            for d, b in enumerate(mesh.blocks)
                        }
                    )
            state.data[category] = per_var
        pf.states.append(state)
    return pf


def write_xplt(model_or_plotfile, results_or_path, path=None) -> None:
    """write_xplt(model, results, path) or write_xplt(plot_file, path)."""
    if isinstance(model_or_plotfile, PlotFile):
        pf, path = model_or_plotfile, results_or_path
    else:
        pf = plot_file_from_results(model_or_plotfile, results_or_path)
    w = _Writer()

    w.begin(PLT_ROOT)
    w.begin(PLT_HEADER)
    w.uint(PLT_HDR_VERSION, pf.version)
    w.int(PLT_HDR_COMPRESSION, 0)
    if pf.software:
        w.counted_string(PLT_HDR_SOFTWARE, pf.software)
    if pf.units is not None:
        w.counted_string(PLT_HDR_UNITS, pf.units)
    w.end()
    w.begin(PLT_DICTIONARY)
    for tag, category in ((PLT_DIC_GLOBAL, "global"), (PLT_DIC_NODAL, "nodal"), (PLT_DIC_DOMAIN, "domain"),
                          (PLT_DIC_SURFACE, "surface"), (PLT_DIC_EDGE, "edge")):  # fmt: skip
        items = pf.dictionary.get(category)
        if not items:
            continue
        w.begin(tag)
        for item in items:
            w.begin(PLT_DIC_ITEM)
            w.int(PLT_DIC_ITEM_TYPE, item.type)
            w.int(PLT_DIC_ITEM_FMT, item.fmt)
            w.int(PLT_DIC_ITEM_ARRAYSIZE, item.array_size)
            w.fixed_string(PLT_DIC_ITEM_NAME, item.name)
            if pf.units is not None and item.units:
                w.fixed_string(PLT_DIC_ITEM_UNITS, item.units)
            w.end()
        w.end()
    w.end()
    w.end()

    w.begin(PLT_MESH)
    w.begin(PLT_NODE_SECTION)
    w.begin(PLT_NODE_HEADER)
    w.int(PLT_NODE_SIZE, len(pf.nodes))
    w.int(PLT_NODE_DIM, 3)
    w.end()
    # Node ID (as int32) followed by three float32 coordinates, per node.
    table = np.zeros((len(pf.nodes), 4), dtype="<f4")
    table[:, 0] = pf.node_ids.astype("<i4").view("<f4")
    table[:, 1:] = pf.nodes
    w.array(PLT_NODE_COORDS, table, "<f4")
    w.end()
    w.begin(PLT_DOMAIN_SECTION)
    for d in pf.domains:
        w.begin(PLT_DOMAIN)
        w.begin(PLT_DOMAIN_HDR)
        w.int(PLT_DOM_ELEM_TYPE, d.elem_type)
        w.int(PLT_DOM_PART_ID, d.part_id)
        w.int(PLT_DOM_ELEMS, len(d.elem_ids))
        if d.name is not None:
            w.counted_string(PLT_DOM_NAME, d.name)
        w.end()
        w.begin(PLT_DOM_ELEM_LIST)
        for eid, conn in zip(d.elem_ids, d.connectivity):
            w.array(PLT_ELEMENT, np.concatenate([[eid], conn]), "<i4")
        w.end()
        w.end()
    w.end()

    def write_surfaces(
        items,
        section,
        item_tag,
        hdr_tag,
        id_tag,
        size_tag,
        name_tag,
        max_tag,
        list_tag,
        face_tag,
        max_last,
    ):
        if not items:
            return
        w.begin(section)
        for s in items:
            w.begin(item_tag)
            w.begin(hdr_tag)
            w.int(id_tag, s.id)
            w.int(size_tag, len(s.faces))
            w.counted_string(name_tag, s.name)
            w.int(max_tag, s.max_nodes)
            w.end()
            w.begin(list_tag)
            for f in s.faces:
                w.array(face_tag, f, "<i4")
            w.end()
            w.end()
        w.end()

    write_surfaces(pf.surfaces, PLT_SURFACE_SECTION, PLT_SURFACE, PLT_SURFACE_HDR, PLT_SURFACE_ID,
                   PLT_SURFACE_FACES, PLT_SURFACE_NAME, PLT_SURFACE_MAX_FACET_NODES, PLT_FACE_LIST, PLT_FACE, True)  # fmt: skip
    if pf.node_sets:
        w.begin(PLT_NODESET_SECTION)
        for sid, name, idx in pf.node_sets:
            w.begin(PLT_NODESET)
            w.begin(PLT_NODESET_HDR)
            w.int(PLT_NODESET_ID, sid)
            w.int(PLT_NODESET_SIZE, len(idx))
            w.counted_string(PLT_NODESET_NAME, name)
            w.end()
            w.array(PLT_NODESET_LIST, idx, "<i4")
            w.end()
        w.end()
    if pf.element_sets:
        w.begin(PLT_ELEMENTSET_SECTION)
        for sid, name, items in pf.element_sets:
            w.begin(PLT_ELEMENTSET)
            w.begin(PLT_ELEMENTSET_HDR)
            w.int(PLT_ELEMENTSET_ID, sid)
            w.int(PLT_ELEMENTSET_SIZE, len(items))
            w.counted_string(PLT_ELEMENTSET_NAME, name)
            w.end()
            w.array(PLT_ELEMENTSET_LIST, items, "<i4")
            w.end()
        w.end()
    write_surfaces(pf.facet_sets, PLT_FACETSET_SECTION, PLT_FACETSET, PLT_FACETSET_HDR, PLT_FACETSET_ID,
                   PLT_FACETSET_SIZE, PLT_FACETSET_NAME, PLT_FACETSET_MAXNODES, PLT_FACETSET_LIST, PLT_FACET, True)  # fmt: skip
    if pf.parts:
        w.begin(PLT_PARTS_SECTION)
        for pid, name in pf.parts:
            w.begin(PLT_PART)
            w.uint(PLT_PART_ID, pid)
            w.fixed_string(PLT_PART_NAME, name)
            w.end()
        w.end()
    w.end()

    for s in pf.states:
        w.begin(PLT_STATE)
        w.begin(PLT_STATE_HEADER)
        w.float(PLT_STATE_HDR_TIME, s.time)
        w.int(PLT_STATE_STATUS, s.status)
        w.end()
        w.begin(PLT_MESH_STATE)
        w.array(
            PLT_ELEMENT_STATE,
            s.element_state if s.element_state is not None else np.zeros(0),
            "<u4",
        )
        w.end()
        w.begin(PLT_STATE_DATA)
        for tag, category in ((PLT_GLOBAL_DATA, "global"), (PLT_NODE_DATA, "nodal"), (PLT_ELEMENT_DATA, "domain"),
                              (PLT_FACE_DATA, "surface"), (PLT_EDGE_DATA, "edge")):  # fmt: skip
            per_var = s.data.get(category)
            if not per_var:
                continue
            w.begin(tag)
            for k, items in enumerate(per_var):
                w.begin(PLT_STATE_VARIABLE)
                w.uint(PLT_STATE_VAR_ID, k + 1)
                w.begin(PLT_STATE_VAR_DATA)
                for item_id, values in items.items():
                    w.array(item_id, values, "<f4")
                w.end()
                w.end()
            w.end()
        w.end()
        w.end()
    Path(path).write_bytes(bytes(w.out))
