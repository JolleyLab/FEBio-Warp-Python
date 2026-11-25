"""Read and write FEBio input files (.feb), format specification 4.0.

FEBio counterpart: FEBioXML
(https://github.com/febiosoftware/FEBio/tree/develop/FEBioXML).
Format reference: https://help.febio.org/docs/FEBioUser-4-0/UM40-Chapter-3.html

The reader fills a `warp_febio.model.Model` directly. Only spec 4.0 is read;
older files can be converted with FEBio Studio. Mesh data becomes arrays.
Everything else is kept as `Record` trees with FEBio's own tags, so nothing
in the file is dropped: what this code does not yet interpret is still
carried, written back, and listed by the coverage report at compile time.
Anything the reader cannot place at all is added to `Model.issues`.

Uses only the Python standard library and NumPy, so it runs inside
3D Slicer's Python with nothing to install.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..model import (
    ELEMENT_NODES,
    FACET_NODES,
    Domain,
    ElementBlock,
    Issue,
    LoadController,
    Material,
    Mesh,
    Model,
    Record,
    Step,
    Surface,
)

SUPPORTED_SPEC = "4.0"

# Top-level sections the reader places in typed fields of the Model.
_TYPED_SECTIONS = {
    "Module",
    "Globals",
    "Control",
    "Material",
    "Mesh",
    "MeshDomains",
    "MeshData",
    "Boundary",
    "Loads",
    "Contact",
    "Step",
    "LoadData",
    "Output",
}
# Top-level sections that are valid FEBio but carried as Records for now.
_CARRIED_SECTIONS = {
    "Initial",
    "Rigid",
    "Constraints",
    "Discrete",
    "MeshAdaptor",
    "Code",
}


class FebFormatError(ValueError):
    """The file is not a FEBio spec 4.0 input this reader can load."""


# ---------------------------------------------------------------------------
# text <-> values


def _parse_text(text):
    """Parse element text: int, float, a numeric array, or a string."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        try:
            return np.array([int(p) for p in parts], dtype=np.int64)
        except ValueError:
            pass
        try:
            return np.array([float(p) for p in parts], dtype=np.float64)
        except ValueError:
            return s
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _format_number(x):
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    # repr gives the shortest text that reads back to the same double.
    return repr(float(x))


def _format_value(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return ",".join(_format_number(x) for x in value.ravel())
    if isinstance(value, str):
        return value
    return _format_number(value)


def _record(elem) -> Record:
    """Generic conversion of an XML element to a Record, losing nothing."""
    return Record(
        tag=elem.tag,
        attrs=dict(elem.attrib),
        value=_parse_text(elem.text),
        children=[_record(c) for c in elem],
    )


def _rows(elems, dtype):
    """Numeric rows from many small elements at once (the bulk-data path).

    Joining the text and converting in one call keeps large meshes fast:
    no per-node Python objects are made.
    """
    if not elems:
        return np.zeros((0, 0), dtype=dtype)
    width = elems[0].text.count(",") + 1
    flat = np.array(",".join(e.text for e in elems).split(","), dtype=dtype)
    if flat.size != width * len(elems):
        raise FebFormatError("rows of unequal length in a mesh block")
    return flat.reshape(len(elems), width)


def _ids(elems, key="id"):
    return np.array([e.get(key) for e in elems], dtype=np.int64).astype(np.int32)


# ---------------------------------------------------------------------------
# reading


def read_feb(path) -> Model:
    """Read a FEBio spec 4.0 input file into a Model."""
    root = ET.parse(str(path)).getroot()
    return _read_root(root)


def read_feb_string(text: str) -> Model:
    return _read_root(ET.fromstring(text))


def _read_root(root) -> Model:
    if root.tag != "febio_spec":
        raise FebFormatError(f"root element is <{root.tag}>, expected <febio_spec>")
    version = root.get("version")
    if version != SUPPORTED_SPEC:
        raise FebFormatError(
            f"febio_spec version {version!r}: only {SUPPORTED_SPEC} is read. "
            "FEBio Studio can convert older files."
        )
    model = Model(spec_version=version)
    top_control = None
    for sec in root:
        tag = sec.tag
        if tag == "Module":
            model.module = sec.get("type", "solid")
            model.module_record = _record(sec)
        elif tag == "Globals":
            model.globals = _record(sec)
        elif tag == "Control":
            top_control = _record(sec)
        elif tag == "Material":
            _read_materials(sec, model)
        elif tag == "Mesh":
            _read_mesh(sec, model)
        elif tag == "MeshDomains":
            for d in sec:
                rec = _record(d)
                model.domains.append(
                    Domain(
                        kind=d.tag,
                        name=d.get("name"),
                        material=d.get("mat"),
                        params=rec,
                    )
                )
        elif tag == "MeshData":
            model.mesh_data += [_read_mesh_data(c) for c in sec]
        elif tag == "Boundary":
            model.boundary += [_record(c) for c in sec]
        elif tag == "Loads":
            model.loads += [_record(c) for c in sec]
        elif tag == "Contact":
            model.contact += [_record(c) for c in sec]
        elif tag == "Step":
            model.single_step_layout = False
            for s in sec:
                model.steps.append(_read_step(s, model))
        elif tag == "LoadData":
            _read_load_data(sec, model)
        elif tag == "Output":
            model.output = _record(sec)
        elif tag in _CARRIED_SECTIONS:
            model.other_sections.append(_record(sec))
        else:
            model.other_sections.append(_record(sec))
            model.issues.append(
                Issue(tag, "section not known to the reader; carried as written")
            )
    if top_control is not None:
        if model.steps:
            model.issues.append(
                Issue("Control", "top-level Control together with Step: not read")
            )
        else:
            model.steps.append(Step(id=1, name="Step1", control=top_control))
    return model


def _read_materials(sec, model):
    for m in sec:
        if m.tag != "material":
            model.issues.append(Issue(f"Material/{m.tag}", "not a <material>; skipped"))
            continue
        rec = _record(m)
        model.materials.append(
            Material(
                id=int(m.get("id")), name=m.get("name"), type=m.get("type"), params=rec
            )
        )


def _read_mesh(sec, model):
    mesh = model.mesh
    coords, node_ids = [], []
    raw_sets, raw_surfaces, raw_blocks = [], [], []
    for item in sec:
        tag = item.tag
        if tag == "Nodes":
            nodes = list(item)
            mesh.node_blocks.append(
                (item.get("name"), sum(len(c) for c in coords), len(nodes))
            )
            if nodes:
                coords.append(_rows(nodes, np.float64))
                node_ids.append(_ids(nodes))
        elif tag == "Elements":
            raw_blocks.append(item)
        elif tag == "NodeSet":
            raw_sets.append(item)
        elif tag == "Surface":
            raw_surfaces.append(item)
        elif tag == "ElementSet":
            mesh.element_sets[item.get("name")] = _id_list(item, "e")
        else:
            # Edge, SurfacePair, DiscreteSet, PartList, ...: carried as written.
            mesh.other.append(_record(item))
    if coords:
        mesh.nodes = np.vstack(coords)
        mesh.node_ids = np.concatenate(node_ids)
    if mesh.nodes.size and mesh.nodes.shape[1] != 3:
        raise FebFormatError("node coordinates must have three components")
    if len(np.unique(mesh.node_ids)) != len(mesh.node_ids):
        raise FebFormatError("duplicate node IDs")

    for item in raw_blocks:
        etype = item.get("type")
        elems = list(item)
        conn = _rows(elems, np.int64)
        if (
            etype in ELEMENT_NODES
            and conn.size
            and conn.shape[1] != ELEMENT_NODES[etype]
        ):
            raise FebFormatError(
                f"Elements {item.get('name')!r}: {etype} needs "
                f"{ELEMENT_NODES[etype]} nodes, found {conn.shape[1]}"
            )
        mesh.blocks.append(
            ElementBlock(
                name=item.get("name"),
                febio_type=etype,
                connectivity=mesh.node_index(conn.ravel()).reshape(conn.shape),
                ids=_ids(elems),
            )
        )
    for item in raw_sets:
        mesh.node_sets[item.get("name")] = mesh.node_index(_id_list(item, "n"))
    for item in raw_surfaces:
        facets = list(item)
        for f in facets:
            if f.tag not in FACET_NODES:
                model.issues.append(
                    Issue(f"Mesh/Surface/{f.tag}", "facet type not known to the reader")
                )
        mesh.surfaces[item.get("name")] = Surface(
            name=item.get("name"),
            facet_types=[f.tag for f in facets],
            facets=[
                mesh.node_index(np.array(f.text.split(","), dtype=np.int64))
                for f in facets
            ],
            ids=_ids(facets) if facets else np.zeros(0, np.int32),
        )


def _id_list(item, child_tag):
    """IDs of a NodeSet or ElementSet: a comma-separated list (spec 4.0), or
    one child element per ID with an "id" attribute (also accepted by FEBio)."""
    kids = list(item)
    if kids:
        return _ids(kids)
    text = (item.text or "").strip()
    if not text:
        return np.zeros(0, np.int32)
    return np.array(text.replace("\n", "").split(","), dtype=np.int64).astype(np.int32)


def _read_mesh_data(item) -> Record:
    """ElementData / NodeData / SurfaceData.

    When the rows are one <e lid="i"> per item in order (the usual case, for
    example shell thickness), they are stored as one (n, k) array in `value`.
    Otherwise the element is carried generically.
    """
    kids = list(item)
    row_tag = {"ElementData": "e", "NodeData": "n", "SurfaceData": "f"}.get(item.tag)
    if (
        row_tag
        and kids
        and all(k.tag == row_tag and len(k) == 0 for k in kids)
        and [k.get("lid") for k in kids] == [str(i + 1) for i in range(len(kids))]
    ):
        try:
            return Record(item.tag, dict(item.attrib), _rows(kids, np.float64), [])
        except (ValueError, FebFormatError):
            pass
    return _record(item)


def _read_step(s, model) -> Step:
    step = Step(
        id=int(s.get("id", len(model.steps) + 1)),
        name=s.get("name", ""),
        control=Record("Control"),
    )
    for sec in s:
        if sec.tag == "Control":
            step.control = _record(sec)
        elif sec.tag == "Boundary":
            step.boundary += [_record(c) for c in sec]
        elif sec.tag == "Loads":
            step.loads += [_record(c) for c in sec]
        elif sec.tag == "Contact":
            step.contact += [_record(c) for c in sec]
        else:
            step.other.append(_record(sec))
    return step


def _read_load_data(sec, model):
    for lc in sec:
        points = None
        rec = Record(lc.tag, dict(lc.attrib), None, [])
        for c in lc:
            if c.tag == "points":
                kids = list(c)
                points = _rows(kids, np.float64) if kids else np.zeros((0, 2))
                rec.children.append(Record("points", dict(c.attrib), points, []))
            else:
                rec.children.append(_record(c))
        model.load_controllers.append(
            LoadController(
                id=int(lc.get("id")), type=lc.get("type"), params=rec, points=points
            )
        )


# ---------------------------------------------------------------------------
# writing


def write_feb(model: Model, path) -> None:
    """Write a Model as a FEBio spec 4.0 input file.

    Section order follows FEBio Studio's exporter (FEBioExport4), which FEBio
    reads. A Model read by `read_feb` and written here reads back equal.
    """
    root = ET.Element("febio_spec", {"version": model.spec_version})
    if model.module_record is not None:
        root.append(_element(model.module_record))
    else:
        ET.SubElement(root, "Module", {"type": model.module})
    if model.single_step_layout and model.steps:
        root.append(_element(model.steps[0].control, "Control"))
    if model.globals is not None:
        root.append(_element(model.globals))
    if model.materials:
        sec = ET.SubElement(root, "Material")
        for m in model.materials:
            sec.append(_element(m.params))
    root.append(_mesh_element(model.mesh))
    if model.domains:
        sec = ET.SubElement(root, "MeshDomains")
        for d in model.domains:
            sec.append(_element(d.params))
    if model.mesh_data:
        sec = ET.SubElement(root, "MeshData")
        for r in model.mesh_data:
            sec.append(_mesh_data_element(r))
    carried = {r.tag: r for r in model.other_sections}
    if "Initial" in carried:
        root.append(_element(carried.pop("Initial")))
    for tag, items in (("Boundary", model.boundary),):
        if items:
            sec = ET.SubElement(root, tag)
            for r in items:
                sec.append(_element(r))
    for tag in ("Rigid",):
        if tag in carried:
            root.append(_element(carried.pop(tag)))
    for tag, items in (("Loads", model.loads), ("Contact", model.contact)):
        if items:
            sec = ET.SubElement(root, tag)
            for r in items:
                sec.append(_element(r))
    for tag in ("Constraints", "Discrete"):
        if tag in carried:
            root.append(_element(carried.pop(tag)))
    for r in carried.values():
        root.append(_element(r))
    if not model.single_step_layout:
        sec = ET.SubElement(root, "Step")
        for s in model.steps:
            e = ET.SubElement(sec, "step", {"id": str(s.id), "name": s.name})
            e.append(_element(s.control, "Control"))
            for tag, items in (
                ("Boundary", s.boundary),
                ("Loads", s.loads),
                ("Contact", s.contact),
            ):
                if items:
                    sub = ET.SubElement(e, tag)
                    for r in items:
                        sub.append(_element(r))
            for r in s.other:
                e.append(_element(r))
    if model.load_controllers:
        sec = ET.SubElement(root, "LoadData")
        for lc in model.load_controllers:
            e = ET.SubElement(sec, lc.params.tag, lc.params.attrs)
            for c in lc.params.children:
                if c.tag == "points":
                    pe = ET.SubElement(e, "points", c.attrs)
                    for row in (
                        np.atleast_2d(c.value)
                        if c.value is not None and len(c.value)
                        else []
                    ):
                        ET.SubElement(pe, "pt").text = _format_value(row)
                else:
                    e.append(_element(c))
    if model.output is not None:
        root.append(_element(model.output))
    ET.indent(root, space="\t")
    Path(path).write_bytes(
        b'<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        + ET.tostring(root, encoding="ISO-8859-1", xml_declaration=False)
        + b"\n"
    )


def _element(rec: Record, tag=None):
    e = ET.Element(tag or rec.tag, {k: str(v) for k, v in rec.attrs.items()})
    e.text = _format_value(rec.value)
    for c in rec.children:
        e.append(_element(c))
    return e


def _mesh_data_element(rec: Record):
    if isinstance(rec.value, np.ndarray) and rec.value.ndim == 2:
        row_tag = {"ElementData": "e", "NodeData": "n", "SurfaceData": "f"}[rec.tag]
        e = ET.Element(rec.tag, rec.attrs)
        for i, row in enumerate(rec.value):
            ET.SubElement(e, row_tag, {"lid": str(i + 1)}).text = _format_value(row)
        return e
    return _element(rec)


def _mesh_element(mesh: Mesh):
    sec = ET.Element("Mesh")
    for name, start, count in mesh.node_blocks:
        e = ET.SubElement(sec, "Nodes", {"name": name} if name else {})
        for i in range(start, start + count):
            ET.SubElement(e, "node", {"id": str(int(mesh.node_ids[i]))}).text = (
                _format_value(mesh.nodes[i])
            )
    for b in mesh.blocks:
        e = ET.SubElement(sec, "Elements", {"type": b.febio_type, "name": b.name})
        ids = mesh.node_ids[b.connectivity]
        for eid, row in zip(b.ids, ids):
            ET.SubElement(e, "elem", {"id": str(int(eid))}).text = ",".join(
                str(int(n)) for n in row
            )
    # Edge items sit between Elements and NodeSet in FEBio Studio's output.
    for r in mesh.other:
        if r.tag == "Edge":
            sec.append(_element(r))
    for name, idx in mesh.node_sets.items():
        ET.SubElement(sec, "NodeSet", {"name": name}).text = ",".join(
            str(int(n)) for n in mesh.node_ids[idx]
        )
    for name, s in mesh.surfaces.items():
        e = ET.SubElement(sec, "Surface", {"name": name})
        for fid, ftype, f in zip(s.ids, s.facet_types, s.facets):
            ET.SubElement(e, ftype, {"id": str(int(fid))}).text = ",".join(
                str(int(n)) for n in mesh.node_ids[f]
            )
    for name, ids in mesh.element_sets.items():
        ET.SubElement(sec, "ElementSet", {"name": name}).text = ",".join(
            str(int(n)) for n in ids
        )
    for r in mesh.other:
        if r.tag != "Edge":
            sec.append(_element(r))
    return sec
