"""Model and results as VTK data, for 3D Slicer and SlicerHeart.

A Slicer module calls these to show a FEBio model or a result, and to turn a
mesh made in Slicer into a Model that `write_feb` can save. VTK is imported
only here, so the rest of the package does not need it.

Node order: FEBio's order for hex8, penta6, tet4, quad4, tri3 and line2 is
the same as VTK's for VTK_HEXAHEDRON, VTK_WEDGE, VTK_TETRA, VTK_QUAD,
VTK_TRIANGLE and VTK_LINE, so connectivity is passed through unchanged.
"""

from __future__ import annotations

import numpy as np

from ..model import ELEMENT_NODES, Domain, ElementBlock, Material, Mesh, Model, Record
from ..results import StateResult

# FEBio element type -> VTK cell type id (vtkCellType.h).
VTK_CELL = {"hex8": 12, "penta6": 13, "tet4": 10, "quad4": 9, "tri3": 5, "line2": 3}
FEBIO_TYPE = {v: k for k, v in VTK_CELL.items()}
STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "xz")


def flat_cells(model: Model):
    """Cells of all parts as flat arrays: (types, offsets, connectivity,
    element ids, part index). Shared by this module and the VTKHDF writer."""
    types, sizes, conn, ids, part = [], [], [], [], []
    for p, b in enumerate(model.mesh.blocks):
        types.append(np.full(b.count, VTK_CELL[b.febio_type], dtype=np.uint8))
        sizes.append(np.full(b.count, b.connectivity.shape[1], dtype=np.int64))
        conn.append(b.connectivity.ravel())
        ids.append(b.ids)
        part.append(np.full(b.count, p, dtype=np.int32))
    if not types:
        return (
            np.zeros(0, np.uint8),
            np.zeros(1, np.int64),
            np.zeros(0, np.int64),
            np.zeros(0, np.int32),
            np.zeros(0, np.int32),
        )
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(sizes))]).astype(np.int64)
    return (
        np.concatenate(types),
        offsets,
        np.concatenate(conn).astype(np.int64),
        np.concatenate(ids).astype(np.int32),
        np.concatenate(part),
    )


def cell_stress(model: Model, state: StateResult) -> np.ndarray:
    """Element stress of all parts stacked in cell order, shape (cells, 6)."""
    return np.vstack([state.stress[b.name] for b in model.mesh.blocks])


def model_to_vtk(model: Model, state: StateResult = None):
    """A vtkUnstructuredGrid of the model, with one state's results if given.

    Point data: "node_id", and "displacement" when a state is given.
    Cell data:  "element_id", "part" (index of the part), and "stress"
                (xx, yy, zz, xy, yz, xz) when a state is given.
    Points are the reference coordinates; apply "displacement" with a warp
    filter or a Slicer transform to see the deformed shape.
    """
    import vtk
    from vtk.util import numpy_support as ns

    types, offsets, conn, ids, part = flat_cells(model)
    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(model.mesh.nodes), deep=True))
    grid.SetPoints(pts)
    cells = vtk.vtkCellArray()
    cells.SetData(
        ns.numpy_to_vtkIdTypeArray(offsets, deep=True),
        ns.numpy_to_vtkIdTypeArray(conn, deep=True),
    )
    grid.SetCells(ns.numpy_to_vtk(types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR), cells)

    def add(data, name, array, components=None):
        a = ns.numpy_to_vtk(np.ascontiguousarray(array), deep=True)
        a.SetName(name)
        if components:
            for i, c in enumerate(components):
                a.SetComponentName(i, c)
        data.AddArray(a)

    add(grid.GetPointData(), "node_id", model.mesh.node_ids.astype(np.int32))
    add(grid.GetCellData(), "element_id", ids)
    add(grid.GetCellData(), "part", part)
    if state is not None:
        add(grid.GetPointData(), "displacement", state.displacement)
        add(grid.GetCellData(), "stress", cell_stress(model, state), STRESS_COMPONENTS)
    return grid


def mesh_from_vtk(grid, part_prefix: str = "Part") -> Mesh:
    """A Mesh from a vtkUnstructuredGrid (or vtkPolyData of quads/triangles).

    Cells are grouped into one part per cell type, named Part1, Part2, ...
    Node and element IDs are numbered from 1, as FEBio Studio does. If the
    grid has "node_id" or "element_id" arrays (as written by model_to_vtk),
    those IDs are used instead.
    """
    from vtk.util import numpy_support as ns

    pts = ns.vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    mesh = Mesh(nodes=pts, node_ids=np.arange(1, len(pts) + 1, dtype=np.int32))
    nid = grid.GetPointData().GetArray("node_id")
    if nid is not None:
        mesh.node_ids = ns.vtk_to_numpy(nid).astype(np.int32)
    mesh.node_blocks = [("Object1", 0, len(pts))]
    eid = grid.GetCellData().GetArray("element_id")
    eid = ns.vtk_to_numpy(eid).astype(np.int32) if eid is not None else None
    by_type = {}
    for c in range(grid.GetNumberOfCells()):
        by_type.setdefault(grid.GetCellType(c), []).append(c)
    next_id = 1
    for k, (ctype, cells) in enumerate(by_type.items()):
        if ctype not in FEBIO_TYPE:
            raise ValueError(f"VTK cell type {ctype} has no FEBio element type here")
        etype = FEBIO_TYPE[ctype]
        n = ELEMENT_NODES[etype]
        conn = np.array(
            [[grid.GetCell(c).GetPointId(i) for i in range(n)] for c in cells],
            dtype=np.int32,
        )
        ids = eid[cells] if eid is not None else np.arange(next_id, next_id + len(cells), dtype=np.int32)
        next_id += len(cells)
        mesh.blocks.append(
            ElementBlock(
                name=f"{part_prefix}{k + 1}",
                febio_type=etype,
                connectivity=conn,
                ids=ids,
            )
        )
    return mesh


def model_from_vtk(grid, material_type="neo-Hookean", material_params=None, shell_thickness=None) -> Model:
    """A minimal Model around a VTK mesh: one material on every part.

    Meant as the starting point for an export from Slicer: the caller adds
    boundary conditions, loads, steps and output requests as Records, then
    calls write_feb.
    """
    model = Model()
    model.mesh = mesh_from_vtk(grid)
    rec = Record("material", {"id": "1", "name": "Material1", "type": material_type})
    for tag, value in (material_params or {}).items():
        rec.set(tag, value)
    model.materials.append(Material(id=1, name="Material1", type=material_type, params=rec))
    for b in model.mesh.blocks:
        if b.febio_type in ("quad4", "tri3"):
            p = Record("ShellDomain", {"name": b.name, "mat": "Material1"})
            if shell_thickness is not None:
                p.set("shell_thickness", float(shell_thickness))
            model.domains.append(Domain("ShellDomain", b.name, "Material1", p))
        else:
            p = Record("SolidDomain", {"name": b.name, "mat": "Material1"})
            model.domains.append(Domain("SolidDomain", b.name, "Material1", p))
    return model
