"""Write results as VTKHDF: one HDF5 file with all time steps, in double precision.

Format: https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format
(UnstructuredGrid with a Steps group, format version 2.0).

Why this exists beside .xplt: FEBio's plot file stores single precision, and
3D Slicer, SlicerHeart and ParaView read VTKHDF directly. The mesh is written
once and shared by every step; "displacement" (points) and "stress" (cells)
are appended per step. FEBio node and element IDs are written as "node_id" and
"element_id" so items can be matched with FEBio's.

Written with h5py, so the solver side does not need VTK. Checked by reading
the file back with VTK's vtkHDFReader.
"""

from __future__ import annotations

import numpy as np

from ..model import Model
from ..results import Results
from .vtk_adapter import cell_stress, flat_cells


def write_vtkhdf(model: Model, results: Results, path) -> None:
    import h5py

    types, offsets, conn, elem_ids, part = flat_cells(model)
    points = np.ascontiguousarray(model.mesh.nodes, dtype=np.float64)
    nsteps, npts, ncells = len(results.states), len(points), len(types)

    with h5py.File(str(path), "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype=np.int64)
        # The reader requires a fixed-length ASCII string here.
        root.attrs.create(
            "Type",
            b"UnstructuredGrid",
            dtype=h5py.string_dtype("ascii", len("UnstructuredGrid")),
        )

        # Geometry and topology, written once (one part).
        root["NumberOfPoints"] = np.array([npts], dtype=np.int64)
        root["NumberOfCells"] = np.array([ncells], dtype=np.int64)
        root["NumberOfConnectivityIds"] = np.array([len(conn)], dtype=np.int64)
        root["Points"] = points
        root["Connectivity"] = conn
        root["Offsets"] = offsets
        root["Types"] = types

        point_data = root.create_group("PointData")
        cell_data = root.create_group("CellData")
        # Static arrays are stored once; every step points at offset 0.
        point_data["node_id"] = model.mesh.node_ids.astype(np.int32)
        cell_data["element_id"] = elem_ids
        cell_data["part"] = part
        point_data["displacement"] = np.concatenate(
            [s.displacement for s in results.states]
        ).astype(np.float64)
        cell_data["stress"] = np.concatenate(
            [cell_stress(model, s) for s in results.states]
        ).astype(np.float64)

        steps = root.create_group("Steps")
        steps.attrs["NSteps"] = np.int64(nsteps)
        steps["Values"] = results.times.astype(np.float64)
        zeros = np.zeros(nsteps, dtype=np.int64)
        steps["PartOffsets"] = zeros
        steps["NumberOfParts"] = np.ones(nsteps, dtype=np.int64)
        steps["PointOffsets"] = zeros
        steps["CellOffsets"] = zeros.reshape(nsteps, 1)
        steps["ConnectivityIdOffsets"] = zeros.reshape(nsteps, 1)
        pdo = steps.create_group("PointDataOffsets")
        cdo = steps.create_group("CellDataOffsets")
        pdo["node_id"] = zeros
        cdo["element_id"] = zeros
        cdo["part"] = zeros
        pdo["displacement"] = np.arange(nsteps, dtype=np.int64) * npts
        cdo["stress"] = np.arange(nsteps, dtype=np.int64) * ncells
