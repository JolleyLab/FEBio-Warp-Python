"""Binary/ASCII STL preparation; supplied coordinates are explicitly millimeters."""

import struct
from collections import defaultdict
import numpy as np


def load_patch(path):
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + count * 50:
        records = np.frombuffer(
            raw,
            dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
            offset=84,
        )
        vertices = records["v"].reshape(-1, 3).astype(float)
    else:
        text = raw.decode("ascii")
        rows = [
            line.split()
            for line in text.splitlines()
            if line.strip().startswith("vertex ")
        ]
        if (
            not text.lstrip().startswith("solid")
            or not rows
            or len(rows) % 3
            or any(len(r) != 4 for r in rows)
        ):
            raise ValueError("Invalid ASCII STL vertices")
        vertices = np.array([[float(v) for v in row[1:]] for row in rows], dtype=float)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite STL vertex")
    x, inverse = np.unique(vertices, axis=0, return_inverse=True)
    tri = inverse.reshape(-1, 3)
    edges = defaultdict(list)
    for k, face in enumerate(tri):
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((int(a), int(b))))].append((k, int(a), int(b)))
    if any(len(v) > 2 for v in edges.values()):
        raise ValueError("Nonmanifold edge")
    if any(len(v) == 2 and v[0][1:] == v[1][1:] for v in edges.values()):
        raise ValueError("Inconsistent winding")
    boundary_edges = [v[0][1:] for v in edges.values() if len(v) == 1]
    neighbors = defaultdict(list)
    for a, b in boundary_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    if not neighbors or any(len(v) != 2 for v in neighbors.values()):
        raise ValueError("Boundary is not a simple loop")
    loop = [min(neighbors)]
    previous = -1
    while True:
        nxt = next(v for v in neighbors[loop[-1]] if v != previous)
        if nxt == loop[0]:
            break
        if nxt in loop:
            raise ValueError("Invalid boundary")
        previous = loop[-1]
        loop.append(nxt)
    if len(loop) != len(neighbors):
        raise ValueError("Multiple boundary loops")
    # Explicitly require the one-component disk expected by this benchmark.
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = {0}
    stack = [0]
    while stack:
        for v in adjacency[stack.pop()] - seen:
            seen.add(v)
            stack.append(v)
    if len(seen) != len(x) or len(x) - len(edges) + len(tri) != 1:
        raise ValueError("Expected one connected topological disk")
    cross = np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]])
    area = np.linalg.norm(cross, axis=1) / 2
    if np.any(area <= 0) or len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri):
        raise ValueError("Degenerate or duplicate faces")
    normal = cross.sum(axis=0)
    normal /= np.linalg.norm(normal)
    depth = float((x[loop].mean(axis=0) - x.mean(axis=0)) @ normal)
    if depth <= 0:
        raise ValueError(
            "Expected supplied normals facing the concavity; inspect geometry"
        )
    lengths = np.array([np.linalg.norm(x[a] - x[b]) for a, b in edges])
    report = dict(
        vertices=len(x),
        triangles=len(tri),
        edges=len(edges),
        boundary_vertices=len(loop),
        boundary_loops=1,
        euler_characteristic=1,
        components=1,
        area_mm2=float(area.sum()),
        bounds_mm=[x.min(0).tolist(), x.max(0).tolist()],
        edge_length_mm=[float(lengths.min()), float(lengths.max())],
        mean_normal=normal.tolist(),
        boundary_to_mesh_centroid_depth_mm=depth,
        welding="exact coordinate equality",
        pressure_sign=-1,
    )
    return x * 1.0e-3, tri.astype(np.int32), np.array(loop, dtype=np.int32), report
