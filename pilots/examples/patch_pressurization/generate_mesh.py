"""Generate an explicitly synthetic 20 x 20 mm curved patch, no anatomical data."""

from pathlib import Path

n = 9
vertices = [
    (x, y, (x * x + y * y) / 100.0)
    for y in [-10 + 20 * j / (n - 1) for j in range(n)]
    for x in [-10 + 20 * i / (n - 1) for i in range(n)]
]
faces = []
for j in range(n - 1):
    for i in range(n - 1):
        a = j * n + i
        faces.extend(((a, a + 1, a + n + 1), (a, a + n + 1, a + n)))
lines = ["solid synthetic_patch_mm"]
for face in faces:
    lines += ["facet normal 0 0 0", "outer loop"]
    lines += ["vertex %.12g %.12g %.12g" % vertices[k] for k in face]
    lines += ["endloop", "endfacet"]
lines += ["endsolid synthetic_patch_mm"]
Path("synthetic_patch.stl").write_text("\n".join(lines) + "\n")
print("Created synthetic_patch.stl: 81 vertices, 128 triangles; coordinates in mm")
