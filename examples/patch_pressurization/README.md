# Initial pressurized membrane

Run the commands in the root README from the root directory. The generator writes
`synthetic_patch.stl` in the working directory: 81 vertices, 128 triangles, coordinates
in millimeters. Positive z normals face the concavity; pressure acts away from it.

`run.py` is a thin CLI. Read `src/warp_febio/analysis/explicit_patch.py`
for preparation/time stepping and `elements/triangular_membrane.py` for mechanics.
This is the initial explicit membrane, not the optimized implicit quad shell.
The original repository is not required for this synthetic case; reproducing the
original patch geometry requires its separate mesh and prototype configuration.


## Original PatchWarp comparisons against FEBio

[Case settings and comparison records](febio_cases.json) contains the original
run parameters, extracted FEBio deck settings, and the archived comparison
metrics. Source paths are relative to the indicated original project, and SHA-256
hashes identify the files from which each record was taken.

- The membrane and hinge studies used triangular meshes, fixed perimeter
  translations, and synthetic plane-stress StVK. The reports distinguish the
  original STL from the FEBio-derived STL and identify load-history comparisons.
- The native-quad static check used full 3D StVK, 1 MPa, ν=0.3, 0.5 mm thickness,
  1000 kg/m³, and 13,300 Pa follower pressure. The top perimeter was fixed and
  bottom DOFs were free. This case tested pressure alone and a 0.1 N A2 force.
- The native-quad four-chord case used two fixed PM anchors and bilateral
  springs at 30 N/mm. Its archived reference was dynamic at t=0.1 s. The later
  input audit recovered and verified the original deck. The prepared static deck
  has its own settings record and is still awaiting a static results comparison.

The in-folder runner reproduces the generated synthetic membrane case. The
historical cases need the original PatchWarp solver, model files and reference
fields. Mesh coordinates and full simulation results stay in the original project.

The native quad case's reusable 3D material functions are in
[`materials/stvk.py`](../../src/warp_febio/materials/stvk.py). The original
quad verification uses E = 1 MPa and ν = 0.3. Thickness, density and
pressure remain case settings in `febio_cases.json`.
