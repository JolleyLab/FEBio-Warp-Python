# FEBio-Warp-Python

A component-by-component repository scaffold for GPU cardiovascular mechanics built in concert with the FEBio (www.febio.org) team.
The [architecture map](docs/ARCHITECTURE.md) shows where to add mesh types,
materials, elements, assembly, and individual solvers. Each planned component
has its own Python file and a short description of what belongs there.

The scaffold includes
an initial PatchWarp membrane example, CardioFEM material-point mechanics and a
working Warp FEM solid-elasticity verification example in one installable package.
It contains synthetic geometry only and imports Warp as an external dependency.

## What works today

| Example | What it demonstrates | What it does not establish |
|---|---|---|
| Pressurized patch | StVK membrane, fixed perimeter, follower pressure, GPU time stepping and viewer | Current native-quad shell or clinical ePTFE behavior |
| Material point | CardioFEM Lee–Sacks energy, PK1/Cauchy stress, exact tangent action | A spatial solid or full valve solve |
| Warp FEM solid | 3D vector space, elasticity forms, sparse assembly, prescribed BCs and CG | Nonlinear tissue, contact or FEBio solid agreement |
| FEBio solid reference | Original synthetic hex8 tension input and final displacement excerpt | A recovered CardioFEM global solid solver |

## Install

From this folder, using Python 3.11 (tested) or a supported Python >=3.10:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

Windows: use `.venv\Scripts\python.exe` instead. Warp is pinned to **1.19.0**.
For wheel/driver requirements see [NVIDIA installation](https://nvidia.github.io/warp/v1.19/user_guide/installation.html).
CPU material/FEM checks do not require CUDA. The explicit patch requires CUDA.

## Run the examples (all commands from this repository root)

```bash
# A prescribed material state; output stresses are explicitly labeled Pa.
.venv/bin/python examples/material_point/run.py

# A manufactured affine-displacement solid test, with a known exact solution.
.venv/bin/python examples/fem_solid.py --device cpu
# Optional GPU verification:
.venv/bin/python examples/fem_solid.py --device cuda:0

# Initial membrane example on a generated, non-anatomical patch:
.venv/bin/python examples/patch_pressurization/generate_mesh.py
.venv/bin/python examples/patch_pressurization/run.py --mesh synthetic_patch.stl --output outputs/patch --device cuda:0 --pressure-mmhg 40 --young-mpa 10 --duration .05 --ramp .01
.venv/bin/python -m warp_febio.visualization.patch outputs/patch/state.npz
```

The patch saves a PNG even in headless mode. Interactive plotting needs a display.
Outputs and kernel caches are ignored by Git and excluded from the upload ZIP.
Small reference records and reproducible checks live in `validation/`.

## FEBio comparison settings

Each example keeps its settings beside the code or reference deck.

| Example folder | Archived settings and evidence |
|---|---|
| [Patch pressurization](examples/patch_pressurization/README.md) | Original membrane and shell comparisons, including the native quad, A2, and four chords |
| [CardioFEM valve](examples/cardiofem_valve/README.md) | Lee–Sacks shell, chord curves, contact, loading, solver settings, and matched-state comparison records |
| [Solid reference](examples/solid_reference/README.md) | Neo-Hookean FEBio tension target and the earlier Newton/Warp leaflet comparison |

## Read and extend

1. [Architecture and status](docs/ARCHITECTURE.md)
2. [Physics walkthrough](docs/PHYSICS.md)
3. [Warp FEM inventory and public imports](docs/WARP_FEM.md)
4. [Development and prototype provenance](docs/DEVELOPMENT.md)
5. [Validation scope](validation/README.md)
6. [Verification harness outline](docs/VERIFICATION.md) (draft for discussion)

The examples run from this repository and use synthetic inputs. See [merge/upload instructions](MERGE_NOTES.md).

## License

FEBio-Warp-Python is released under the [Apache License 2.0](LICENSE), the same license as
[NVIDIA Warp](https://github.com/NVIDIA/warp). Contributions are accepted under the same terms.

[FEBio](https://github.com/febiosoftware/FEBio) is distributed separately under the MIT license.
Material in this repository that derives from FEBio, including the reference inputs under
`references/febio/`, keeps its attribution to the FEBio project.
