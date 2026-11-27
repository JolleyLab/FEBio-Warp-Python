# FEBio-Warp-Python

GPU-accelerated, differentiable finite element mechanics for cardiovascular simulation,
written in Python on [Warp](https://github.com/NVIDIA/warp) and built in concert with the
[FEBio](https://febio.org) team. The formulations follow FEBio. The architecture is designed
for the GPU and for differentiation.

FEBio-Warp is intended to stand next to FEBio, not to replace it. FEBio remains the place for
broad constitutive modeling and general biomechanics. FEBio-Warp targets a focused set of
applications where speed and gradients matter.

## Project status

This project is at an early stage. Interfaces will change, and most areas listed below are in
design or draft. Nothing here is released.

Development has now moved to a design-led process. Each area is specified in a design note,
reviewed by the FEBio and Warp developers involved, implemented, and verified against FEBio
reference problems where one exists. The pilot work is preserved in [`pilots/`](pilots/README.md) and at the
[`pilot-2026-11`](https://github.com/JolleyLab/FEBio-Warp-Python/tree/pilot-2026-11) tag.

## Driving applications

The first targets set the scope. Capabilities are added because one of these needs them.

| Application | Needs |
|---|---|
| Atrioventricular valve closure and repair | Shell leaflets, fiber-reinforced hyperelastic tissue, chordae, pressure loading, leaflet self-contact |
| Self-expanding nitinol devices (transcatheter pulmonary valve replacement, septal occluders) | Beam frames, nitinol material, device-to-vessel contact, hyperelastic vessel wall |
| Patch and material verification problems | Small problems with known or FEBio reference solutions, used to verify each component |

## Development areas

Status as of November 2026. "Design" means a design note is in progress. "Draft" means code
exists on a branch or draft pull request and is under review. "Not started" means the package
directory exists as a place for the work and holds no code yet.

| Area | Package | Scope for the first targets | Status |
|---|---|---|---|
| Materials | `materials/` | Common material interface; neo-Hookean, HGO, Ogden, Lee-Sacks, nitinol | Draft ([PR #2](https://github.com/JolleyLab/FEBio-Warp-Python/pull/2)), under redesign |
| Contact | `contact/` | Potential-based contact for leaflets and devices; sliding contact later | Draft ([PR #1](https://github.com/JolleyLab/FEBio-Warp-Python/pull/1)) |
| Assembly | `assembly/` | Full assembly of residual and tangent on the GPU | Design |
| Elements | `elements/` | Shells, solids, beams, springs | Not started |
| Mesh and discretization | `mesh/`, `discretization/` | Mesh types, degree-of-freedom maps, quadrature | Not started |
| Loads and constraints | `loads/`, `constraints/` | Follower pressure, nodal loads, prescribed and multipoint constraints | Not started |
| Solvers | `solvers/` | Newton with line search; iterative linear solvers and preconditioners | Not started |
| Analysis | `analysis/` | Static and implicit dynamic steps, load stepping, staged procedures | Not started |
| Sensitivity | `sensitivity/` | Gradients through equilibrium and through trajectories | Not started |
| Central model, input and output | `model/`, `io/` | One model object holding what a FEBio input defines; FEBio input (`.feb`) and plot (`.xplt`) files read and written here; VTKHDF output and a VTK adapter for 3D Slicer | Draft ([PR #5](https://github.com/JolleyLab/FEBio-Warp-Python/pull/5)) |
| Verification | `tests/`, `references/` | FEBio-Warp run on FEBio input files and compared with pinned FEBio references; cases from the FEBio test suite and the driving applications | Draft ([PR #4](https://github.com/JolleyLab/FEBio-Warp-Python/pull/4)), tabled until input and output |

Design notes live in [`docs/design/`](docs/design/README.md).

The four drafts are reviewed separately. They are also merged, and made consistent where they
meet, on the branch
[`mattjolley/integration-base`](https://github.com/JolleyLab/FEBio-Warp-Python/tree/mattjolley/integration-base);
its note `docs/design/integration.md` says what that involved and what is left open.

### Relation to FEBio's structure

The package layout under `src/warp_febio/` follows FEBio's separation of concerns.

| FEBio-Warp | FEBio counterpart |
|---|---|
| `mesh/`, `discretization/` | FECore mesh, degrees of freedom, element traits |
| `elements/` | FEBioMech domains |
| `materials/` | FEBioMech materials |
| `loads/`, `constraints/`, `contact/` | FEBioMech loads, constraints and contact interfaces |
| `assembly/`, `solvers/` | FECore solver interfaces, NumCore linear solvers |
| `analysis/` | FECore analysis steps |
| `io/` | FEBioXML, FEBioPlot |
| `sensitivity/` | Added here |

## How code enters the package

1. A design note in `docs/design/` states the formulation, the FEBio source it follows, and
   how it will be verified.
2. A draft pull request implements it and is reviewed.
3. It is verified against FEBio reference problems where one exists, in double precision.

Exploratory work is welcome and belongs in `pilots/` until it has been through these steps.
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Pilot work

The initial prototypes were contributed by [@aslanseda9](https://github.com/aslanseda9): an
explicit membrane patch solver, Lee-Sacks and neo-Hookean material-point functions, and a solid
elasticity check built on `warp.fem`. They were written to explore the approach, not as verified
implementations, and each one lists what it does and does not establish. They remain available
and runnable in [`pilots/`](pilots/README.md). They are not maintained and are not a dependency
of the package.

The FEBio inputs and results assembled during the pilot phase are kept in `references/febio/`.
They will be reviewed for inclusion in the verification harness and should not be read as
verified benchmarks until then.

## Install

The project uses [uv](https://docs.astral.sh/uv/) for environments and
[Ruff](https://docs.astral.sh/ruff/) for linting and formatting, as Warp does.

```bash
uv sync --extra dev
uv run -m unittest discover -s tests -v
```

`uv sync` creates `.venv` with the versions recorded in `uv.lock`. The `dev` extra installs
everything the tests use, including Warp. To use the package, the `warp` extra
(`uv sync --extra warp`, or `pip install -e ".[warp]"`) installs Warp, which compiling and
running a model needs. Without it, `pip install -e .` installs only the model and the FEBio
file readers and writers (`warp_febio.model`, `warp_febio.io`), which depend on NumPy alone
and can be used inside 3D Slicer.

Python 3.10 or later; 3.11 is the tested version and the one uv selects from
`.python-version`. Warp is pinned to 1.19.0; see the
[Warp installation guide](https://nvidia.github.io/warp/v1.19/user_guide/installation.html).

See [CONTRIBUTING.md](CONTRIBUTING.md) for formatting and the pre-commit hooks.

## Acknowledgments

FEBio-Warp builds on the formulations, documentation and test suites of
[FEBio](https://github.com/febiosoftware/FEBio), developed at the University of Utah, and on
[Warp](https://github.com/NVIDIA/warp).

## License

FEBio-Warp-Python is released under the [Apache License 2.0](LICENSE), the same license as
[Warp](https://github.com/NVIDIA/warp). Contributions are accepted under the same terms.

[FEBio](https://github.com/febiosoftware/FEBio) is distributed separately under the MIT license.
Material in this repository that derives from FEBio, including the reference inputs under
`references/febio/`, keeps its attribution to the FEBio project.
