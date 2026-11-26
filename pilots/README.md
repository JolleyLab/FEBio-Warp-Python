# Pilot work

This directory holds the prototypes that started FEBio-Warp-Python in November 2026, contributed
by [@aslanseda9](https://github.com/aslanseda9). They showed that the approach was feasible in
Warp and assembled the first FEBio comparison cases.

The pilots were written to explore, not as verified implementations. The package in
`src/warp_febio/` is now developed from reviewed design notes and does not import anything from
here. The pilots are kept as they were, runnable, for reference. They are not maintained and are
not part of continuous integration.

The repository as it stood before this reorganization is preserved at the
[`pilot-2026-11`](https://github.com/JolleyLab/FEBio-Warp-Python/tree/pilot-2026-11) tag.

## What each pilot shows

| Pilot | What it demonstrates | What it does not establish |
|---|---|---|
| Pressurized patch | StVK membrane, fixed perimeter, follower pressure, GPU time stepping and viewer | Current native-quad shell or clinical ePTFE behavior |
| Material point | CardioFEM Lee-Sacks energy, PK1 and Cauchy stress, exact tangent action | A spatial solid or full valve solve |
| Warp FEM solid | 3D vector space, elasticity forms, sparse assembly, prescribed boundary conditions and CG | Nonlinear tissue, contact or FEBio solid agreement |
| FEBio solid reference | Original synthetic hex8 tension input and final displacement excerpt | A recovered CardioFEM global solid solver |
| CardioFEM valve records | Archived settings and comparison records for the valve pressurization case | Full valve agreement with FEBio |

## Layout

```text
pilots/
  warp_febio_pilot/   the pilot code, in its original package layout
  examples/           patch_pressurization, material_point, fem_solid.py, solid_reference, cardiofem_valve
  tests/              material tests (Lee-Sacks, neo-Hookean, StVK)
  validation/         recorded results and the patch comparison script
  docs/               the pilot architecture, physics, Warp FEM and development notes, as written
```

The code was moved from `src/warp_febio/` to `pilots/warp_febio_pilot/` with the same
subpackage paths. The only edit is the import name, `warp_febio` to `warp_febio_pilot`. File
history follows each file (`git log --follow`).

The pilot documents in `docs/` describe the repository as it was and refer to the old paths.

## Running the pilots

With the package environment active and `matplotlib` installed
(`pip install -e ".[pilots]"` from the repository root):

```bash
cd pilots
export PYTHONPATH=.

# Material tests
python -m unittest discover -s tests -v

# Prescribed material state; stresses reported in Pa
python examples/material_point/run.py

# Manufactured affine-displacement solid check
python examples/fem_solid.py --device cpu

# Membrane patch on a generated, non-anatomical mesh (requires CUDA)
python examples/patch_pressurization/generate_mesh.py
python examples/patch_pressurization/run.py --mesh synthetic_patch.stl --output outputs/patch \
    --device cuda:0 --pressure-mmhg 40 --young-mpa 10 --duration .05 --ramp .01
python -m warp_febio_pilot.visualization.patch outputs/patch/state.npz
```

## Related reference data

The FEBio inputs and results gathered during the pilot phase are in
[`references/febio/`](../references/febio/). They are FEBio runs, not pilot code, and are being
reviewed for the verification harness.
