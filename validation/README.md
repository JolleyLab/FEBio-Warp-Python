# Validation records and limits

- `patch_pressurization/`: old Warp1.12.1 scalar/field reference and new Warp1.19
  synthetic-patch run. `compare_patch.py` computes nodal vector errors between
  exactly the same generated mesh and run settings; this is not a FEBio comparison.
- `fem_solid/`: analytical affine-field/residual checks on CPU/GPU. Compilation
  is included in reported wall times; do not interpret them as warm performance.
- `material_point/`: local constitutive tests and sample response only.
- `solid_reference/`: raw final FEBio displacement excerpt and provenance;
  no new stress/strain semantics or global solid agreement claim.

No anatomical mesh, patient fields or historical simulation trajectories are
included. Reference numbers are numerical verification, not clinical validation.

- `febio_suite/`: survey of the FEBio test suite against what FEBio-Warp can
  run. See `febio_suite/README.md`.
- `cases/`: small generated FEBio problems for the verification harness and
  the script that runs and checks them. See `cases/README.md`.
