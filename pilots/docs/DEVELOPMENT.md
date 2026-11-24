# Development and provenance

The initial PatchWarp membrane/mesh/viewer and CardioFEM constitutive functions
were reorganized into one namespace at the user's explicit request. Original
repositories remain unchanged. Edits expand compressed formatting, document units
and intermediate mechanics, update import paths and change the example GPU default
from cuda:1 to cuda:0. No physical coefficients or constitutive equations were
intentionally changed. `validation/source_provenance.json` records original hashes.

| Source | Original environment | This folder |
|---|---|---|
| Initial PatchWarp membrane | Python3.9, Warp1.12.1 | Python3.11, Warp1.19.0 tested on synthetic patch |
| CardioFEM material tests | Python3.11, Warp1.17.0 | Warp1.19.0 checks recorded |
| New Warp FEM affine solid | No prototype counterpart | Warp1.19.0 manufactured-solution check |

This does not migrate the native-quad shell, whole valve, contact or all old runs.
The historical solid input is included as a future target; no original CardioFEM
global solid solver was found. Do not compare different cases as equivalent.

## Workflow

1. Add one physics change and name its independent verification case.
2. Keep units and quadrature/state locations explicit in outputs.
3. Test energy/force/tangent consistency before adding solver optimizations.
4. Keep exact derivatives separate from approximate iteration/preconditioner matrices.
5. Save settings and small reference metrics; keep trajectories and data outside Git.

`python -m unittest discover -s tests -v` runs bounded material tests. The FEM
example performs its own exact-field and residual assertions. Patch migration
comparison is provided by `validation/compare_patch.py` after running the example.

To rebuild the GitHub ZIP: `python tools/package_upload.py`. It whitelists source,
docs, examples, tests, tools and small validation files; excludes caches, outputs,
virtual environments, private data and generated metadata. It checks browser limits.
The package imports neither `patchwarp` nor the original `cardiofem` package.
