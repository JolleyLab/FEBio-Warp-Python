# Warp 1.19 public API inventory

Install `warp-lang==1.19.0`; do not copy Warp source into this repository.
Importing reads the installed package and does not download it.

```python
import warp as wp
import warp.fem as fem
from warp.optim.linear import cg
from warp.sparse import bsr_mv
```

| Component | Where it belongs | Current status |
|---|---|---|
| `Grid3D`, vector `make_polynomial_space` | Discretization/backend | Used by the solid check |
| `Cells`, `BoundarySides`, test/trial fields | Domain/forms | Used |
| `integrand`, `integrate`, `D`, `div` | Element weak forms | Used |
| `project_linear_system` | Constraint/assembly adapter | Used for nonzero Dirichlet values |
| BSR, `bsr_mv`, `cg` | Sparse operator/linear backend | Used; checks actual residual |
| `Tetmesh`, `Hexmesh`, quadrature choices | Future vessel/device solids | Not yet integrated |
| `Quadmesh3D`, `Trimesh3D` | Surface geometry/field adapters | Not director-shell mechanics |
| `NonconformingField` and lookups | Transfer/diagnostics | Future; not monolithic contact assembly |
| `TemporaryStore`, mapped BSR updates | Backend memory/cache policy | Future measured optimization |
| GMRES and alternative preconditioners | General operator solver | Future; select from actual Jacobian |
| Tape / autograd diagnostic tools | Local derivatives and tests | Future adjunct to analytic mechanics |
| Core Mesh/BVH queries | Contact search | Future; contact law still custom |
| Tiles | Kernel optimization | Only after profiling |
| Geometry extraction / interop / rendering | Preprocessing, optimization and viewer adapters | Optional future features |

The toolkit guide explains workflows; the API reference defines exact signatures:
- https://nvidia.github.io/warp/v1.19/domain_modules/fem.html
- https://nvidia.github.io/warp/v1.19/api_reference/warp_fem.html
- https://nvidia.github.io/warp/v1.19/api_reference/warp_sparse.html
- https://nvidia.github.io/warp/v1.19/api_reference/warp_optim.html

Curved geometry, batching, partitions and adaptive grids are future opportunities.
They do not automatically supply unstructured remeshing, shell thickness integration,
contact, nitinol transformation or cardiovascular FSI. Keep those physical contracts
explicit rather than inferring capability from mesh/field class names.
