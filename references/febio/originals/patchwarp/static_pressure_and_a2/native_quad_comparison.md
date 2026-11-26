# Native quad StVK shell: pressure and static A2 validation

The separate GPU-native quad shell reproduces the matched FEBio static patch
fields to approximately the precision of FEBio's saved nodal displacement
values. It removes the hinge model's 45.7% A2 compliance deficit for this case.
The hinge formulation and its physical settings are retained unchanged.

| Quantity | PatchWarp native quad | FEBio static reference |
|---|---:|---:|
| Pressure-only maximum displacement | 1.594439140 mm | 1.594439146 mm |
| Pressure-only surface-integrated vector error | 0.00000180583% | reference |
| A2 incremental displacement x | −0.02424109995 mm | −0.02424111404 mm |
| A2 incremental displacement y | 0.17249514793 mm | 0.17249518633 mm |
| A2 incremental displacement z | 0.50145115742 mm | 0.50145116448 mm |
| A2 directional compliance | 5.01581905288 mm/N | 5.01581912144 mm/N |
| Compliance difference | −0.00000136688% | reference |
| Incremental field error within 2 mm of A2 (103 nodes, nodal L2) | 0.0000238905% | reference |
| Full-surface incremental displacement error | 0.0000549382% | reference |

The pressure-only maximum nodal vector difference is 8.95658e-8 mm. Native
FEBio node IDs and quad connectivity match exactly; no projection is used.
The largest input-coordinate difference from XPLT is 6.97798e-8 mm, consistent
with its coordinate precision. Surface integrals use the existing bilinear
quad reference-area methodology with 8×8 Gauss quadrature, interpolating the
vector differences before squaring. A2 is FEBio node 1868; compliance is
`dot(u_loaded-u_pressure, force/0.1 N)/(0.1 N)`.

For continuity with earlier reports, the same new static pressure solution
has **0.1501633%** error against the original **dynamic** no-chord FEBio output
(maximum 1.598130463 mm). This is not the matched static validation pair.
Previous hinge errors of 6.98279% on the FEBio-derived STL and 7.52248% on
patch.stl used that older reference and a slightly different pressure. Do not
present their ratio to the static error as a controlled formulation-only metric.

## Mechanics and integration

- 1,863 native quad4 elements, 1,932 physical top nodes, 3,864 top/bottom
  translational nodes (11,592 scalar DOFs before constraints).
- Extensible directors represented by independent top and bottom displacement.
  Reference bottom positions are top minus corner-normal-averaged thickness
  directors, without renormalizing the averaged director.
- Full 3D StVK energy `W = lambda/2 tr(E)^2 + mu E:E`,
  `E = (F^T F-I)/2`; 2×2×2 Gauss integration. Full material and geometric
  tangent, including director coupling, transverse shear and thickness stretch.
- E = 1 MPa, nu = 0.3, thickness = 0.5 mm, density = 1000 kg/m³.
  SI units in the solver; mm–N–s/MPa in the source FEBio input.
- Follower pressure 13,300 Pa = 0.0133 MPa, matching the static reference;
  2×2 top-surface quadrature and its exact tangent. Complete top perimeter
  fixed (136 nodes); bottom/director boundary DOFs free, as in FEBio.
- Static pressure continuation in ten increments, followed by four A2 force
  increments. No chords. Force vector [0.01132693660, 0.00374357172,
  0.09928588106] N is fixed in global coordinates.
- Newton absolute free residual tolerance 1e-7 N; CG relative 0.01,
  absolute 1e-12, maximum 500 iterations. No tolerance relaxation.

The constitutive residual and exact analytic blocks are independently written
PatchWarp code. Cached sparse GPU tangent application, conditional-graph CG,
Newton/Armijo logic and convergence checks reuse PatchWarp infrastructure.
The preconditioner groups four neighboring top/bottom pairs per 24-DOF block,
retaining strong through-thickness coupling. Its material-only approximation
changes search directions, not the converged equations. The only shared-solver
change is an assembly hook preserving the original hinge/membrane operations.

## Runtime

Warmed wall times on Quadro RTX 8000, CUDA, double precision:

| Solve | Wall time | Newton iterations | CG iterations |
|---|---:|---:|---:|
| Pressure continuation | 0.37923 s | 42 | 3,336 |
| Additional A2 loading from pressure equilibrium | 0.12794 s | 16 | 1,049 |
| Combined | 0.50717 s | 58 | 4,385 |

Timing includes residual evaluation, tangent/preconditioner construction,
GPU CG, line search and synchronization; excludes mesh preparation, kernel
compilation, warmup, result serialization and validation. These are one warmed
run, not a statistical hardware benchmark. Final residual norms are
2.73758e-9 N (pressure) and 3.56374e-9 N (loaded).

FEBio logs report 1.80492 s for static pressure and 1.69785 s for the full
loaded case, including I/O. Those runs were on another machine and the loaded
FEBio run includes the full pressure ramp, so they are context, not a speedup
ratio. Prior CardioFEM timings use different materials/problems; no controlled
CardioFEM speedup is claimed.

## Differentiability and validation

Native `QuadStaticSensitivity` supports E, nu and pressure tangents and adjoints
through the unshifted exact equilibrium Hessian. Full-patch central-difference
relative field errors: E 5.36e-7, nu 2.75e-6, pressure 1.42e-6. Tangent/adjoint
objective derivatives agree within 9.88e-9 relative. True linear residuals are
checked. These derivative-only verification solves use 1e-10 N equilibrium
accuracy to suppress finite-difference noise; benchmark settings stay unchanged.
Thickness/shape derivatives are not yet exposed for this new formulation.

GPU component tests pass: rigid objectivity, affine 3D StVK energy, force/energy
and tangent finite differences, cached/matrix-free equality, symmetry, material
parameter partials, mass integration, and projected follower-pressure tangent
with multiple free nodes. Independent review is in `independent_review.md`,
with source-physics inspection in `../../diagnostics/native_quad_shell_review.md`.

This establishes agreement for the StVK static patch and A2 diagnostic, not
arbitrary materials, dynamics, contact or chords. Full integration can lock on
other thin-shell meshes; no reduced-integration stabilization was introduced.
Use this as the high-fidelity native-quad base while preserving the hinge
baseline. A matched chord test is the next extension, not part of this run.

## Reproduce

From the PatchWarp repository:

```bash
.venv/bin/python -m patchwarp.quad_benchmark
.venv/bin/python validation/comparisons/native_quad_shell/compare.py
.venv/bin/python -m patchwarp.quad_derivative_check
```

The solver is standalone. The comparison script reuses the existing read-only
FEBio XPLT reader from the sibling NewtonValveDev checkout; this analysis-only
dependency is absent from the simulation and differentiation implementation.
