# Physics walkthrough

## Initial PatchWarp membrane

Read `mesh/stl.py`, `elements/triangular_membrane.py`, then `analysis/explicit_patch.py`.

The STL reader welds identical coordinates, checks manifold edges/winding, walks
one perimeter loop and requires a connected disk. It converts millimeters to
meters and checks the concavity convention used by this fixture. It is a bounded
benchmark reader, not a general CAD repair/orientation service.

Each triangle has a 2D rest basis. Its current edges produce a 3x2 deformation
gradient F. Green strain is E=(F^T F-I)/2; the StVK membrane energy is
W=lambda_ps/2 tr(E)^2+mu E:E, integrated with reference area and thickness.
`lambda_ps=E_young*nu/(1-nu^2)` is the plane-stress coefficient. Force is minus
the energy gradient. Pressure uses the current oriented area vector, divided
equally among the triangle's three nodes; the negative sign pushes away from
the concavity. The boundary update restores reference coordinates and zero velocity.

The runner uses damped semi-implicit Euler, its original altitude-based step rule,
a smooth pressure ramp and CUDA-graph batches. It reports residual, speed and
kinetic energy instead of claiming that finite runtime proves equilibrium.
At 40 mmHg it uses synthetic E=10 MPa, nu=.3, h=.5 mm, rho=1000 kg/m3 and damping
2000/s. These are not clinical ePTFE values. Geometry is an explicitly synthetic
paraboloid, not the original anatomical patch. There is no bending/contact/chord.

## CardioFEM material point

Read `materials/lee_sacks.py`. F is 3x3, J=det(F)>0, I1=F:F. The coupled energy is

W=c0[0.5(I1-3)-ln J] + c1/2[exp(c2(I1-3)^2)-1] + k/2(ln J)^2.

PK1 is dW/dF; Cauchy stress is P F^T/J. `lee_sacks_tangent_action` computes the
exact directional derivative of PK1. The host `evaluate` helper checks admissibility,
launches the kernel and reads small arrays for diagnostics. It is intentionally
not the production device-resident quadrature interface for a large solve.

The standalone material example uses synthetic parameters labeled Pa and a
prescribed F; it does not solve equilibrium. The inherited tests check energy
and stress derivatives, rotation, invalid states and CPU/GPU agreement.

## New Warp FEM affine solid verification

Read `backends/warp_fem_elasticity.py`. This is small-strain isotropic elasticity with
sigma=lambda tr(eps)I+2mu eps, E=1 MPa, nu=.3 on a 10 mm cube with eight cells.
All boundary nodes receive a nonzero affine displacement; the interior must recover
that field. Both rows and columns are projected; the RHS accounts for prescribed
values. Positive elastic moduli, full boundary constraints and this regular grid
remove rigid modes and justify CG for this particular linear problem.

This check exercises actual FEM assembly and a solve. It is distinct from the
historical neo-Hookean tension deck in `examples/solid_reference/` and does not
validate large deformation, mixed incompressibility, shell locking or contact.

## Neo-Hookean material points

`materials/neo_hookean.py` exposes two laws as FP64 Warp functions. Call them
from a kernel or a `warp.fem` integrand with dimensionless F and moduli in Pa.
The caller owns deformation admissibility and the equilibrium solve.

| Functions | Formulation and intended use |
|---|---|
| `febio_energy`, `febio_pk1`, `febio_tangent_action` | FEBio compressible 3D law for the synthetic solid target. Energy is μ/2(F:F−3)−μ log J+λ/2(log J)². The tangent returns dP[F][H] in the reference frame. |
| `stable_energy_2d`, `stable_pk1_2d`, `stable_tangent_action_2d` | The 2D stable variant from the NVIDIA Warp example, with energy referenced to zero at identity and an exact derivative of PK1. The coefficient of (J−1)² is (λ+μ)/2. |
| `stable_gauss_newton_action_2d` | The NVIDIA example's approximation. It drops the term proportional to the second derivative of J. |

Use `lame_from_young_poisson` to compute the Lamé moduli. For the 3D FEBio law,
convert PK1 to Cauchy stress with σ = P Fᵀ/J. Material-point finite differences
verify both exact derivatives. The synthetic FEBio tension deck still needs a
finite-strain solid equilibrium driver.

- [NVIDIA Warp mixed elasticity example](https://github.com/NVIDIA/warp/blob/main/warp/examples/fem/example_mixed_elasticity.py)
- [FEBio Neo-Hookean implementation](https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FENeoHookean.cpp)

## FEBio isotropic elastic material

The native PatchWarp quad verification uses FEBio's `isotropic elastic`
material. Its full 3D StVK functions live in `materials/stvk.py`.
They compute Green–Lagrange strain, reference-volume energy, PK2, PK1 and
the directional derivative of PK1. A director shell or volume element can
call the same material point functions with its own deformation gradient.
The initial triangle example uses its existing plane-stress reduction.
See the patch case records for the verified modulus, Poisson ratio,
thickness and density. StVK is a benchmark law whose large-compression
response should be assessed independently for any new application.

[Source for FEBio's strain energy and stress](https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEIsotropicElastic.cpp)
