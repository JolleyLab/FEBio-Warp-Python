# Warp FEBio architecture

The layout follows two ideas from established FEM software. MFEM separates mesh,
finite-element spaces, assembly, and linear algebra. FEBio organizes mechanics
around domains, materials, loads, constraints, and analysis steps. This project
uses those boundaries with public NVIDIA Warp APIs.

- [MFEM code organization](https://mfem.org/code-overview/)
- [FEBio solid mechanics module](https://febiosoftware.github.io/febio-docs/features/modules/module_solid/)
- [Warp FEM Toolkit](https://nvidia.github.io/warp/v1.19/domain_modules/fem.html)

```text
src/warp_febio/
  mesh/              triangle, quadrilateral, tetrahedral, hexahedral, structured_grid, topology
  discretization/    dof_map, function_space, quadrature
  elements/          triangular_membrane, linear_elastic_solid, director_shell, hyperelastic_solid, axial_spring, beam
  materials/         lee_sacks, stvk, neo_hookean, hgo, nitinol
  loads/             follower_pressure, nodal_force
  constraints/       dirichlet, multipoint
  contact/           search, enforcement
  assembly/          residual, tangent
  solvers/linear/    cg, gmres, preconditioners
  solvers/nonlinear/ newton, line_search
  analysis/          explicit_patch, static, implicit_dynamics, procedure
  backends/          warp_runtime, warp_fem_geometry, warp_fem_elasticity
  sensitivity/       equilibrium, trajectory
  io/                febio, results
  visualization/     patch
```

Files labeled **Planned component** are editable Python placeholders. Their
docstrings give the intended responsibility. The working entry points appear
in the root README. The affine verification driver in
`backends/warp_fem_elasticity.py` uses the shared solid form in
`elements/linear_elastic_solid.py` and the boundary projection in
`constraints/dirichlet.py`. The driver owns the prescribed affine field,
material values, and error check. It calls Warp CG directly.

Keep reusable physics and numerics in `src/warp_febio/`. Keep input decks,
benchmark settings, reference outputs, and comparisons in `examples/` and
`validation/`. When bringing over a proven implicit solver, place its Newton
iteration and line search under `solvers/nonlinear/`, its linear algebra
and preconditioners under `solvers/linear/`, and its load stepping under
`analysis/`. Keep patch and valve parameters in the corresponding case.

## Where to add a component

| Component | Files that own the work |
|---|---|
| A tetrahedral vessel domain | `mesh/tetrahedral.py`, `discretization/`, `elements/hyperelastic_solid.py` |
| A new constitutive law | `materials/` for stress, tangent, and material history |
| A native quad director shell | `mesh/quadrilateral.py`, `elements/director_shell.py`, and `discretization/quadrature.py` |
| A pressure continuation strategy | `loads/follower_pressure.py` for the force and its derivative, `analysis/static.py` for load increments |
| A nonlinear solve | `solvers/nonlinear/newton.py` for iteration and convergence, `solvers/linear/` for each linear method |
| A deployment procedure | `analysis/procedure.py` for stages, `contact/` for interactions, and `materials/nitinol.py` for state |
| An adjoint | `sensitivity/` using the constrained tangent and transpose actions |

Use `contracts.py` as the starting point for residual and tangent interfaces.
Define physical units, DOF ordering, and state ownership before wiring a new
domain into a solver. The first extension can use the synthetic verification
cases to check assembly and constraints before attempting a larger model.
