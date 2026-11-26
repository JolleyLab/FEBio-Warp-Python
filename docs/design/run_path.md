# Run path: from a compiled model to a result

Draft v0.1 for discussion, November 26, 2026. For Steve Maas, Jeff Weiss, Ryan Black and the Warp team. The code is a local template on branch `mattjolley/run-path-template`, stacked on `mattjolley/integration-base`. Nothing is merged or pushed.

**Scope.** The structure that takes a model and drives it through steps to a result: unknowns, terms, the Newton step, the step driver, and gradients. It is shown on the basic example Steve suggested (one element, one material, prescribed nodal displacements, with free unknowns and a Newton solve).

**Not in scope.** New elements, loads, dynamics, contact in a run, performance. Each has a marked place to attach and no implementation.

**What it replaces.** The test fixture of pull request 5 (`compiled.py`, `analysis/static.py`), which ran the same example in one piece. The model, reader, writers, material layer and contact are used unchanged.

## 1. Summary

- **The solver sees a field table and a list of terms, and nothing else.** A domain, a load, a set of springs and a contact interface would all be terms.
- **It follows FEBio's structure.** The same passes as `FESolidSolver2`, FEBio's residual sign, FEBio's treatment of prescribed displacements, FEBio's convergence tests and time step controller.
- **It departs from FEBio in three places, each for Warp.** Unknowns are numbered by (node, variable) so every matrix block is 3x3. Each domain is a batch with flat point data. State is separate from the model.
- **Gradients are part of the structure.** A gradient with respect to material parameters costs one extra linear solve and one reverse pass of the residual, using the same kernels as the forward solve.
- **It reproduces the fixture and FEBio** on the four inputs that run today (section 6).
- **Three points need Steve's view** (section 7).

## 2. The path

```mermaid
flowchart TB
    feb([".feb"]) --> read["read_feb<br/><i>pull request 5</i>"] --> model["<b>Model</b>"]
    model --> build["<b>build_system</b><br/><i>analysis/steps.py</i>"]

    subgraph system["System, on the device"]
        fields["<b>FieldTable</b><br/>entries = (node, variable)<br/>free, fixed, prescribed<br/><i>discretization/fields.py</i>"]
        terms["<b>Terms</b><br/>BrickDomain today<br/>loads, springs, contact next<br/><i>assembly/system.py</i>"]
    end
    build --> fields & terms
    mat["Material layer<br/><i>pull request 2</i>"] --> terms

    system --> run["<b>run</b>: steps, time step controller<br/><i>analysis/steps.py</i>"]
    run --> newton["<b>solve_step</b>: Newton<br/><i>solvers/nonlinear/newton.py</i>"]
    newton --> lin["<b>solve</b>: linear<br/><i>solvers/linear/warp_linear.py</i>"] --> newton
    newton --> state["<b>State</b><br/>one per converged time"]
    state --> results(["Results, .xplt<br/><i>pull request 5</i>"])
    state --> adj["<b>parameter_gradient</b><br/><i>sensitivity/adjoint.py</i>"]
```

| Here | FEBio | What is the same | What differs |
| --- | --- | --- | --- |
| `FieldTable` | `DOFS`, `FESolver::InitEquations` | Variable names; free, fixed, prescribed | Numbered by (node, variable); constrained components masked, not removed |
| `Term` | `FEElasticDomain`, `FEModelLoad`, `FEContactInterface` | What each computes; the sign | One contract for all of them |
| `BrickDomain` | `FEElasticSolidDomain`, `FEElasticShellDomain` | Update, internal force, stiffness; integration points | Flat point data; one kernel set per material tree |
| `System` | `FEModel` as `FESolidSolver2` uses it | The three passes | Holds no application knowledge |
| `solve_step` | `FENewtonSolver::SolveStep`, `FESolidSolver2::Quasin` | Prescribed displacements; `dtol`, `etol`, `rtol`, `min_residual`, `max_refs` | Full Newton, no line search yet |
| `run` | `FEAnalysis`, `FETimeStepController` | Control settings; retries; must points | A step maps one State to the next |
| `parameter_gradient` | none | | Added |

## 3. One Newton iteration

```mermaid
sequenceDiagram
    participant N as solve_step
    participant S as System
    participant T as Terms
    participant L as Linear solve
    Note over N,T: point data is current from the previous iterate
    N->>S: tangent(state)
    S->>T: linearize, 3x3 blocks
    N->>S: mask right-hand side, constrained matrix
    N->>L: solve
    L-->>N: increment
    N->>N: apply increment by each variable's rule
    N->>S: update(state)
    S->>T: update: kinematics and material, once per point
    T-->>S: status code
    N->>S: residual(state)
    S->>T: residual
    N->>N: norms, convergence tests
```

The material is called in one place, `update`, once per integration point per iterate. The residual that gives the convergence norms and the tangent of the next iteration both read the same stored point data.

## 4. The three departures from FEBio, and why

### Unknowns by (node, variable)

An entry is one node and one FEBio variable (`displacement`, `shell displacement`), holding a 3-vector. The solution is one array of 3-vectors and the matrix has 3x3 blocks.

- Warp's sparse matrix stores fixed-size blocks, and its solvers and preconditioner work on them directly.
- The contact draft already assembles 3x3 blocks, so contact and structure share one layout.
- Each variable has an update rule. Both variables here add. A beam's `rotation` would compose through the exponential map, as FEBio's beam domain does, and the Newton step already looks the rule up.
- Constrained components stay in the system and are masked, as the fixture did. The sparsity then depends on the mesh only.

### A domain is a batch with flat point data

`BrickDomain` is one element block with one material tree. Its three kernels have no branch on element or material type.

| Call | FEBio | Kernel launch | Writes |
| --- | --- | --- | --- |
| `update` | `Update` | elements x points | Cauchy stress, 6x6 tangent, spatial shape gradients, weight, status |
| `residual` | `InternalForces` | elements x nodes | The global residual, by atomic subtraction |
| `linearize` | `StiffnessMatrix` | elements x nodes x nodes | One 3x3 block per node pair |

FEBio throws on a negative Jacobian. A kernel cannot, so `update` writes a status code per point and the host reads the worst one, using the codes of the material layer.

### State apart from the model

`State` is the solution at one time. A step produces a new one and keeps the old until the new one converges, so a failed attempt has nothing to undo, and a converged state can be handed to the adjoint afterward.

## 5. How Warp is used for the GPU and for gradients

### On the device

| Stays on the device | Crosses to the host |
| --- | --- |
| Solution, residual, point data, matrix blocks, the matrix, the linear solve | Per iterate: the worst status code and four norms |
| | Per step: target values of prescribed components, from the load curves |
| | Per output state: displacement and stress for the writers |

Three things in the template are not yet as they should be for a GPU, and are marked in the code. The matrix is rebuilt from triplets each iteration, although its pattern is fixed and Warp can refill values through a stored map. Blocks from several terms are joined through the host. Nothing has been run on a GPU or captured as a CUDA graph.

### Gradients through equilibrium

A converged state satisfies R(u, p) = 0, and K = -dR/du is the tangent the Newton step already forms. For an output J(u):

    dJ/dp = lambda . dR/dp,        K^T lambda = dJ/du

```mermaid
flowchart LR
    state["Converged State"] --> K["Tangent K<br/><i>same linearize as the forward solve</i>"]
    out["dJ/du"] --> solve["Solve K^T lambda = dJ/du<br/><i>same linear solve</i>"]
    K --> solve
    solve --> tape["Tape: update, residual<br/><i>same kernels as the forward solve</i>"]
    state --> tape
    tape --> grad["lambda . dR/dp<br/>for every parameter array"]
```

- **Cost.** One linear solve and one reverse pass, whatever the number of parameters. The Newton iterations are not recorded.
- **One implementation of the element.** The reverse pass runs the forward `update` and `residual` kernels under `wp.Tape`. This is why those two are written for reverse mode (fixed loop bounds, no array element written twice) and `linearize` is kept separate.
- **Parameters are device arrays.** Each term lists its own (`Term.parameters()`), which is what the material layer already provides. A term with a closed-form derivative can supply it instead, as contact does with `residual_parameter_vjp`.

## 6. The basic example, and what was checked

`examples/run_path/basic_example.py` runs the path on `hex8_neo_hookean.feb` in six numbered stages: read, build, run, compare with FEBio, write a plot file, compute a gradient.

| Function | File | Role |
| --- | --- | --- |
| `build_field_table(model)` | `discretization/fields.py` | Entries and boundary condition states |
| `select_domain(...)` | `analysis/steps.py` | FEBio's domain choice for one element block |
| `build_system(model, device)` | `analysis/steps.py` | Coverage check, field table, term list |
| `System.update`, `residual`, `tangent` | `assembly/system.py` | The three passes |
| `solve_step(system, state, time, ...)` | `solvers/nonlinear/newton.py` | One time step |
| `solve(A, b, method)` | `solvers/linear/warp_linear.py` | CG, BiCGSTAB, GMRES on the device; dense on the host for checks |
| `run(system)` | `analysis/steps.py` | The step, with the time step controller |
| `parameter_gradient(system, state, dJ_du)` | `sensitivity/adjoint.py` | Adjoint gradient |

Results, CPU only, Warp 1.19.0, double precision:

| Check | Inputs | Result |
| --- | --- | --- |
| Displacement and stress against FEBio log data | Single hex8, single quad4 shell, 8-element block, 9-element curved shell patch | Largest relative difference 4e-12 |
| The same with Warp's CG, BiCGSTAB and GMRES | The same four | Largest relative difference 4e-11 |
| Same output times and Newton iteration counts as the fixture | The same four | Equal; displacements agree to 1e-11 |
| Tangent equals minus the finite-difference derivative of the residual | Single hex8, single quad4 shell | 1e-8 relative; symmetric to 1e-12 |
| Adjoint gradient against central differences of complete runs | Single hex8, single quad4 shell, 8-element block | Agree to 1e-6 relative for Poisson's ratio; zero for the modulus in both |
| An input needing unsupported features is refused with the coverage report | Open mitral closure input | Refused |

The gradient check has a limit worth stating. These inputs are displacement-driven, so the modulus has no effect on the displacements and only Poisson's ratio gives a nonzero gradient. A load-driven case is needed to exercise both.

The whole test suite on this branch: 125 tests, 121 pass and 4 skip (two need a GPU, two need VTK and h5py).

## 7. Points for review

Each was checked against the FEBio source (`e4df3f8`), Warp 1.19.0 and the code on the integration branch.

1. **Numbering by (node, variable).** FEBio numbers each free degree of freedom and removes constrained ones. Here every entry is a 3-vector and constrained components are masked. Does this lose anything FEBio relies on that we should plan for now, for example linear constraints or rigid bodies, which this template does not have?
2. **One contract for domains, loads and contact.** FEBio keeps these as separate class families that the solver calls in turn. Here they share `Term`. Is there a case in FEBio where the order or the separation of those passes matters to the result?
3. **Residual sign.** The template uses FEBio's (external minus internal). The contact draft documents the opposite. The proposal is to change contact to FEBio's.

## 8. Where the next features attach

| Feature | Attaches as | In |
| --- | --- | --- |
| tet4, penta6, tri3 domains; three-field domain | A branch of `select_domain` returning another `Term` | `analysis/steps.py`, `elements/` |
| `pressure`, `mass damping` | A `Term` appended in `build_system` | `loads/` |
| Chordae (`Discrete`, nonlinear spring) | A `Term` on the uniaxial law of the material layer | `elements/` |
| Contact potential | The contact draft's interface as a `Term`, with its blocks kept in their own matrix | `contact/` |
| Several steps, per-step activation | A loop in `run` that switches terms and conditions | `analysis/steps.py` |
| Dynamic analysis | Inertia from the domains through the same terms; generalized-alpha in the step | `analysis/`, `solvers/nonlinear/` |
| Beams, rigid bodies | New variables in the field table with their own update rule | `discretization/fields.py` |

## 9. Limits

- Static analysis, one step, hex8 solids and quad4 shells, the materials the binding accepts, fixed and prescribed displacements. This is exactly what the fixture ran.
- No loads, so no load-driven case and no follower stiffness.
- No history, so `commit` and `rollback` are called and do nothing.
- CPU only. No timing was taken, and none would mean much at this size.
- The first-iteration handling of prescribed displacements and the convergence tests were carried over from the fixture, which was compared with FEBio; they were not re-derived from the FEBio source for this note.
