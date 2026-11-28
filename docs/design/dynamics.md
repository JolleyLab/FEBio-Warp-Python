# Dynamics, mass damping and steps

Template for discussion, November 27, 2026 (v0.3). Nothing here is settled.

**What changed from v0.2.** That version was design only. This one comes with a first implementation for one dynamic step with alpha_f = 1: velocity and acceleration in the State, consistent mass for hex8 solids, quad4 shells and trusses, FEBio's time integration and initial accelerations, and mass damping. Two stored FEBio references are reproduced (section 6). Several steps, alpha_f below 1 and tet4 mass are still design only. Part of the set introduced in [remaining_areas.md](remaining_areas.md).

**Scope.** Implicit dynamic analysis as FEBio's solid solver does it: mass, time integration, the `mass damping` load, and several steps with features switched on by step. Both application inputs need all of it.

## 1. Summary

- **FEBio's scheme is followed exactly**: the same parameters, the same update of velocity and acceleration, the same scaling of the mass matrix.
- **State grows by two arrays.** Velocity and acceleration sit beside the displacement, one 3-vector per entry of the field table. A step is still a function from one State to the next.
- **Mass costs no new matrix pattern.** A domain's mass blocks couple the same node pairs as its stiffness blocks. They are computed once and added into the slice the domain already owns.
- **Mass damping becomes one block product.** Its force is C times the mass blocks of its element set times the velocity.
- **Mass damping is a third approximate tangent, and the first deliberate deviation.** FEBio's stiffness for it is not the derivative of its force, and its shell domain adds it with the opposite sign of its solid and truss domains. Newton here uses the derivative by default, which is also what the adjoint needs: the same solution in a quarter to a half of the iterations on the two reference cases. FEBio's form is kept as a setting and reproduces FEBio's iteration counts (decision D4).
- **The FEBio test suite covers most of this**: one-element dynamic solids and shells, two-step problems, and trusses with mass damping. It has no simple problem at the time integration setting both applications use, so one small input should be added.

## 2. One dynamic step

```mermaid
flowchart TB
    prev(["State n<br/>u, v, a"]) --> guess["Start of step<br/>u unchanged<br/>v, a from the update rule"]
    guess --> upd["Update<br/>v and a from u (one kernel)<br/>domains: kinematics and material"]
    upd --> res["Residual<br/>loads + contact - internal force<br/>- M (alpha_m a + (1 - alpha_m) a_n)<br/>- C M v on damped sets"]
    res --> tan["Tangent<br/>K + alpha_m / (beta dt^2) M<br/>+ damping stiffness"]
    tan --> solve["Mask, linear solve, increment"]
    solve --> conv{"Converged ?"}
    conv -- no --> upd
    conv -- yes --> next(["State n+1<br/>u, v, a committed"])
    conv -- "failed" --> cut["Cut the step, restart from State n"]
    cut --> guess
```

## 3. What FEBio does

From `FEBioMech/FESolidSolver2.cpp`, `FEElasticSolidDomain.cpp`, `FEElasticShellDomain.cpp` and `FEMassDamping.cpp` at `e4df3f8`.

| Item | FEBio |
| --- | --- |
| Parameters | `rhoi`, `alpha`, `beta`, `gamma` on the solver. Default `rhoi` is -2, which means: use `alpha`, `beta`, `gamma` as given (defaults 1, 0.25, 0.5) |
| `rhoi` in [0, 1] | alpha_f = 1/(1 + rhoi), alpha_m = (2 - rhoi)/(1 + rhoi), beta = (1 + alpha_m - alpha_f)^2 / 4, gamma = 1/2 + alpha_m - alpha_f |
| Acceleration | a = (x - x_n)/(beta dt^2) - v_n/(beta dt) + (1 - 1/(2 beta)) a_n |
| Velocity | v = v_n + ((1 - gamma) a_n + gamma a) dt |
| Shell | The same two formulas for `shell displacement` and its rates |
| Start of a step | Displacement kept; a and v from the formulas with x = x_n |
| Inertial force | Consistent mass with the reference density, on alpha_m a + (1 - alpha_m) a_n |
| Mass in the stiffness | The mass matrix scaled by alpha_m / (beta dt^2) |
| Where stress is evaluated | At the deformation alpha_f F + (1 - alpha_f) F_n |
| `mass damping` | A body force C v per unit mass on an element set. Its stiffness is C/dt per unit mass |
| Steps | Each `FEAnalysis` activates its own boundary conditions, loads and contact on top of the global ones when it starts, and deactivates them when it ends |

Two consequences for the first targets.

- Both application inputs set `rhoi` to 0, so alpha_f = 1, alpha_m = 2, beta = 1, gamma = 1.5. With alpha_f = 1 the stress is evaluated at the end of the step, as in a static step. The interpolated deformation (alpha_f below 1) is not needed by either input. Four small suite problems set `rhoi` to 1, which gives alpha_f = alpha_m = 1/2, so it is needed to use them (decision D3).
- FEBio's damping stiffness C/dt differs from the derivative of the damping force, which is C gamma/(beta dt): 1.5 C/dt at `rhoi` 0, and 2 C/dt at the defaults.
- FEBio's domains add the damping stiffness with different signs. With K_b = `bf.stiffness(mp)`, which is -C/dt for mass damping: `FEElasticSolidDomain::BodyForceStiffness` forms K_b times (-H_a H_b rho J0) and `FELinearTrussDomain::BodyForceStiffness` subtracts K_b, so both add +C/dt M; `FEElasticShellDomain::ElementBodyForceStiffness` adds K_b rho J0 M_a M_b, which is -C/dt M. Read from the source at `e4df3f8` and confirmed by iteration counts (section 6). Whether the shell sign is intended is a question for Steve.

## 4. Proposed

| Topic | Proposal | Why |
| --- | --- | --- |
| State | `State` holds `u`, `v`, `a`, each one 3-vector per entry | Same layout as the unknowns; the update of v and a is one kernel over entries with no host round trip |
| Mass | Each domain computes its consistent mass once when the system is built, in the (row, column) order of its stiffness blocks, as one number per block (`Term.mass_coefficients`). The system keeps them beside its block buffer (`System.set_dynamics`). Built: hex8, quad4 shell (as the brick it is run as), truss. Springs have none, as in FEBio. Not built: tet4, so a dynamic model with tet4 is refused | The pattern is unchanged and the mass is constant. Adding it to the tangent is one scaled add into the domain's slice of the block buffer |
| Inertial force | The mass blocks applied to alpha_m a + (1 - alpha_m) a_n | One block product, the same kernel the tangent product already uses |
| Mass damping | Not a term of its own in the template: its C is given to the system, which applies it through the mass blocks of the part. An element set that is not a whole part, and a load curve on C, are refused. Force C M_set v; Newton tangent (C/dt) M_set as FEBio; exact tangent C gamma/(beta dt) M_set | FEBio integrates density times shape function times C v, with v interpolated from the nodes, which is C M v. No new element code. Confirmed on `bk_mass_damping` and `sh_lee_sacks_pressure` (section 6) |
| Time integration | A small settings object read from the solver record, giving alpha_f, alpha_m, beta, gamma by FEBio's rules | One place for the rules above |
| Steps (not built) | `run` loops over the model's steps. Each step has a list of active terms and boundary condition records; the field table's masks are rebuilt between steps, the matrix pattern is not | Masking was chosen so that a change of boundary conditions renumbers nothing |
| Time step controller | Unchanged from the run path; a rejected attempt restarts from State n | State n is kept until the new state converges |

**Alternative considered.** Lumped mass on the diagonal. It would be cheaper and would improve the conditioning of the matrix, but FEBio uses consistent mass, so results would not match the reference. It could be offered later as an option of the faster tier.

## 5. What Warp provides

| Piece | Use |
| --- | --- |
| The block buffer and pattern map of the run path (`bsr_set_from_triplets` with a map, then `bsr_set_from_mapped_values`) | Mass enters the tangent through the same map |
| `bsr_mv` | Inertial and damping forces as block products |
| Kernels over entries | The v and a update, differentiable by the tape as written |

## 6. Initial explorations

In double precision, on CPU and on a GPU (Quadro RTX 8000, `cuda:0`), against FEBio results that were stored before this code existed (FEBio 4.13.0, provisional source build).

| Explored | Result | Where |
| --- | --- | --- |
| `bk_mass_damping`: one hex8, a pressure stepped on at time zero, mass damping, 200 steps | Displacement, velocity and stress agree with FEBio to 6e-14 over the whole run. The run starts from FEBio's initial accelerations (M a = F at time zero) | `tests/test_dynamics.py` |
| `sh_lee_sacks_pressure`: 36 quad4 shells, isotropic Lee-Sacks, pressure ramp, mass damping, 100 steps. The valve setting in small | Displacement and stress agree to 3e-8 relative at the input's `dtol` of 1e-7, where both codes stop on a linearly converging iteration. With `dtol` 1e-10 in both, the displacement agrees to 6e-13 (FEBio 4.13.0 run on WS02; not a stored reference) | Same |
| Exact tangent of a dynamic step (mass and damping included) against central differences of the residual | Agrees to 1e-7. It differs from the Newton tangent by (gamma/beta - 1) C M / dt, as section 3 says, and by the antisymmetric part of the pressure stiffness | Same |
| Newton iteration counts against FEBio (4.13.0 at `e4df3f8`, run on WS02), with FEBio's damping stiffness (`damping_tangent = "febio"`) | `bk_mass_damping`: 902, FEBio 902. `sh_lee_sacks_pressure`: 1400, FEBio 1397 | Same |
| The three damping stiffnesses Newton can use, same two cases, total iterations | Shell case: FEBio's form 1400, the sign of the solid for all domains 744, the derivative 325. Solid case: 902, 902, 416. The solutions agree within the tolerance of the input | Same |
| Gradient through one dynamic step: a displacement of the shell patch with respect to every parameter of the system, adjoint against central differences | Agree to 1e-4 relative, the test tolerance. The step starts from rest with no load at time zero, so the start does not depend on the parameters | Same |
| Suite `dy16` (trusses with mass) and `dy07` (40 hex8), against the suite's stored values | FEBio's 100 steps and its 276 and 200 iterations exactly; solution norm to 1.4e-5 and 4e-7 (FEBio ran them with BFGS at `dtol` 1e-3) | Same, with `FEBIO_TEST_SUITE` set |
| Suite `sh58` (40 quad4 shells, nodal force on at time zero, automatic time stepping), against FEBio run on WS02 | The first 14 steps have FEBio's step sizes and iteration counts, and displacements equal to the precision of the plot file (2e-8 to 5e-8). From step 15 FEBio's line search shortens steps (it reports step lengths near 0.5), so iteration counts, then step sizes, then the final norm (2 percent) differ. The cause is the missing line search, not the dynamics | Checked once, not a test |
| Initial accelerations on shells (`sh58`: a load on at time zero, both faces of the shell in M a = F) | Step 1 agrees with FEBio to 1e-8 | Same check |
| The kinematic update against FEBio's formulas; the parameter rules for `rhoi` -1, 0, 1 and the defaults | Agree | Same |
| A model with `rhoi` 1 (alpha_f = 1/2) | Refused by the coverage report, with the reason | Same |
| FEBio's parameter rules, kinematic update, mass scaling and damping stiffness | Read from source and written out in section 3 | `FESolidSolver2.cpp`, `FEMassDamping.cpp` at `e4df3f8` |
| Time integration settings of the two application inputs | Both `rhoi` 0 | The input files |
| Time integration settings across the suite | Of 44 dynamic problems in the solid module, 25 use the default parameters; `rhoi` is 1 in eight, four of them one-element problems; `rhoi` is 0 in four, all with rigid bodies | Suite at `986005f` |
| Static steps with FEBio's time stepper | Same step sequence as FEBio | Run path note |

## 7. Decision points

| ID | Decision | Options | Proposed | For |
| --- | --- | --- | --- | --- |
| D1 | Mass | (a) Consistent, as FEBio. (b) Lumped | (a) for the reference tier. (b) as an option of the faster tier later | Steve; Warp team for (b) |
| D2 | Where velocity and acceleration live | (a) In `State`, as arrays per entry. (b) On nodes of the model, as FEBio | (a). It is what lets a step be replayed for the adjoint | Steve, Warp team |
| D3 | Stress at an interpolated deformation (alpha_f below 1) | (a) Build it now, to use suite problems `dy10`, `dy11`, `dy13`, `dy14`. (b) Reserve it; start with suite problems at default parameters | (b) first, then (a). The applications do not need it; the suite does | Steve |
| D4 | The stiffness Newton uses for mass damping. FEBio's is C/dt per unit mass, which is not the derivative of the damping force, C gamma/(beta dt); and FEBio's shell domain adds it with the opposite sign of its solid and truss domains (section 3) | (a) FEBio's form, per domain, for Newton; the derivative for the adjoint. (b) C/dt with one sign for all domains. (c) The derivative for both | (c) as the default, a deliberate deviation: one tangent for Newton and the adjoint, and 325 iterations where FEBio's form takes 1400 on the shell case (416 against 902 on the solid). (a) stays available as `System.damping_tangent = "febio"` and is what the comparisons of iteration counts with FEBio use. The automatic time stepper follows iteration counts, so with (c) step sizes can differ from FEBio's where it is active. For Steve: is the sign in the shell domain intended, and would the derivative be welcome in FEBio | Steve, Jeff |
| D5 | Step sizes in the adjoint | The time step controller's choices are treated as fixed when differentiating | As stated | Warp team |

## 8. Verification

FEBio test suite first (commit `986005f`), then small inputs for what the suite lacks.

| Input | Source | What it exercises | Time integration |
| --- | --- | --- | --- |
| `dy01` | Suite | One hex8, neo-Hookean, constant body force | Default |
| `dy07` | Suite | 40 hex8, prescribed displacement. Passes (section 6) | Default |
| `sh32` | Suite | One quad4 shell, constant body force | Default |
| `sh34`, `sh58` | Suite | 40 quad4 shells, body force or nodal force. `sh58` runs and follows FEBio until FEBio's line search acts (section 6); `sh34` needs the constant body force | Default |
| `dy16`, `dy17` | Suite | Trusses; `dy17` has mass damping ([chordae.md](chordae.md)). `dy16` passes (section 6); `dy17` needs the constant body force | Default |
| `ms01`, `ms02`, `ms03`, `ms05` | Suite | Two steps, one static and one dynamic | Default |
| `dy10`, `dy11`, `dy13`, `dy14` | Suite | One hex8 or one quad4 shell | `rhoi` 1 (needs D3) |
| `bk_mass_damping`, `sh_lee_sacks_pressure` | `references/small_problems` | Mass damping on a solid and on Lee-Sacks shells with pressure. The suite has mass damping only on trusses | Default |
| A copy of `bk_mass_damping` with `rhoi` 0 | To add | The setting both application inputs use. No simple suite problem has it | `rhoi` 0 |
| A two-step input that switches a load or contact on in step 2 | To add | Activation by step, which `ms` problems exercise only through the analysis type | |

The suite problems above need three small load terms the run path does not have: the constant `body force`, `nodal_load` and `nodal_force`. Suite inputs with `max_ups` 0 can be compared on iteration counts; those with `max_ups` 10 (BFGS) on results only, until quasi-Newton exists.

## 9. Gradients

A step solves R_n(u_n; u_{n-1}, v_{n-1}, a_{n-1}, p) = 0. The reverse sweep is in [sensitivity.md](sensitivity.md). Two points belong here.

- The v and a update is a kernel, so the tape gives its derivatives with respect to the previous state without extra code.
- `linearize(exact=True)` must return C gamma/(beta dt) M for mass damping and the true pressure and material tangents, with the same mass term as the forward solve.
- FEBio has no counterpart to check a gradient against. The check is finite differences of FEBio-Warp itself on a one-element dynamic case over a few steps ([sensitivity.md](sensitivity.md)).
- Built so far: the gradient through one dynamic step that starts from a state independent of the parameters (section 6). The exact dynamic tangent, with the mass and the derivative of the damping force, is the matrix of that adjoint solve. Gradients over several steps need the reverse sweep of [sensitivity.md](sensitivity.md).

## 10. Reserved

- alpha_f below 1 (stress at an interpolated deformation), until decision D3. The domains would take alpha_f as an input to `update`.
- Rigid body inertia, for the dilation model.
- Initial velocity conditions. Neither application input has an `Initial` section.

## 11. Depends on

Numbering by (node, variable), and the `Term` contract. Both are points for review in the run path note.
