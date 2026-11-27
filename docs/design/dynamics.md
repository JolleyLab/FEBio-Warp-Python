# Dynamics, mass damping and steps

First template for discussion, November 26, 2026. Design only: there is no code for this note. Nothing here is settled. Part of the set introduced in [remaining_areas.md](remaining_areas.md).

**Scope.** Implicit dynamic analysis as FEBio's solid solver does it: mass, time integration, the `mass damping` load, and several steps with features switched on by step. Both application inputs need all of it.

## 1. Summary

- **FEBio's scheme is followed exactly**: the same parameters, the same update of velocity and acceleration, the same scaling of the mass matrix.
- **State grows by two arrays.** Velocity and acceleration sit beside the displacement, one 3-vector per entry of the field table. A step is still a function from one State to the next.
- **Mass costs no new matrix pattern.** A domain's mass blocks couple the same node pairs as its stiffness blocks. They are computed once and added into the slice the domain already owns.
- **Mass damping becomes one block product.** Its force is C times the mass blocks of its element set times the velocity.
- **Mass damping is a third approximate tangent.** FEBio's stiffness for it is not the derivative of its force. Newton follows FEBio; the adjoint asks for the exact form.
- **The stored references do not yet exercise the parameters the applications use.** One small input should be added.

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

- Both application inputs set `rhoi` to 0, so alpha_f = 1, alpha_m = 2, beta = 1, gamma = 1.5. With alpha_f = 1 the stress is evaluated at the end of the step, as in a static step. The interpolated deformation (alpha_f below 1) is not needed by either input and is reserved.
- FEBio's damping stiffness C/dt differs from the derivative of the damping force, which is C gamma/(beta dt): 1.5 C/dt at `rhoi` 0, and 2 C/dt at the defaults.

## 4. Proposed

| Topic | Proposal | Why |
| --- | --- | --- |
| State | `State` holds `u`, `v`, `a`, each one 3-vector per entry | Same layout as the unknowns; the update of v and a is one kernel over entries with no host round trip |
| Mass | Each domain computes its consistent mass blocks once when the system is built, in the (row, column) order of its stiffness blocks | The pattern is unchanged and the mass is constant. Adding it to the tangent is one scaled add into the domain's slice of the block buffer |
| Inertial force | The mass blocks applied to alpha_m a + (1 - alpha_m) a_n | One block product, the same kernel the tangent product already uses |
| Mass damping | A term on its element set: force C M_set v; Newton tangent (C/dt) M_set as FEBio; exact tangent C gamma/(beta dt) M_set | FEBio integrates density times shape function times C v, with v interpolated from the nodes, which is C M v. No new element code. To be confirmed on `bk_mass_damping` |
| Time integration | A small settings object read from the solver record, giving alpha_f, alpha_m, beta, gamma by FEBio's rules | One place for the rules above |
| Steps | `run` loops over the model's steps. Each step has a list of active terms and boundary condition records; the field table's masks are rebuilt between steps, the matrix pattern is not | Masking was chosen so that a change of boundary conditions renumbers nothing |
| Time step controller | Unchanged from the run path; a rejected attempt restarts from State n | State n is kept until the new state converges |

**Alternative considered.** Lumped mass on the diagonal. It would be cheaper and would improve the conditioning of the matrix, but FEBio uses consistent mass, so results would not match the reference. It could be offered later as an option of the faster tier.

## 5. What Warp provides

| Piece | Use |
| --- | --- |
| The block buffer and pattern map of the run path (`bsr_set_from_triplets` with a map, then `bsr_set_from_mapped_values`) | Mass enters the tangent through the same map |
| `bsr_mv` | Inertial and damping forces as block products |
| Kernels over entries | The v and a update, differentiable by the tape as written |

## 6. Verification

| Input | What it exercises | Status |
| --- | --- | --- |
| `bk_mass_damping.feb` | One hex8, dynamic, pressure and mass damping | Input and FEBio reference stored. Uses the default parameters (beta 0.25, gamma 0.5) |
| `sh_lee_sacks_pressure.feb` | 36 quad4 shells, dynamic, pressure and mass damping, the leaflet material | Stored. Default parameters |
| A copy of `bk_mass_damping.feb` with `rhoi` 0 | The parameters both application inputs use | To add, with its FEBio reference |
| A two-step input with a load switched on in step 2 | Activation by step | To add |

All twenty stored small problems set `max_ups` to 0, which makes FEBio reform the stiffness at every iteration, so iteration counts can be compared with full Newton directly.

## 7. Gradients

A step solves R_n(u_n; u_{n-1}, v_{n-1}, a_{n-1}, p) = 0. The reverse sweep is in [sensitivity.md](sensitivity.md). Two points belong here.

- The v and a update is a kernel, so the tape gives its derivatives with respect to the previous state without extra code.
- `linearize(exact=True)` must return C gamma/(beta dt) M for mass damping and the true pressure and material tangents, with the same mass term as the forward solve.

## 8. Reserved

- alpha_f below 1 (stress at an interpolated deformation). The domains would take alpha_f as an input to `update`.
- Rigid body inertia, for the dilation model.
- Initial velocity conditions. Neither application input has an `Initial` section.

## 9. Depends on

Numbering by (node, variable), and the `Term` contract. Both are points for review in the run path note.
