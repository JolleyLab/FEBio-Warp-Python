# Sensitivity over steps

First template for discussion, November 26, 2026 (v0.2). Design only: there is no code for this note beyond what the run path already has. Nothing here is settled. Part of the set introduced in [remaining_areas.md](remaining_areas.md).

**Scope.** How the adjoint of the run path, which differentiates one equilibrium solve, extends to a dynamic run of many steps with history and contact, and which decisions the team needs to make for that to work.

**Why this note carries more decisions than the others.** FEBio has no adjoint: no derivative of a solution with respect to a parameter through the solve. It does have forward-mode automatic differentiation by dual numbers (`FECore/ad.h`, `ad2.h`), used locally, for example to obtain the stress and tangent of `neo-Hookean AD` and `Mooney-Rivlin AD` from a strain energy. Its route to parameter identification is its optimization module, which calls the forward model repeatedly (the suite's eleven `op` problems use Levenberg-Marquardt). So for every other area there is a FEBio design to follow, and for gradients of a solution there is not. The choices below are FEBio-Warp's own and are the ones where input from the Warp team and from Steve and Jeff matters most.

## 1. Summary

- **What exists.** A gradient through one converged static state, with respect to every parameter any term exposes, at the cost of one transpose solve. The solve can sit on a `wp.Tape` as one step. Checked against a closed form and finite differences on the pressure case.
- **What both applications need.** A gradient through a dynamic run: 200 steps for the mitral input, more than 200 for TPVR.
- **The extension is a reverse sweep over the stored States**, one transpose solve per step, with the tape giving the dependence of each step on the one before.
- **The transpose solve is likely the main added cost of a gradient.** With an iterative solver it is a second Krylov solve. With a direct solver that can solve with the transpose from existing factors, as cuDSS is documented to, it would be two substitutions where Newton's tangent is the exact one ([linear_solve.md](linear_solve.md), section 4 and decision L5).
- **Every term must be able to give its exact tangent.** There are now three known places where FEBio's Newton tangent is not the derivative of the residual.
- **Memory is modest for the first targets** if only converged States are stored.

## 2. Forward run and reverse sweep

```mermaid
flowchart LR
    subgraph fwd["Forward: Newton iterations are not recorded"]
        s0(["State 0"]) --> st1["step 1"] --> s1(["State 1"]) --> st2["step 2"] --> s2(["..."]) --> stn["step N"] --> sn(["State N"])
    end
    sn --> J["Output J"]

    subgraph rev["Reverse: one transpose solve per step"]
        an["adjoint N<br/>K_N^T lambda_N = dJ/du_N"] --> an1["adjoint N-1<br/>adds the dependence of<br/>step N on State N-1"] --> a1["..."] --> a0["adjoint 1"]
    end
    J --> an
    an -.-> g["dJ/dp<br/>summed over steps"]
    an1 -.-> g
    a0 -.-> g
```

For step n with residual R_n(u_n; s_{n-1}, p) = 0, where s holds u, v, a and committed history:

- solve K_n^T lambda_n = (the derivative of J with respect to u_n, plus what later steps passed back);
- run the tape backward over `update` and `residual` of step n to get lambda_n times the derivatives with respect to p and to s_{n-1};
- add the first to dJ/dp and pass the second to step n-1.

The kernels are the forward ones. `linearize` is never taped.

## 3. What each area must supply

| Area | For the reverse sweep | State |
| --- | --- | --- |
| Run path | `linearize(exact=True)`; `parameters()` per term; the solve as one taped step | In the template |
| Dynamics | The update of v and a as a kernel, so the tape gives its derivatives; the mass term in the exact tangent | [dynamics.md](dynamics.md) |
| Materials with history | Trial and committed history kept apart, so a step can be revisited | In the material layer for uniaxial models; `commit` and `rollback` are called and do nothing in the run path |
| Contact | Pairs frozen at each converged State and stored with it | The contact note does this for one state |
| Time step controller | Step sizes treated as fixed in the reverse sweep | The run path note states this |
| Springs, pressure, mass damping, boundary values | Their parameters as device arrays | Pressure exists; the others are proposed in their notes |

## 4. Tangents that are not derivatives

Newton only needs a direction, so FEBio is free to use an approximate tangent. The adjoint formula holds only for the true derivative. Known cases in the features the applications use:

| Feature | FEBio's Newton tangent | Exact tangent | Source |
| --- | --- | --- | --- |
| `pressure` with `symmetric_stiffness` 1 | Symmetrized form | The full follower-load derivative | Run path note, measured |
| `tangent_scale` on `isotropic Lee-Sacks` | `c0` multiplied by the scale in the tangent only | The unscaled tangent | Integration note |
| `mass damping` | C/dt | C gamma/(beta dt) | `FEMassDamping::stiffness`; found while writing [dynamics.md](dynamics.md) |

Point 4 of the run path note asks Steve and Jeff whether there are others. The third row is one answer found in the source. One case found to be exact: the stiffness of FEBio's `linear-truss` domain is the derivative of its force ([chordae.md](chordae.md)). Not yet checked: the three-field solid.

## 5. Initial explorations

| Explored | Result | Where |
| --- | --- | --- |
| Adjoint through one static equilibrium, for every parameter of every term | Matches a closed form to 2e-9 and central differences | Run path note, `sensitivity/adjoint.py` |
| The solve as one step on a `wp.Tape`, between other Warp kernels | Gradient with respect to an upstream variable to 1e-6 or better | Run path note |
| The adjoint with FEBio's symmetrized pressure tangent in place of the exact one | Wrong sign: 0.0020 against -0.0350 | Run path note |
| The reverse pass through `warp.fem` integrands | Passes the gradient check | Run path note |
| Gradients of the contact potential with pairs frozen | Tested | Contact draft |
| Gradients of material stress with respect to parameters | Tested | Materials draft |
| Any gradient through more than one step, through history, or through a run with contact | Not explored | |

## 6. Decision points

| ID | Decision | Options | Proposed | For |
| --- | --- | --- | --- | --- |
| S1 | How a run is differentiated | (a) Adjoint per converged step, with the tape used only for one residual evaluation. (b) Tape the whole run including Newton iterations | (a). Memory and cost do not grow with the number of Newton iterations, and the result does not depend on how the solve converged | Warp team |
| S2 | The exact-tangent contract | Every term must be able to return the true derivative of its residual, apart from the tangent Newton uses | Keep `linearize(exact=True)` as part of the `Term` contract, with a finite-difference test per term | Steve, Jeff |
| S3 | Choices that are not differentiable | Contact pairs and gates, the slack corner of a spring, step sizes, line search and quasi-Newton history | Hold each fixed at its converged value. The gradient is then that of the path actually taken. State this in the documentation of every gradient | Warp team, Steve |
| S4 | Which parameters are differentiable | (a) Only material parameters. (b) Every parameter a term holds as a device array | (b): material, pressure, `kc`, spring curve, truss area, modulus and rest length, prescribed values. Node coordinates later | MJ, team |
| S5 | Parameter names | A registry from FEBio's names (`fem.material('Valve').c0`, as FEBio's optimization module writes them) to device arrays | Use FEBio's naming so an optimization input can name the same parameters | Steve |
| S6 | Checking a gradient when FEBio has none | (a) Finite differences of FEBio-Warp itself. (b) Closed forms on single elements. (c) Central differences of two FEBio runs with a perturbed input, on a suite problem. (d) Recovering the parameter of a suite `op` problem | (a) and (b) in the unit tests for every term. (c) once per area as an independent check. (d) as an outcome-level check | Steve, Warp team |
| S7 | Storage for the reverse sweep | (a) Keep every converged State. (b) Checkpoint and recompute | (a) for the first targets (estimate below) | Warp team |
| S8 | Precision | (a) Double for forward and reverse. (b) Single for the reverse pass | (a) | Warp team |
| S9 | The port for other tools | The adjoint solution is the sensitivity to a nodal force. A nodal-force term gives Newton, the physics library, forces in and positions out with gradients | Build the nodal-force term early. The suite's truss and spring problems need it for their loads in any case | Warp and Newton teams |
| S10 | Tangents from automatic differentiation | (a) Hand-written tangents transcribed from FEBio. (b) Tangents derived from the residual or energy by Warp's differentiation, as FEBio does with dual numbers for two materials | (a) for Newton, so iteration counts match FEBio. (b) as the standing check that a tangent is exact, and as a candidate for new terms. A material-layer question as much as one for this note | Steve, Jeff, Warp team |

On S6 (c): FEBio's convergence tolerances limit how accurate a difference of two FEBio runs can be, so the inputs would be run with tight tolerances and a step in the parameter chosen well above the solver noise. On S6 (d): the suite pairs eleven models (`oi01` to `oi11`) with optimization inputs (`op01` to `op11`). Four fit a neo-Hookean parameter on small solid models (`op06`, `op08`, `op10`, `op11`). Running FEBio's side needs a build with the Levenberg-Marquardt library, which the source build used so far does not have.

## 7. Proposed

| Topic | Proposal | Why |
| --- | --- | --- |
| Storage | Keep every converged State on the device (u, v, a, history, contact pairs) | The reverse sweep needs each one. See the estimate below |
| Entry points | `parameter_gradient` and `solve_on_tape` extended from one step to a run, with the same signatures | Code that uses a static gradient today keeps working |
| Outputs | J may read any stored State, not only the last | Closure is judged over the cycle; a device's final shape is one state |
| Checks | `check_parameter_gradients` on a dynamic single element over three steps, for every parameter | The standing gradient case, extended. It runs in the unit tests |
| Checkpointing | Not now | Only needed when stored States no longer fit |

Storage estimate, converged States only, double precision, without history or pairs:

| | Entries | Per State (u, v, a) | States | Total |
| --- | --- | --- | --- | --- |
| Mitral closure | Up to 14,884 | About 1.1 MB | 200 | About 0.2 GB |
| TPVR (file 3599) | At least 65,011 | About 4.7 MB | 220 at the nominal step sizes | About 1 GB |

## 8. Design variables the applications point to

Listed to show which parameters must be device arrays. None is implemented.

| Application | Variable | Where it lives |
| --- | --- | --- |
| Valve | Chord cross-sectional area, modulus and rest length (truss chordae) | Per-element arrays of the truss domain ([chordae.md](chordae.md)) |
| Valve | Chord slack length and stiffness (spring chordae) | Spring curve points and `scale` ([chordae.md](chordae.md)) |
| Valve | Leaflet material parameters | Material parameter arrays (pull request 2) |
| Valve | Pressure | The pressure array of the run path |
| Device | Vessel wall parameters, which are often unknown for a patient | Material parameter arrays |
| Device | Crimp depth | `scale` of `normal displacement` ([constraints.md](constraints.md)) |

## 9. Reserved

- **Gradients through a change of contact pairs.** The pair list is piecewise constant in the positions.
- **Shape gradients.** Derivatives with respect to node coordinates.
- **Gradients with respect to step sizes** chosen by the time step controller.
- **Single-precision reverse passes.**

## 10. Depends on

State apart from the model, and one implementation of each term for forward and reverse (run path, sections 4 and 7).
