# FEBio-Warp design overview

Initial draft, November 26, 2026. This page is the entry point to the design. It shows how the pieces fit, says what state each is in, and links to the note that holds the detail. It is a summary of drafts under review, not a specification. Sections 1 to 4 describe pieces that have a design note and code in a draft pull request. Each note opens with its own diagrams. Section 5 holds first templates for the areas after that; nothing in it is settled.

**The rule the design follows.** Follow FEBio where it matters to a user or a reviewer: the same input files, the same names, the same formulations, the same outputs, and FEBio results as the reference. Build the internals for Warp: flat arrays on the device, one kernel per batch, a matrix pattern that is analysed once, state kept apart from the model, and gradients as a design requirement.

## 1. The whole path

```mermaid
flowchart TB
    feb([".feb, FEBio spec 4.0"]) --> model
    slicer["3D Slicer and SlicerHeart<br/>VTK mesh"] <--> model
    model["<b>Model</b><br/>host, FEBio names and IDs<br/><i>model and I/O</i>"]
    model --> cov["Coverage report<br/>runnable, or what is missing"]
    cov --> build["Build the system<br/>bindings read the model"]

    subgraph device["On the device"]
        fields["Unknowns<br/>(node, variable), 3-vectors"]
        domains["Domains<br/>solid, shell<br/><i>springs, beams later</i>"]
        mats["<b>Materials</b><br/>one kernel per material tree"]
        loads["Loads<br/>pressure"]
        contact["<b>Contact</b><br/>points, pair search, potential"]
        matrix["One block matrix<br/>pattern analysed once"]
    end

    build --> fields & domains & loads & contact
    mats --> domains
    domains & loads & contact --> matrix

    matrix --> run["<b>Run path</b><br/>steps, time stepper, Newton, linear solve"]
    run --> state["State<br/>one per converged time"]
    state --> results(["Results<br/>.xplt for FEBio Studio, VTKHDF for Slicer and ParaView"])
    state -.-> grad["Gradients<br/>adjoint through equilibrium, wp.Tape"]
    febio["FEBio run of the same input<br/><i>verification</i>"] -.->|compared with| results
```

## 2. The pieces

| Piece | What it is | State | Note | Code |
| --- | --- | --- | --- | --- |
| Model and I/O | One `Model` object holding what a FEBio input defines. Reads and writes `.feb`, writes `.xplt` and VTKHDF, exchanges meshes with 3D Slicer. A coverage report says whether an input can run and lists what is missing. | Draft, first in review order | [model.md](model.md) | [PR #5](https://github.com/JolleyLab/FEBio-Warp-Python/pull/5) |
| Materials | FEBio's material structure (typed parameters with ranges, axes and fibers, mixtures, the uncoupled split, history), composed in Python so that Warp compiles each material tree into one kernel. | Draft, parked until model and I/O | [materials.md](materials.md) | [PR #2](https://github.com/JolleyLab/FEBio-Warp-Python/pull/2) |
| Contact | Point-based contact potential after Kamensky et al. and FEBio's `FEContactPotential`: contact points, pair search on the device, force and tangent, gradients. | Draft, parked until model and I/O | [contact.md](contact.md) | [PR #1](https://github.com/JolleyLab/FEBio-Warp-Python/pull/1) |
| Verification | FEBio-Warp run on FEBio inputs and compared with pinned FEBio results, at four levels: material point, element, small problem, application. | Draft, tabled until model and I/O | [verification.md](verification.md) | [PR #4](https://github.com/JolleyLab/FEBio-Warp-Python/pull/4) |
| Integration | The four drafts merged and made consistent where they meet. | Branch | [integration.md](integration.md) | [`mattjolley/integration-base`](https://github.com/JolleyLab/FEBio-Warp-Python/tree/mattjolley/integration-base) |
| Run path | From a model to a result: unknowns, terms, the Newton step, the step driver, pressure, solids on `warp.fem`, adjoint gradients. | Template, draft | [run_path.md](run_path.md) | [PR #6](https://github.com/JolleyLab/FEBio-Warp-Python/pull/6), stacked on the integration branch |

## 3. One run

```mermaid
flowchart TB
    start(["Model"]) --> system["build_system<br/>field table and terms"]
    system --> step["Step driver<br/>load curves, time stepper, must points"]
    step --> update["Update<br/>each domain evaluates its material once per point"]
    update --> status{"Any failed point ?<br/>negative Jacobian, non-finite"}
    status -- yes --> cut["Reject, cut the step back"] --> step
    status -- no --> assemble["Residual and tangent<br/>every term writes its slice of one buffer"]
    assemble --> mask["Mask constrained components"]
    mask --> solve["Linear solve on the device"]
    solve --> conv{"Converged ?<br/>dtol, etol, rtol"}
    conv -- no, within max_refs --> update
    conv -- no, limit reached --> cut
    conv -- yes --> commit["Commit history, store the State"]
    commit --> more{"End time ?"}
    more -- no --> step
    more -- yes --> out(["Results"])
    commit -.-> adj["Adjoint at the converged State<br/>transpose of the same tangent"]
```

## 4. Where it follows FEBio and where it differs

| Topic | Same as FEBio | Different, and why |
| --- | --- | --- |
| Input and output | `.feb` in, `.xplt` out, FEBio names for parameters and plot variables | A host `Model` that can also be built in code or from Slicer |
| Unknowns | Variable names; free, fixed and prescribed states | Numbered by (node, variable) with 3x3 blocks; constrained components masked, not removed, so the matrix pattern never changes |
| Domains and loads | What each computes, integration rules, node order, sign of the residual (external minus internal) | One contract for domains, loads and contact; flat point data; one kernel set per material tree |
| Materials | Parameters, ranges, axes, mixtures, formulas checked against FEBio source and runs | Composition happens before compilation; failure is a status code per point, since a kernel cannot throw |
| Contact potential | The law, its parameters, the default integration points | Gates applied per point pair; an optional radius offset per point; listed in the contact note |
| Nonlinear solve | Prescribed displacements, convergence norms, `max_refs`, the time stepper and its retries | Full Newton to start; no line search or BFGS yet |
| State | What a converged step holds | A step maps one State to the next and does not modify the model, so a run can be differentiated |
| Gradients | None in FEBio | Adjoint through equilibrium using the transpose of the tangent, and `wp.Tape` for the residual. An exact tangent is available where FEBio's is approximate (symmetrized pressure stiffness) |

## 5. First templates: nothing here is settled

These areas have a first shape in the run path template, or only a placeholder. Each now has a short design note with a flow diagram, a proposed approach, what has been explored and the decision points for the team, and no code beyond the template. [remaining_areas.md](remaining_areas.md) orders them along the path to the first application runs.

| Area | First template | Not yet | Design note |
| --- | --- | --- | --- |
| Elements | hex8 and tet4 solids on `warp.fem`; hex8 and quad4 shell on a dedicated kernel | Trusses and springs for chordae (valve path, first); the three-field element, penta6 and tri3 shell (device path) | [chordae.md](chordae.md), [elements.md](elements.md) |
| Loads | Follower pressure on quad4 facets | `mass damping`, nodal forces, pressure on tri3 facets | [dynamics.md](dynamics.md) for `mass damping` |
| Constraints | Fixed and prescribed displacements by masking | `normal displacement`, linear constraints, rigid bodies and joints | [constraints.md](constraints.md) |
| Linear solve | Warp's CG, BiCGSTAB and GMRES with a Jacobi preconditioner | A stronger preconditioner or a direct solver on the device, which thin shells need; line search and quasi-Newton | [linear_solve.md](linear_solve.md) |
| Steps | One static step with FEBio's time stepper | Dynamic analysis, several steps, activation by step | [dynamics.md](dynamics.md) |
| Contact in the run | The contact draft's interface is planned to attach as a term | Its residual sign changed to FEBio's; contact points built from the model's surfaces; pair updates inside the Newton loop | [contact_in_run.md](contact_in_run.md) |
| Sensitivity | Parameter gradients through one equilibrium solve; the solve as one step on a tape | History, dynamics, contact pairing, shape gradients | [sensitivity.md](sensitivity.md) |
| Output | Displacement, stress, reaction force | The registry of FEBio plot variables | [output.md](output.md) |
| Newton (the physics library) | The step is functional and state is separate, as Newton's solvers expect | A wrapper with Newton's `step` signature; nodal forces in, positions out | [run_path.md](run_path.md), section 6 |
| Performance | CPU, double precision, for correctness | GPU runs and timing; single precision where it is adequate | [linear_solve.md](linear_solve.md), section 6 |

Design shells, kept in view so that nothing built now closes them off: beam elements for device frames, a superelastic nitinol law for those beams, rigid bodies and joints, sliding and sticky contact.

## 6. Targets and checks

| Application | Open FEBio input | What it adds beyond the shared core |
| --- | --- | --- |
| Atrioventricular valve closure | `P2Prolapse_Mitral.feb` (FEBio model repository, file 3572) | Chordae springs, leaflet self-contact |
| Transcatheter pulmonary valve replacement | Six deployments (files 3598 to 3603), for example `RVOT1_TPV25.feb` | Two steps, tet4 device, penta6 vessel with uncoupled HGO, tri3 shell tubes, three contact pairs |
| Patch and verification problems | Small inputs in `references/` | One feature each |

The shared core both applications need: dynamic analysis, mass damping and the contact potential. Pressure is a valve load; TPVR file 3599 has none ([remaining_areas.md](remaining_areas.md), section 1).

Checks so far compare with FEBio on the same input, in double precision on CPU: material points, single elements, small meshes, pressure, two materials in two domains, reaction forces, and the automatic time stepper. Figures are in the notes linked above. No application input runs end to end yet.

## 7. Reading order

1. This page.
2. [integration.md](integration.md): how the four drafts meet.
3. [model.md](model.md), then [materials.md](materials.md) and [contact.md](contact.md).
4. [run_path.md](run_path.md): the path through all of them on small cases.
5. [verification.md](verification.md): how results are compared with FEBio.
6. [remaining_areas.md](remaining_areas.md): what is left before the application inputs run, with one short note per area.
