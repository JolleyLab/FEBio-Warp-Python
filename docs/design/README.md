# Design notes

One note per development area. A note states the formulation, the FEBio source it follows, the
interfaces it adds to `src/warp_febio/`, and how the implementation will be verified. A note is
added or updated in the same pull request as the code it describes, or ahead of it for discussion.

Start with the [design overview](overview.md): one page on how the pieces fit, what state each is in, and what is only a first template.

| Area | Note | Status |
|---|---|---|
| Materials | [materials.md](materials.md) ([decisions](materials.md#decision-points)), with [PR #2](https://github.com/JolleyLab/FEBio-Warp-Python/pull/2) | On `main`, under redesign |
| Contact | [contact.md](contact.md) ([decisions](contact.md#decision-points)), with [PR #1](https://github.com/JolleyLab/FEBio-Warp-Python/pull/1) | On `main` |
| Assembly | [assembly.md](assembly.md) | Decided (full assembly); carried out in the run path template |
| Run path (unknowns, terms, Newton step, steps, pressure, warp.fem solids, gradients) | [run_path.md](run_path.md) ([decisions](run_path.md#12-decision-points)), with [PR #6](https://github.com/JolleyLab/FEBio-Warp-Python/pull/6) | Template on `main` |
| Verification | [verification.md](verification.md) ([decisions](verification.md#decision-points)) | On `main`; the comparisons beyond material points wait on the run path |
| Central model, input and output | [model.md](model.md) ([decisions](model.md#9-decision-points)), with [PR #5](https://github.com/JolleyLab/FEBio-Warp-Python/pull/5) | On `main` |

[integration.md](integration.md) says how these drafts were made to fit together before they were merged to `main` on November 27, 2026, and what is left open.

## Remaining areas: design only

First templates for discussion. Each is a short note with a flow diagram, the FEBio source it follows, a proposed approach, what has been explored, the decision points for the team, and the FEBio test suite problems that would verify it. None has code yet, except the chordae note, which has a draft of both elements for static steps. Start with [remaining_areas.md](remaining_areas.md), which orders them along the path to the first application runs and lists the decisions by reviewer. [work_plan.md](work_plan.md) says what must be agreed first and which work packages can then proceed in parallel.

| Area | Note | Decision points | Needed by |
|---|---|---|---|
| Dynamics, mass damping, steps | [dynamics.md](dynamics.md) | [D1 to D5](dynamics.md#7-decision-points) | Valve and device |
| Chordae: truss and spring elements (draft code for static steps) | [chordae.md](chordae.md) | [C1 to C10](chordae.md#7-decision-points) | Valve |
| Contact in the run | [contact_in_run.md](contact_in_run.md) | [K1 to K5](contact_in_run.md#7-decision-points) | Valve and device |
| Linear solve, nonlinear strategy, first GPU runs | [linear_solve.md](linear_solve.md) | [L1 to L6](linear_solve.md#7-decision-points) | Valve and device |
| Output registry | [output.md](output.md) | [O1 to O4](output.md#5-decision-points) | Valve and device |
| Elements: order of work; three-field solid, penta6, tri3 shell | [elements.md](elements.md) | [E1 to E5](elements.md#7-decision-points) | Valve (order), device |
| Constraints: normal displacement | [constraints.md](constraints.md) | [B1 to B3](constraints.md#5-decision-points) | Device |
| Sensitivity over steps: adjoints and automatic differentiation | [sensitivity.md](sensitivity.md) | [S1 to S10](sensitivity.md#6-decision-points) | Gradients for both |

Notes describing the pilot phase are in [`pilots/docs/`](../../pilots/docs/).
