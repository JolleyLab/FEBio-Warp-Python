# Design notes

One note per development area. A note states the formulation, the FEBio source it follows, the
interfaces it adds to `src/warp_febio/`, and how the implementation will be verified. A note is
added or updated in the same pull request as the code it describes, or ahead of it for discussion.

Start with the [design overview](overview.md): one page on how the pieces fit, what state each is in, and what is only a first template.

| Area | Note | Status |
|---|---|---|
| Materials | [materials.md](materials.md), with [PR #2](https://github.com/JolleyLab/FEBio-Warp-Python/pull/2) | Draft, under redesign |
| Contact | [contact.md](contact.md), with [PR #1](https://github.com/JolleyLab/FEBio-Warp-Python/pull/1) | Draft |
| Assembly | [assembly.md](assembly.md) | Decided (full assembly); carried out in the run path template |
| Run path (unknowns, terms, Newton step, steps, pressure, warp.fem solids, gradients) | [run_path.md](run_path.md), with [PR #6](https://github.com/JolleyLab/FEBio-Warp-Python/pull/6) | Template, draft |
| Verification | [verification.md](verification.md) | Draft, tabled until input and output |
| Central model, input and output | [model.md](model.md), with [PR #5](https://github.com/JolleyLab/FEBio-Warp-Python/pull/5) | Draft |

[integration.md](integration.md) says how these drafts fit together on the combined branch and what is left open.

## Remaining areas: design only

First templates for discussion. Each is a short note with a flow diagram, the FEBio source it follows, a proposed approach and the input that would verify it. None has code yet. Start with [remaining_areas.md](remaining_areas.md), which orders them along the path to the first application runs.

| Area | Note | Needed by |
|---|---|---|
| Dynamics, mass damping, steps | [dynamics.md](dynamics.md) | Valve and device |
| Chordae: discrete springs | [discrete.md](discrete.md) | Valve |
| Contact in the run | [contact_in_run.md](contact_in_run.md) | Valve and device |
| Linear solve, nonlinear strategy, first GPU runs | [linear_solve.md](linear_solve.md) | Valve and device |
| Output registry | [output.md](output.md) | Valve and device |
| Elements: tri3 shell, penta6, three-field solid | [elements.md](elements.md) | Device |
| Constraints: normal displacement | [constraints.md](constraints.md) | Device |
| Sensitivity over steps | [sensitivity.md](sensitivity.md) | Gradients for both |

Notes describing the pilot phase are in [`pilots/docs/`](../../pilots/docs/).
