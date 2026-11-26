# Design notes

One note per development area. A note states the formulation, the FEBio source it follows, the
interfaces it adds to `src/warp_febio/`, and how the implementation will be verified. A note is
added or updated in the same pull request as the code it describes, or ahead of it for discussion.

| Area | Note | Status |
|---|---|---|
| Materials | [materials.md](materials.md), with [PR #2](https://github.com/JolleyLab/FEBio-Warp-Python/pull/2) | Draft, under redesign |
| Contact | [contact.md](contact.md), with [PR #1](https://github.com/JolleyLab/FEBio-Warp-Python/pull/1) | Draft |
| Assembly | To be added | Design |
| Run path (unknowns, terms, Newton step, steps, pressure, warp.fem solids, gradients) | [run_path.md](run_path.md) | Template, local draft |
| Verification | [verification.md](verification.md) | Draft, tabled until input and output |
| Central model, input and output | [model.md](model.md), with [PR #5](https://github.com/JolleyLab/FEBio-Warp-Python/pull/5) | Draft |

[integration.md](integration.md) says how these drafts fit together on the combined branch and what is left open.

Notes describing the pilot phase are in [`pilots/docs/`](../../pilots/docs/).
