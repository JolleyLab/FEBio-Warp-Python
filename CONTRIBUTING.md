# Contributing to FEBio-Warp

FEBio-Warp brings the mechanics of [FEBio](https://github.com/febiosoftware/FEBio) to GPU-accelerated, differentiable code built on [Warp](https://github.com/NVIDIA/warp). It builds on the work of the FEBio team and is meant to be additive to FEBio. FEBio is the reference implementation for the mechanics, and code here is checked against it where a counterpart exists.

Contributions of all kinds are welcome: bug reports, verification cases, documentation, examples, and code. This guide follows the [FEBio contribution conventions](https://github.com/febiosoftware/FEBio/blob/master/CONTRIBUTING.md), with additions adopted from [Warp](https://github.com/NVIDIA/warp/blob/main/CONTRIBUTING.md).

## Contents

- [Before you start](#before-you-start)
- [Project setup](#project-setup)
- [Writing code](#writing-code)
- [Testing](#testing)
- [Branches and commits](#branches-and-commits)
- [Pull requests](#pull-requests)
- [AI-assisted contributions](#ai-assisted-contributions)
- [Licensing and sign-off](#licensing-and-sign-off)

## Before you start

### Reporting issues

Search the existing [issues](https://github.com/JolleyLab/FEBio-Warp-Python/issues) first to see whether the problem is already known.

When you open a new issue, include enough for someone else to reproduce it:

- Operating system and version
- Python version, Warp version, and FEBio-Warp commit
- Device: CPU or CUDA, and for CUDA the GPU model, driver version, and CUDA Toolkit version
- Steps to reproduce, ideally a minimal script or input file
- What you expected and what happened, including the log output
- If the report concerns a discrepancy with FEBio: the FEBio version, the `.feb` input, and the quantity that differs

### Planning a contribution

- Search existing issues and pull requests before starting.
- Report bugs and propose substantial changes in an issue before opening a pull request. Explain what the change addresses and which simulation workflow needs it.
- Code enters `src/warp_febio/` through a design note in `docs/design/`, a draft pull request, and review. Exploratory prototypes are welcome under `pilots/`; see the [README](README.md#how-code-enters-the-package).
- Discuss new features with the maintainers first. A new material model, element, or contact formulation should come with a concrete use case and an FEBio reference to verify against.
- An open issue is not an assignment. If you would like to take on an issue someone else filed, ask in the issue first.
- A fix for a bug you found yourself can go straight to a pull request.

## Project setup

See the [README](README.md) for installation:

```bash
git clone git@github.com:<username>/FEBio-Warp-Python.git
cd FEBio-Warp-Python
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Python 3.10 or newer is supported, and 3.11 is the tested version. Warp is pinned to an exact version in `pyproject.toml`. Do not change the pin as part of an unrelated pull request.

Run the test suite before you start to confirm your environment works:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Writing code

### Style

- Follow PEP 8, with `snake_case` function names and Google-style docstrings.
- Match the style of the file you are editing.
- Pass both `inputs` and `outputs` to `wp.launch()` in any code path that may be differentiated.
- No new required dependencies without discussion. Optional dependencies are considered case by case.

### Naming

Use descriptive names that say what a variable holds. Avoid abbreviations unless they are standard across the field. For example, prefer `face_indices` to `fc_ind` or `fi`.

Short names are acceptable where they match established continuum mechanics and FEBio notation, for example:

- `F`, `C`, `J` for the deformation gradient, right Cauchy-Green tensor, and volume ratio
- `S`, `P`, `sigma` for the second Piola-Kirchhoff, first Piola-Kirchhoff, and Cauchy stress
- `X`, `x`, `u` for reference position, current position, and displacement
- `i`, `j`, `k` for loop and node indices

Reuse existing terminology. When a quantity has a name in FEBio or elsewhere in this code base, use that name or note the correspondence in a comment.

### Comments

Comments should be reasonably verbose. Many readers will be checking the code against FEBio.

- State the formulation being implemented and its source (FEBio source file, FEBio Theory or User Manual section, or paper).
- Link to the relevant FEBio documentation or source where it helps a reader verify the math.
- Note units, reference versus current configuration, and any sign conventions.

### Formatting

Python is formatted with [Black](https://black.readthedocs.io/), pinned in the `dev` extra. Format before you commit:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m black src tests examples
```

## Testing

### What to include

- A bug fix should come with a test that reproduces the failure without the fix and passes with it.
- A new feature needs tests for its core behavior and important edge cases.
- Code meant to be differentiable should include a gradient check (for example, `wp.Tape` against finite differences).
- Changes to code that runs on CUDA must be tested on a supported CUDA GPU, even if the problem was found on CPU. Fixes for behavior shared by CPU and CUDA must address both. If you have not done this testing, say so clearly in the pull request description.
- Every test adds to the suite runtime, so favor meaningful tests of real behavior over exhaustive trivial variations.

### Verification against FEBio

Every contribution that changes numerical behavior needs evidence that it is correct. Following FEBio practice, that means example files and a description of the expected outcome.

- Provide the input (a `.feb` file or a script) and reference results. Reference results can be a log file or a data file with partial solution data.
- Code that implements mechanics also found in FEBio (materials, elements, contact, loads, solvers) should be tested against FEBio reference problems. Report the quantity compared, the tolerance, and the FEBio version. The FEBio test suite problems are the preferred reference (for example the `ma*.feb` cases for materials).
- Where no FEBio counterpart exists, use an analytical or manufactured solution, or explain what the change was checked against.
- Agreement with FEBio shows the port is faithful to FEBio. It is not validation against experiment, so describe results accordingly.
- Reference-tier code runs in double precision (`wp.float64`). State the precision used for any comparison.
- Keep reference files small. Large meshes and result files belong in the FEBio model repository or another linked location, not in git.
- Use synthetic or published data only. Never commit patient data or anything derived from it that could identify a patient.

### Writing and running tests

Tests use `unittest` and live in `tests/`.

```bash
# full suite
.venv/bin/python -m unittest discover -s tests -v

# a subset
.venv/bin/python -m unittest discover -s tests -p 'test_material*.py' -v
```

- Run kernel tests on CPU and, where available, on CUDA. Pass the device explicitly, through `wp.ScopedDevice(device)` or `device=device`, instead of relying on the default device.
- When a test cannot run in an environment (no CUDA device, missing optional dependency), skip it with a `unittest` skip decorator so it shows up in the skip count. Do not leave it out silently.
- Compare arrays with a stated tolerance, and choose the tolerance deliberately. For FEBio comparisons, record where the tolerance came from.
- Never call `wp.clear_kernel_cache()` or `wp.clear_lto_cache()` from a test. It is unsafe when tests run in parallel.
- Tests that do not depend on a device (parsers, file readers) can be plain test methods.

## Branches and commits

1. Fork the repository and clone your fork. Team members with write access may branch in the main repository.
2. Create a branch named `username/short-description`.
3. Write commits that stand on their own:
   - Imperative mood in the subject line ("Add Ogden material", not "Added").
   - Subject under about 50 characters. Use the body to explain why.
   - Reference issues as `(GH-123)`.
   - Sign off every commit with `git commit --signoff` (or `-s`) to certify the Developer Certificate of Origin.
   - A few logical commits, often one. No fixup or merge commits in the final history. Each commit should pass the tests.

To squash a branch onto the current `main`:

```bash
git fetch https://github.com/JolleyLab/FEBio-Warp-Python.git main
git reset --soft FETCH_HEAD
git commit -s
```

## Pull requests

Review time is limited, and focused changes with thorough validation are easier to review. To keep review efficient and the code base healthy, pull requests should meet these expectations:

- **Keep changes focused.** A pull request should address a single bug fix, feature, or improvement. Keep it small, avoid unrelated edits, and split unrelated changes into separate pull requests.
- **Include a test plan.** Explain how you verified the change and which devices it ran on, in enough detail that a reviewer can reproduce and confirm the result.
- **Link the issue.** If the pull request resolves an issue, include `closes #ISSUE_NUMBER` in the description or commit message.
- **Use drafts for early work.** Open exploratory or design-level work as a draft pull request. Early drafts are encouraged for anything that affects structure (interfaces, file formats, solver architecture), so reviewers can comment before the design settles.
- **Minimize review cost.** Well-tested, clearly described contributions with a clean commit history are reviewed faster. Contributions that take substantial reviewer effort to verify may be deprioritized or declined.

Open pull requests against `main`.

### Review process

1. When the pull request is ready, a maintainer assigns one or more reviewers. Reviewers are chosen for relevant expertise: FEBio or finite element mechanics, or Warp, GPU, or differentiable programming.
2. The reviewer either accepts the pull request or leaves comments.
3. Address the comments, then ping the reviewer in a comment so the review can continue.
4. Once accepted, the change is merged to `main`.

## AI-assisted contributions

AI-assisted contributions are welcome and are held to the same expectations as any other contribution. The person who opens the pull request is responsible for the changes and must be able to explain their behavior, verification, and known limitations.

## Licensing and sign-off

FEBio-Warp is released under the Apache-2.0 license. FEBio itself is MIT licensed. Code translated from FEBio must keep attribution to the FEBio source it derives from.

Rather than requiring a formal Contributor License Agreement (CLA), we use the [Developer Certificate of Origin](https://developercertificate.org/) to ensure contributors have the right to submit their contributions to this project. Please ensure that all commits have a [sign-off](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s) added with an email address that matches the commit author to agree to the DCO terms for each particular contribution.

```bash
git commit --signoff
```

### Developer Certificate of Origin

```text
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
