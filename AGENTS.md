# FEBio-Warp Development Guidelines

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [README](README.md) before changing code. This file holds the workflow instructions for coding agents; CONTRIBUTING.md is authoritative for style, testing and review. The layout follows the `AGENTS.md` files in [Warp](https://github.com/NVIDIA/warp/blob/main/AGENTS.md) and [Newton](https://github.com/newton-physics/newton/blob/main/AGENTS.md).

## Environment and tooling

- Use `uv` for all commands. Prefer `uv run` to run Python (`uv run script.py`, `uv run -m module_name`), or activate `.venv` first. Do not use the system Python directly. Fall back to `venv` and `pip` only if `uv` is unavailable.
  - Set up: `uv sync --extra dev`
  - With additional packages: `uv run --with rich script.py`
- Lint and format with Ruff through pre-commit: `uvx pre-commit run --files <files>` for changed files, or `uvx pre-commit run -a` for all files. Do not use Black, isort or flake8, and do not add them as dependencies.
- Let uv take the Python version from `.python-version`.
- If a change alters dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock` with it. Do not change the Warp pin as part of an unrelated change.
- Never use `python -c "..."` to run temporary scripts that define `@wp.kernel` functions. Always write to a `.py` file, because Warp's code generation calls `inspect.getsourcelines()`, which fails for code that is not in a file.
- Pin GitHub Actions to commit hashes, not tags, and add a comment with the corresponding tag.
- Tooling follows Warp. Do not change the Python version, the Ruff or uv versions, the line length or the lint rules on their own; see [docs/WARP_ALIGNMENT.md](docs/WARP_ALIGNMENT.md) for what is tracked and how an update is applied. `uv run tools/check_warp_alignment.py` reports the current differences.

## Where code goes

- `src/warp_febio/` holds designed and reviewed code. Code enters it through a design note in `docs/design/`, a pull request and review. Plans and open questions live in the design notes, not in pull requests.
- `pilots/` holds the initial prototypes. They are not maintained, are excluded from linting, and are not a starting point for new work. Do not import from `pilots/` in the package or its tests.
- `references/` holds FEBio inputs and results. Do not reformat or edit these files.
- The targets for new work are FEBio's own test suite and the open models in the FEBio model repository, not the pilot data.

## Mechanics conventions

- FEBio is the reference for the mechanics. When implementing something FEBio has, read the FEBio source and follow its formulation, input names and output definitions. Build the internals for Warp (GPU execution and differentiation).
- Reference-tier code runs in double precision (`wp.float64`).
- Comments are reasonably verbose. State the formulation and its source (FEBio source file, FEBio manual section, or paper), and link to FEBio documentation where it helps a reader check the math.
- Pass both `inputs` and `outputs` to `wp.launch()` in any code path that may be differentiated.
- Write "Warp", not "NVIDIA Warp", in documents and comments. Capitalize proper names in docstrings and error messages (NumPy, Warp, FEBio).
- Never commit patient data or anything derived from it that could identify a patient.

## Tests

Always use `unittest`, not pytest.

```bash
uv run -m unittest discover -s tests -v                              # quick tier, CPU
WARP_FEBIO_TEST_LEVEL=full uv run -m unittest discover -s tests -v   # full tier
WARP_FEBIO_TEST_DEVICE=cuda:0 uv run -m unittest discover -s tests -v   # on a GPU
uv run -m unittest discover -s tests -p 'test_material*.py' -v       # a subset
```

- Test modules that use Warp import `DEVICE`, `setUpModule` and `tearDownModule` from `tests/support.py`; slow tests carry `@full_tier(...)` with a fast case kept in the quick tier. See CONTRIBUTING, "Writing and running tests".
- Run the full tier on CPU and, for device code, on a GPU before a change is merged; CI runs it on CPU only. The `gpu (manual)` workflow uses a billed runner: never start it unless explicitly asked.

- Verify that a regression test fails without the fix before committing.
- Code that implements mechanics also found in FEBio is tested against an FEBio reference from the same input. Record the quantity compared, the tolerance and the FEBio version.
- Code meant to be differentiable includes a gradient check (for example `wp.Tape` against finite differences).
- Use `np.testing.assert_allclose()` instead of `np.allclose()` so failures report the difference.
- Skip tests that cannot run in an environment (no CUDA device, missing optional dependency) with a `unittest` skip decorator. Do not leave them out silently.
- Never call `wp.clear_kernel_cache()` or `wp.clear_lto_cache()` in test files.
- Avoid timing-based assertions. They are unreliable on shared CI runners.
- State plainly in the pull request what was and was not run, including whether a GPU was used.

## Docstrings

Follow Google-style docstrings. Document `__init__` parameters in the class docstring, and do not repeat default values from signatures.

## Commits and pull requests

- Create a feature branch named `username/short-description` before committing. Do not commit to `main` unless a maintainer asks for it.
- Use imperative mood in commit messages ("Fix X", not "Fixed X"), with a subject of about 50 characters. The body explains why.
- Always use `git commit --signoff` (or `-s`) to add a `Signed-off-by` line (DCO).
- Run `uvx pre-commit run -a` and the test suite before committing.
- When opening a pull request, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Plan in a design note under `docs/design/`, not in a long-lived draft pull request. Keep pull requests short-lived and do not stack them.
- Never push, open a pull request, or set, change or clear reviewers or assignees unless explicitly asked.
