## What does this change, and why?

<!--
Describe the problem or need and the approach. Link the design note in
docs/design/ if there is one. Use "Closes #123" only if merging this pull
request fully resolves the issue; otherwise "Part of #123".
-->

## Validation

<!--
Say how the change was checked, in enough detail that a reviewer can repeat it.
-->
- Tests added or changed:
- FEBio reference (input file, quantity compared, tolerance, FEBio version), or N/A:
- Devices the full tier ran on (CPU, and the GPU model if CUDA; say so clearly if no GPU was used):

## Checklist

- [ ] I have read the [contributing guide](https://github.com/JolleyLab/FEBio-Warp-Python/blob/main/CONTRIBUTING.md).
- [ ] `uvx pre-commit run --all-files` passes.
- [ ] The full tier passes: `WARP_FEBIO_TEST_LEVEL=full uv run -m unittest discover -s tests`, and with `WARP_FEBIO_TEST_DEVICE=cuda:0` for code that runs on the device.
- [ ] Open design questions are in the design note's decision points, not only in this pull request.
- [ ] The change is focused, and every commit is signed off (`git commit -s`).
