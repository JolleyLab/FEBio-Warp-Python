# FEBio test suite survey

Reads the FEBio test suite (https://github.com/febiosoftware/TestSuite) and
records what each problem needs, so we can say which suite inputs FEBio-Warp
should be able to run as it grows. The suite itself is not copied into this repository.

| File | Purpose |
|---|---|
| `survey.py` | Classifies every input file in the suite by materials and tags its features |
| `coverage.json` | Output of `survey.py` at suite commit `986005f` |

```bash
git clone https://github.com/febiosoftware/TestSuite
python validation/febio_suite/survey.py TestSuite --out validation/febio_suite/coverage.json
```

When a material model is added to FEBio-Warp, add its FEBio type string to
`SUPPORTED_MATERIALS` in `survey.py` and re-run. See
[docs/VERIFICATION.md](../../docs/VERIFICATION.md).
