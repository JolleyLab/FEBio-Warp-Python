# Small FEBio problems for the verification harness

Seventeen small FEBio 4.0 input files that fill gaps in the FEBio test suite
for the valve and device applications. Each mirrors or extends a suite
problem. See [docs/VERIFICATION.md](../../docs/VERIFICATION.md) for what each
one is for and for the results. All geometry is generated; nothing here is
patient data.

| File | Purpose |
|---|---|
| `generate.py` | Writes the input files to `febio/` and the index `cases.json` |
| `febio/*.feb` | The input files, committed so they can be read and run without Python |
| `check.py` | Runs every file in FEBio and applies the check that fits it |
| `results.json` | Output of the last `check.py` run, with the FEBio version |

```bash
python validation/cases/generate.py
FEBIO_BIN=/path/to/febio4 python validation/cases/check.py
```

`check.py` compares FEBio with FEBio-Warp's material models when the checkout
has the material interface (`warp_febio.materials.model`). On a checkout
without it, the material point cases report only that FEBio applied the
requested deformation.

Prefixes: `mp_` material point (one element, uniform deformation), `ch_`
chord, `bk_` small solid problems, `sh_` shell problems.
