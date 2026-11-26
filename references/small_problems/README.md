# Small FEBio inputs and their FEBio references

Twenty small FEBio 4.0 input files for features the FEBio test suite
covers thinly for the valve and device applications, with the FEBio results
for each. Each input mirrors or extends a suite problem. All geometry is
generated; nothing here is patient data. See
[docs/design/verification.md](../../docs/design/verification.md).

| Path | Contents |
|---|---|
| `inputs/*.feb` | The FEBio input files |
| `febio/*.json` | What FEBio produced for each input: logged data, FEBio version and build, and the hash of the input it came from |
| `generate.py` | Writes the input files and the index `cases.json` |
| `make_references.py` | Runs FEBio on each input and writes the reference. It records FEBio's output and checks nothing |

```bash
python references/small_problems/generate.py
FEBIO_BIN=/path/to/febio4 python references/small_problems/make_references.py --build "description of the build"
```

The references here are provisional: they come from a FEBio build from
source without MKL, not an official release.

FEBio-Warp is compared with these references in `tests/`. Only the material
point inputs (`mp_`) can be compared today; the others wait on FEBio-Warp
being able to read and run a FEBio input file.

Prefixes: `mp_` material point (one element, uniform deformation), `ch_`
chord, `bk_` small solid problems, `sh_` shell problems.
