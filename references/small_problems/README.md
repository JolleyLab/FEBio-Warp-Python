# Small FEBio inputs and their FEBio references

Twenty-two small FEBio 4.0 input files for features the FEBio test suite
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
source without MKL, not an official release. The two chord networks added on
November 27 (`ch_truss_chords`, `ch_spring_chords`) come from the build at
commit `e4df3f8` made by `tools/build_febio.sh`; each reference names its build.

FEBio-Warp is compared with these references in `tests/`. Compared today: the
material point inputs (`mp_`); the three chord inputs (`ch_`); the two dynamic
inputs, `bk_mass_damping` and `sh_lee_sacks_pressure` (`tests/test_dynamics.py`);
and two contact inputs, `bk_contact_potential_patch` and `sh_self_contact_fold`
(`tests/test_contact_in_run.py`). The others wait on the features they need:
`bk_hgo_three_field` on the three-field element, `bk_ring_in_tube` on a
prescribed deformation, `bk_modulus_load_curve` on its comparison being written.

Prefixes: `mp_` material point (one element, uniform deformation), `ch_`
chord, `bk_` small solid problems, `sh_` shell problems.
