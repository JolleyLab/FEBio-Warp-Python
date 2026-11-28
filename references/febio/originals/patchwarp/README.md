# PatchWarp FEBio verification

FEBio inputs and FEBio results for a patch of 1863 quad4 shells, from the
pilot phase. Each folder keeps the input with the FEBio output it came with.

| Folder | Analysis | Compared with FEBio-Warp in |
| --- | --- | --- |
| `static_pressure_and_a2` | Static: pressure only, and a point load | `tests/test_run_path_pressure.py` |
| `four_chords_static_prepared` | Static: pressure, four chord springs. FEBio 4.13.0 was run on it on November 27, 2026; `febio_final_state.json` holds the final state | `tests/test_truss_and_spring_elements.py` (full tier, GPU) |
| `a2_dynamic` | Dynamic: a point load | `tests/test_dynamics.py` (full tier, GPU) |
| `pressure_dynamic` | Dynamic: a pressure ramp | Same |
| `four_chords_dynamic` | Dynamic: a pressure ramp, four chord springs | Same |

The dynamic runs were made with BFGS at `dtol` 1e-3, so FEBio-Warp is compared
with them to that tolerance: it takes FEBio's 15 steps at FEBio's times and
ends within 1e-6 to 4e-6 of FEBio's final displacement.
