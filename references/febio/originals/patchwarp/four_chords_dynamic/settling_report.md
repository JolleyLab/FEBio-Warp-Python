# FEBio four-chord settling evidence

**The supplied outputs do not establish that the final dynamic state is statically settled.** The final step converged, but pressure was still increasing up to its endpoint, and there is no constant-pressure hold interval. The displacement history is measurably changing. This does not by itself prove appreciable dynamic disequilibrium: a sequence of nearly static equilibria would also move while pressure changes.

## Data and provenance

The archived XPLT extraction was checked against SHA256 hashes of all three supplied reference files before analysis. It contains the initial configuration plus 15 converged dynamic states. CSV times were used for interval lengths and checked against the lower-precision XPLT times. No simulation was rerun and no solver settings changed.

The log names the historical input, which differs from the shortened reference-output name:

`/Users/aslans1/Library/CloudStorage/OneDrive-Children'sHospitalofPhiladelphia/Documents/research/newton-patch/simulation2/with_chords/jobs/patch_2PM_4chords_mitral_pattern.feb`

Evidence: `reference/febio/chords/patch_4chords.log:27`. Possession of this path does not establish possession or exact identity of its original input bytes.

## Quantitative final-interval evidence

| Quantity | Value |
|---|---:|
| Penultimate output time | 0.0969791209 s |
| Final output time | 0.1 s |
| Final output interval | 0.0030208791 s |
| Pressure/load fraction | 96.9791209% → 100% |
| Pressure | 12,898.2230797 → 13,300 Pa |
| Penultimate maximum shell displacement | 1.344489314 mm |
| Final maximum shell displacement | 1.368857649 mm |
| Change between displacement maxima | 0.0243683352 mm |
| Largest individual nodal displacement-vector increment | 0.0298429459 mm, node 1050 |
| Mean nodal displacement-vector increment | 0.0153218912 mm |
| Increment nodal L2 / final displacement nodal L2 | 2.07385324% |
| Largest interval secant speed | 9.87889448 mm/s |
| RMS interval secant speed | 5.78897888 mm/s |
| Three-output quadratic endpoint derivative, maximum magnitude | 10.0649937 mm/s |

The pressure values use the user's explicit confirmation of a linear ramp from zero at 0 s to 0.0133 MPa at 0.1 s. The log independently lists the final coefficient, LINEAR interpolation and CONSTANT extrapolation, but omits the curve points (`patch_4chords.log:131`, `:141`). There are no output states beyond the ramp endpoint.

The secant and quadratic derivatives are **estimates derived from saved displacement**, not FEBio-exported Newmark velocities or accelerations. They cannot be treated as exact instantaneous velocity, kinetic energy, or a static force-balance diagnostic. All two PM anchor displacements remain exactly zero.

`settling.json` records every interval and the final A1/A2/B1/B2 displacement increments and secant vectors. `settling_analysis.py` reproduces these read-only calculations.

## What the files do and do not contain

XPLT dictionary contains nodal **displacement** and shell-domain **stress**, **Lagrange strain**, and **relative volume**. It does **not** contain velocity, acceleration, kinetic energy, strain energy, external work, reaction forces or discrete spring forces. The shell tensors are reduced element outputs, not a complete integration-point state suitable for exact energy/residual reconstruction.

The CSV fields `Ux/Uy/Uz` are averaged diagonal right-stretch components, not nodal displacement or velocity. `s1/E1` are averaged principal stress/strain reductions. They provide no missing dynamic-energy or force-balance evidence. Sources: `validation/comparisons/hinge_chords/chord_febio_metadata.json:4`, `validation/comparisons/hinge_chords/FEBIO_OUTPUT_DEFINITIONS.md:25`.

The final log residual quantity is 3.082100e-13, with required value zero because `rtol=0`. The log's `energy` and `displacement` rows are **nonlinear-iteration convergence norms**, not physical energy or velocity histories. Normal termination and the final converged dynamic step do not establish zero inertial force or static equilibrium. Sources: `patch_4chords.log:87`, `:1153`, `:1162`, `:1222`.

## Smallest justified follow-up

Prepare/run the same four-chord model as a **STATIC** FEBio problem, preserving mesh, top-only BCs, material, thickness, springs, anchors and final pressure. Confirm those settings against the original input if it becomes available; otherwise explicitly label the static deck as reconstructed from the validated no-chord input plus confirmed four-chord data. Compare final static top displacement and reconstructed chord lengths/forces against the existing PatchWarp static result. Use a nonzero stringent residual criterion, and retain the final convergence log.

An optional dynamic settling experiment is a continuation past 0.1 s at constant pressure, with explicitly exported velocity, acceleration, kinetic/strain energies and force/reaction diagnostics. That is a different test and is unnecessary for the immediate static-to-static comparison. No dynamic hold or exact velocity/energy evidence is fabricated from the current data.
