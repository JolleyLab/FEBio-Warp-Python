# Native quad shell with four FEBio-law chords

Surface-integrated displacement-vector error is **0.13488205%** against the supplied FEBio four-chord output. Maximum displacement is **1.368144480 mm** versus **1.368857649 mm** (difference -0.052100%).

The previous hinge-shell chord error was 10.89597%. This new result strongly supports the native quad shell/chord coupling. It is not an exact equilibrium-to-equilibrium comparison: the validated PatchWarp solver is static, while FEBio saved a dynamic Newmark state at t=0.1 s. The previous hinge run also used a slightly different pressure. No tuning was performed.

## Model and comparison

The unchanged native quad shell uses full 3D StVK, 2×2×2 integration, E=1 MPa, nu=0.3, thickness=0.5 mm, density=1000 kg/m³, and 13,300 Pa follower pressure. The 136 top perimeter nodes are fixed; bottom/director DOFs remain free. Two appended external PM anchors are fixed. Existing bilateral GPU springs supply W=k(L−L0)²/2, k=30 N/mm, with exact geometric/material tangent and zero initial pretension. No compression clipping is introduced.

The cached tangent includes shell, pressure and chord contributions exactly once. The original director-paired preconditioner is retained with chord diagonal contributions and fixed anchor blocks. Newton tolerance remains 1e-7 N; CG relative tolerance 0.01, absolute 1e-12, maximum 500 iterations. Ten static pressure increments preserve the validated benchmark continuation.

Native FEBio shell node IDs and quad connectivity match exactly: no interpolation/projection is needed. Surface error uses the established bilinear displacement interpolation and 8×8 Gauss integration on the reference quad area. Shell-coordinate roundoff between text input and XPLT is at most 6.978e-8 mm; PM-coordinate differences are 1.388e-6 and 3.586e-6 mm.

The chord .feb input is not available. Material, thickness and top-only boundary constraints therefore remain those inherited from the validated patch input; their complete sets cannot be independently recovered from the chord output. Pressure 0.0133 MPa and the spring law are confirmed by the existing source audit/user input.

## Attachments

Displacement vectors are global Cartesian values in mm.

| Attachment (FEBio ID) | PatchWarp (x,y,z) | FEBio (x,y,z) | Vector error [mm] | Relative error |
|---|---|---|---:|---:|
| A1 (1891) | (-0.07008643, -0.35679214, 0.00884234) | (-0.07010441, -0.35720384, 0.00887110) | 0.00041310 | 0.11345% |
| A2 (1868) | (-0.10397890, -0.48487443, 0.04720960) | (-0.10396273, -0.48539996, 0.04727700) | 0.00053008 | 0.10630% |
| B1 (1274) | (-0.26253912, -0.59250645, 0.03067467) | (-0.26245961, -0.59233272, 0.03065720) | 0.00019186 | 0.02958% |
| B2 (3) | (-0.18942572, -0.59299807, 0.01924305) | (-0.18936965, -0.59295076, 0.01924149) | 0.00007338 | 0.01178% |

Rounded user target coordinates map to these exact attachment nodes; target-to-node distances for A1/A2/B1/B2 are 4.14482e-5, 6.49108e-6, 3.36592e-5, and 1.23249e-5 mm respectively. These are target rounding distances, not displacement errors.

## Chord lengths and forces

Each entry below is **PatchWarp / FEBio**. Lengths and elongations are mm; forces are N. FEBio forces are **reconstructed**, using 30*(L−L0) from its saved endpoints. They are not directly exported discrete-force values or inferred from shell CSV stress.

| Chord | Initial length | Final length | Elongation | Force | Force difference |
|---|---:|---:|---:|---:|---:|
| PM1->A1 | 6.00002410 / 6.00002289 | 6.00220613 / 6.00220091 | 0.00218203 / 0.00217803 | 0.06546084 / 0.06534077 | +0.18376% |
| PM1->A2 | 5.73676615 / 5.73676511 | 5.74144859 / 5.74144321 | 0.00468245 / 0.00467810 | 0.14047342 / 0.14034303 | +0.09290% |
| PM2->B1 | 5.99999260 / 5.99999619 | 6.00439528 / 6.00439559 | 0.00440268 / 0.00439940 | 0.13208027 / 0.13198205 | +0.07442% |
| PM2->B2 | 5.96967211 / 5.96967573 | 5.97308759 / 5.97309042 | 0.00341548 / 0.00341469 | 0.10246443 / 0.10244070 | +0.02317% |

Reconstructed force differences include the finite precision of FEBio endpoint data. All four springs are tensile at the compared state.

## GPU cost and differentiation

| Quantity | Value |
|---|---:|
| Warm solve time, Quadro RTX 8000 | 0.383225 s |
| Newton iterations | 43 |
| CG iterations | 3,355 |
| Final static residual | 5.72597e-9 N |

Time includes GPU solves, residual/tangent/preconditioner assembly, line search and host synchronization; excludes compilation, warmup, mesh setup and output. The prior no-chord native result was 0.379 s / 42 Newton / 3,336 CG: chord overhead is small, although single-run timing variation precludes a precise overhead percentage. FEBio log solve time 1.63376 s is context only: different machine, dynamic history and output costs.

Coupled cached tangent finite-difference relative error is 1.66e-9. Equilibrium sensitivity field errors versus finite differences are 3.10e-8 for pressure, 6.56e-6 for uniform chord stiffness, and 2.96e-7 for uniform chord rest length. Tangent/adjoint derivatives agree within 4.18e-9 relative. All checks pass. Existing E and nu sensitivities remain available.

Finite-difference verification uses tighter 1e-12 N equilibrium solves, separate from unchanged 1e-7 N benchmark runs. An initial 3 N/m stiffness perturbation at 1e-10 N verification accuracy was contaminated by solve noise; the final 30 N/m perturbation and stricter verification resolve the tiny stiffness response. Physical benchmark parameters and tolerances were not changed.

The no-chord regression also passes: maximum position changes are 1.39e-14 mm for the pressure state and 4.34e-14 mm for the A2-loaded state. Independent review is in `independent_review.md`. Complete numeric values are in `comparison.json`, `attachments.csv`, `chords.csv`, and `node_errors.csv`; field arrays/integrals are in `fields.npz`.

## Reproduce

From the PatchWarp repository:

```bash
.venv/bin/python -m patchwarp.quad_chord_benchmark
.venv/bin/python validation/comparisons/native_quad_chords/compare.py
.venv/bin/python -m patchwarp.quad_chord_check
```

The next comparison, if tighter agreement is needed, is a static FEBio four-chord reference with the same inputs. The current evidence does not identify the remaining 0.13488% entirely as transient error.
