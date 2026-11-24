# FEBio-Warp verification harness: outline (draft for discussion)

**Draft for discussion, v1.0, November 24, 2026.** Nothing here is settled. It is written to be corrected by Steve Maas, Jeff Weiss, Ryan Black, the Warp team and the FEBio-Warp team, and the plan, the small problems and the asks are all open to change.

Scope: solid mechanics only; fluid, mixture and FSI problems are set aside for now. The code with this outline is small: a survey script that reads the FEBio test suite, a generator for 17 small FEBio problems, and a script that runs them. Statements about FEBio come from reading its source and the suite, and from runs of FEBio 4.13.0 built from source at commit `552138d` (Linux, GCC 13, no MKL, skyline solver, one thread). That build is not an official FEBio release.

## Purpose

One way, used by every part of FEBio-Warp, to show that a result agrees with FEBio, and to show where it does not yet. It applies to materials, elements, contact, loads and solvers alike, which is why it is proposed on its own and not inside the material models draft.

It should answer three questions on every pull request:

1. Did anything that agreed with FEBio stop agreeing?
2. What new FEBio problems does this change make runnable?
3. For each comparison, what was compared, at what tolerance, against which FEBio version?

## Summary

- **What it is.** A common way to compare FEBio-Warp with FEBio, built first on the FEBio test suite, then on a few small problems where the suite is thin for our applications, then on the valve and device models that are already public.
- **The suite.** 549 input files. 156 use only materials FEBio-Warp has, 194 are in scope but need a missing material, 35 are rigid only, and 164 are fluid, mixture or FSI and set aside. The suite stores one number per problem, so field references need FEBio runs of our own.
- **Small problems.** 17 FEBio input files, each mirroring or extending a suite problem, for the features the suite covers thinly: Lee-Sacks, HGO, chordae, contact potential including self-contact, mass damping on solids and shells, and a modulus on a load curve. All run in FEBio.
- **First results.** FEBio-Warp's material models agree with FEBio on stress and tangent to about 1e-12 at the material point, the limit of FEBio's log output. The suite's stored values reproduce on a FEBio build from source for most problems tried.
- **Application cases.** The open models in the FEBio model repository: the mitral valve models of project 134 and the pulmonary valve deployments of project 136.
- **The ask.** One basic valve case and one basic device case running on the open models as published, and an active FEBio build with Pardiso for the harness to compare against.
- **Questions for the FEBio team** are at the end; each was checked against the source first.

## Order of work

1. **Use everything the FEBio test suite already provides.** Every input file in the suite gets a manifest entry, including those FEBio-Warp cannot run yet. Nothing is requested from anyone until the suite has been checked for it.
2. **Fill the gaps with small problems of our own,** only where the suite is thin for what the applications need.
3. **Use the application models that are already open** in the FEBio model repository. The harness is built on what is open; cases that are not public are left out.

## What the FEBio test suite provides

Read from https://github.com/febiosoftware/TestSuite at commit `986005f` (November 11, 2026).

- **549 input files** in `Verify/`, 538 in format 4.0 and 11 in format 2.0.
- **Stored reference values per platform** (`code/linuxGoldStandards.py` and the macOS and Windows equivalents), 584 entries on Linux. For each problem: termination status, time steps, equilibrium iterations, right-hand-side evaluations, stiffness reformations, plot file size, and one value read from the first data record of the log file under `-task=test`. Optimization problems (`op`) store the final objective value and iteration count instead.
- **The 11 `oi` files have no stored value of their own.** They are forward models run only through the matching `op` optimization problem. This matters for chordae: `oi05` is one of the two nonlinear spring problems.
- **No field data.** The suite does not store displacements or stresses. A comparison of fields needs FEBio to be run.
- **The comparison is exact.** `runTests.py` compares each stored entry with `!=`, the log value as a 12-digit string.
- **The three platforms do not agree exactly with each other.** Of 563 problems with a numeric log value on all three platforms, 446 are identical, 28 differ by more than 1e-6 relative, 16 by more than 1e-4, and 8 by more than 1e-2. Solver counts differ across platforms in 71 problems. Example: `ma34` (uncoupled HGO) takes 126 iterations on Linux and 110 on macOS, and its log value differs in the sixth digit.
- **No license file** in the suite repository. FEBio itself is MIT.

Consequences:

- Reference fields for FEBio-Warp have to be generated by running FEBio once, under a pinned version, and stored.
- The suite's stored value checks that step: if our FEBio run of a problem does not reproduce the suite's value for that platform, the reference is not trusted.
- The spread across platforms is a measured guide for tolerances. A tolerance tighter than FEBio achieves between Linux and macOS on the same problem is not meaningful for that problem.

## Coverage of the suite today

`validation/febio_suite/survey.py` reads every input file and lists the material types it needs, including nested ones, with its elements, domains, contact types, loads and analysis type. The full result is `validation/febio_suite/coverage.json`. Against the three models on `main` (neo-Hookean, isotropic elastic, isotropic Lee-Sacks):

| Class | Problems |
|---|---|
| Use only materials FEBio-Warp has | 156 |
| In scope, but need a material that is missing | 194 |
| Outside the planned scope (fluid, biphasic, multiphasic, solute, FSI modules) | 164 |
| No deformable solid material (rigid bodies only) | 35 |

This counts materials only. The 156 still need the right elements, contact and solver features; by file prefix they are mostly `sh` (55), `co` (35) and `mi` (22) problems.

The single material additions that unlock the most problems: Mooney-Rivlin (21), transversely isotropic Mooney-Rivlin (9), Ogden (5), linear beam (4), Ogden unconstrained (4).

## What the suite covers for the driving applications

The first cases are chosen for the applications driving the project: mitral valve closure as in Aslan et al. (arXiv 2509.22865), and transcatheter pulmonary valve deployment as in Zelonis et al. (arXiv 2507.06337). The application column follows the open models described two sections below. Counts are problems within scope (the 385 that are not fluid, mixture or FSI), since out-of-scope problems cannot serve as references.

| Feature the applications rely on | Application | Suite problems in scope | Examples |
|---|---|---|---|
| Hexahedral (hex8) solid elements | Device (vessel) | 295 | most of the suite |
| Neo-Hookean material | Device (crimping tubes) | 220 | most of the suite |
| Quadrilateral (quad4) shell elements | Valve | 51 | `sh` series |
| Dynamic analysis | Valve, device | 44 | `dy` series |
| Sliding-elastic contact | Alternative contact for devices | 27 | `co07`, `co32` |
| Pressure load | Valve | 24 | `ma06`, `ma11` |
| Load curve on a material parameter | Device (frame modulus) | 13 | `ma20` to `ma25` |
| Isotropic elastic material | Device (frame) | 12 | `dy10`, `mi02` |
| Tetrahedral (tet4) solid elements | Device (frame) | 10 | `ar05`, `ar09` |
| Elastic shell domain, named explicitly | Valve | 7 | `sh36`, `sh60`, `sh61` |
| Beam elements | Device (beam frames, planned; not in the open models) | 4 | `bt02` to `bt05` |
| Three-field solid domain | Uncoupled materials in general; the open device model uses an ordinary solid domain | 4 | `fi03`, `ma34`, `ma56`, `ma61` |
| Sticky contact | Valve (dilation model) | 3 | `co42`, `sh30`, `sh31` |
| Triangular (tri3) shell elements | Device (crimping tubes) | 3 | |
| Local material axis | Device (vessel fibers) | 2 | |
| Nonlinear spring | Valve (chordae) | 2 | `di04`, `oi05` (no stored value) |
| Holzapfel-Gasser-Ogden, uncoupled | Device (vessel) | 1 | `ma34` |
| Contact potential | Valve, device | 1 | `co62` |
| Mass damping | Valve, device | 1 | `dy17` (a truss problem) |
| `normal displacement` boundary condition | Device (crimping tubes) | 1 | `ma49` |
| Wedge (penta6) solid elements | Device (vessel) | 1 | |
| HGO unconstrained | Device (vessel) | 0 | only `mp10`, multiphasic |
| Isotropic Lee-Sacks | Valve | 0 | none |

The general machinery (elements, dynamics, neo-Hookean, sliding contact, pressure) is well covered by the suite. The features most specific to these applications are thin: one contact potential problem, one with mass damping, one usable nonlinear spring problem with a stored value, one uncoupled HGO problem, and none with Lee-Sacks. No suite problem combines them the way the applications do (Lee-Sacks shells with contact potential, chordae and damping; a contracting frame under contact potential with a material load curve).

## Small problems where the suite is thin

The aim is to fill the gaps in the feature table for what the applications need, not to add breadth. Each problem mirrors a suite problem or extends one, is an ordinary FEBio 4.0 input file, and can be offered to the FEBio team for the suite. A first set of 17 is in `validation/cases/febio/`, written by `validation/cases/generate.py`. All 17 run to normal termination in FEBio; `validation/cases/check.py` runs them and applies the checks reported under "First results".

| Problem | Gap it fills | Mirrors or extends |
|---|---|---|
| `mp_lee_sacks_*` (uniaxial, biaxial, shear, general) | Lee-Sacks: no suite problem | Leaflet parameters of the mitral model in this repository |
| `mp_hgo_uncoupled_*`, `mp_hgo_unconstrained_*` (biaxial, general) | HGO: one in-scope problem, none for the unconstrained form | Parameters of `ma34`, at about 5 percent strain |
| `mp_neo_hookean_general` | None; a control for the method | Parameters of `co62` |
| `ch_chord_slack_tension` | Nonlinear spring through slack; `oi05` has no stored value | Chord A1 of the mitral model; extends `di04` |
| `bk_hgo_three_field` | A three-field HGO problem small enough for every pull request | `ma34`, reduced from 7,700 lines to a 2x2x2 block |
| `bk_mass_damping` | Mass damping on a solid | `dy17`, which uses truss elements |
| `bk_modulus_load_curve` | A modulus on a load curve, the effective-modulus frame law | `ma20` to `ma25`, which drive other parameters |
| `bk_contact_potential_patch` | A second contact potential problem, with non-matching surface meshes | `co62` |
| `sh_lee_sacks_pressure` | Lee-Sacks on shells under pressure, with dynamics and mass damping | The mitral model's material, thickness, pressure and damping on a 6x6 patch |
| `sh_self_contact_fold` | Contact potential self-contact of one shell surface | The mitral model's leaflet contact (same surface as primary and secondary); `co62` has two solids |
| `bk_ring_in_tube` | The device mechanism in one small problem: crimp under contact potential, then a modulus step | Zelonis et al.; combines `co62` and `ma20` |

The `mp_` problems are one hex8 element with every node driven by FEBio's `prescribed deformation` boundary condition, so the deformation is uniform and the logged deformation gradient, stress and tangent are point values. No suite problem uses that boundary condition.

One change is planned: drive the ring problem's tube with `normal displacement`, as the open device model does. After that the next additions are reduced versions of the open valve and device models.

## Open application models

The application models are already public in the FEBio model repository (https://repo.febio.org/modelRepo/UserProjects), so the harness refers to them there and nothing has to be contributed. They are fetched by a script and not copied into this repository.

| Project | Contents | Size |
|---|---|---|
| 134, Regurgitant mitral valve (Aslan et al., arXiv 2509.22865) | `P2Prolapse_Mitral.feb`, `Dilation_Mitral.feb` | 7,442 and 12,376 nodes |
| 136, TPVR RVOT Workflow (Zelonis et al., arXiv 2507.06337) | Six files: Harmony TPV-25 and Alterra in RVOT1 to RVOT3 | 65,011 nodes, 87,984 elements for `RVOT1_TPV25.feb` |
| 152, TPVR RVOT Sensitivity Analysis and Transannular Patch Positioning (Maheshwari et al., arXiv 2601.05410) | `sensitivity_baseline_sim.feb` and two patch cases | open, not used for now |
| 81, Mitral Valve (Wu et al., 2022) | One model | not yet read |

The valve file already in this repository (`references/febio/originals/cardiofem/valve_pressurization/mitral_p2prolapse.feb`) is byte for byte the project 134 file `P2Prolapse_Mitral.feb`.

What the files settle, read directly from them:

- **Valve (project 134).** Isotropic Lee-Sacks on quad4 elastic shells, contact potential with one surface as both sides, nonlinear spring chordae, pressure, mass damping, implicit dynamics. The dilation model adds sticky contact, tet4 elements and a rigid body.
- **TPVR (project 136, `RVOT1_TPV25.feb`).** The frame is `isotropic elastic` on tet4 elements with E = 4e7 on a STEP load curve that drops it to 0.75 of that value at t = 1. This answers two things the papers leave open: the frame law as run, and the load curve values. The vessel is uncoupled `Holzapfel-Gasser-Ogden` on hex8 and penta6 elements in an ordinary solid domain (not three-field), with a local material axis. The crimping tubes are neo-Hookean tri3 shells driven by a `normal displacement` boundary condition. Three contact potentials, one of them introduced in the second step; mass damping on the frame; two dynamic steps.
- **Four features of the device model are thin in the suite and not yet in the small problems:** the `normal displacement` boundary condition, a local material axis, penta6 elements and tri3 shells.

## First results

**The suite's stored values reproduce on the source build, within its limits.** Four problems chosen for the applications, run with `-task=test`:

| Problem | Feature | Solver counts | Log value, this build | Log value, suite (Linux) |
|---|---|---|---|---|
| `co62` | Contact potential | identical | 1.23961560006 | 1.23961560006 |
| `dy17` | Mass damping | identical | 1.93707056563 | 1.93707056563 |
| `di04` | Nonlinear spring | identical | 10.8125164485 | 10.8125164451 |
| `ma34` | Uncoupled HGO, three-field | 174 right-hand-side evaluations against 176 | 89.326418142 | 89.326408082 |

The differences in `di04` (3e-10) and `ma34` (1e-7) are inside the spread between the suite's own platforms for those problems.

A sweep of the 151 runnable problems that use only materials FEBio-Warp has (the 156, less five `oi` files), with a 30 second limit each on two cores:

| Outcome | Problems |
|---|---|
| Ran to normal termination | 105 |
| of those, log value identical to the stored Linux value | 81 |
| of those, within 1e-9 of it | 89 |
| of those, solver counts identical | 88 |
| Needs a linear solver this build lacks (no MKL, so no Pardiso) | 20 |
| Ran out of memory in the skyline solver | 4 |
| Other error | 1 |
| Over the time limit | 21 |

Six problems differ from the stored value by more than the suite's platforms differ from each other (`co07`, `dy04`, `mi48`, `sh26`, `sh27`, `sh45`; the largest is `dy04` at 1e-3). The different linear solver is the likely cause and this has not been examined further. The conclusion is that this build is adequate for the small problems and for most of the suite, and that pinned references should come from an official build with Pardiso.

**FEBio and FEBio-Warp agree at the material point.** Using the material models of the material models draft (branch `materials-interface-proposal`), ten states per case, relative to the largest reference value:

| Case | Cauchy stress | Spatial tangent |
|---|---|---|
| Lee-Sacks, uniaxial | 1.7e-12 | 3.2e-12 |
| Lee-Sacks, biaxial | 3.0e-12 | 3.7e-12 |
| Lee-Sacks, shear | 1.5e-12 | 5.1e-13 |
| Lee-Sacks, general | 2.2e-12 | 5.7e-13 |
| HGO unconstrained, biaxial | 1.3e-12 | 3.9e-12 |
| HGO unconstrained, general | 1.4e-12 | 4.8e-12 |
| Neo-Hookean, general | 2.6e-12 | 3.6e-12 |

This is at the limit of what FEBio's log can show (12 significant digits). It is the first comparison of FEBio-Warp against a FEBio run. The tangent comparison covers the 21 entries FEBio writes, which exclude the shear-shear block (question 1). Uncoupled HGO has FEBio references but no FEBio-Warp model yet.

**The other small problems,** against closed forms where one exists.

| Case | Check | Result |
|---|---|---|
| Chord | Force against 60 x max(0, elongation - 4), 25 states, 18 of them slack | exact to the digits written |
| Chord | Reaction at the driven node against the spring force | 5e-18 |
| Modulus on a load curve | Stretch against the neo-Hookean closed form as E drops from 1 to 0.25 | 8e-10 |
| Mass damping | Final displacement against the static solution; peak overshoot 5 percent | 2e-9 |
| Three-field HGO block | Stress uniform across the 8 elements; volume change | 2e-12; 1e-7 |
| Contact potential patch | Runs; reaction rises monotonically to 0.673 | no independent reference yet |
| Lee-Sacks shell under pressure | Runs; center deflection settles at 2.446 mm, overshoot 0.5 percent | no independent reference yet |
| Shell self-contact | Runs; closing force rises from 0.26 to 18.6 over the five steps within the potential's range, final gap 0.2 mm | no independent reference yet |
| Ring in tube | Runs; hoop force 0.050 crimped and soft, 0.197 after the modulus is raised fourfold, 0.048 after the tube opens | no independent reference yet |

The four cases without an independent reference are FEBio results to compare FEBio-Warp against once it has elements and contact. They show that FEBio runs each feature on a problem small enough to debug.

## FEBio build needed to run the open models

The source build used for the results above (FEBio 4.13.0 at commit `552138d`, GCC, no MKL) runs the small problems and most of the suite. It cannot run the open application models. What a build needs:

1. **A linear solver for non-symmetric matrices, in practice Pardiso through Intel MKL (oneAPI).** `RVOT1_TPV25.feb` and `Dilation_Mitral.feb` request a non-symmetric stiffness. Without MKL, FEBio falls back to its skyline solver, which rejects non-symmetric matrices (the same error stops 20 suite problems) and runs out of memory on large ones (4 suite problems). The official FEBio release binaries include Pardiso.
2. **A FEBio version that reads the files as published.** FEBio 4.13 development source stops on both valve files at `<laugon>PENALTY</laugon>` in the contact potential block ("invalid value"). The files were evidently written for a version that accepted that text. Either the FEBio version the models were run with, or the same files with that line removed. The TPVR file has no such line. Whether results match after removing the line has not been tested.
3. **No optional libraries beyond that.** The models use only standard FEBio features: no plugins, and no need for LEVMAR, MMG, HYPRE or FFTW. zlib only if plot files are compressed.
4. **OpenMP and a multi-core machine.** The TPVR model has about 195,000 displacement unknowns and 220 time steps; the valve models take 100 to 200 dynamic steps on 7,000 to 12,000 nodes. Run times and memory on the original machines are not known to us.
5. **Recorded version.** Whatever build is used, the harness stores its version string and commit with every reference, and checks it against the suite's stored values first.

The simplest route is an official FEBio release for Linux, at the version the open files were run with, installed on a lab machine or supplied to the agent workspace.

## What we have and what we need

**Have**

- The FEBio test suite at a pinned commit, surveyed and classified.
- 17 small problems that fill its gaps for these applications, all run in FEBio.
- A source build of FEBio adequate for the small problems and 105 of 151 suite problems tried.
- The open valve and TPVR input files (projects 134, 136, 152, 81).
- FEBio-Warp material models agreeing with FEBio at the material point for Lee-Sacks, HGO unconstrained and neo-Hookean.

**The ask, for discussion: two basic running cases on the open models, and an active FEBio build to compare against.**

1. **A basic valve case.** The open mitral model `P2Prolapse_Mitral.feb` (project 134), running start to finish as published. Seda Aslan is the natural person to confirm it.
2. **A basic TPVR case.** One open deployment, `RVOT1_TPV25.feb` (project 136), running start to finish as published. Jalaj is the natural person to confirm it.
3. **An active FEBio build for the harness.** An official FEBio release with Pardiso, on a machine the harness can use, so the same two cases and the suite can be run on demand and FEBio-Warp compared against them. Without it there is nothing to compare with beyond the small problems.

Only the basics on open models are asked for: no new cases, no data that is not already public, and no scripts. For each of the two cases, what is needed to run it and to know the run is right:

| Item | Valve, `P2Prolapse_Mitral.feb` | TPVR, `RVOT1_TPV25.feb` |
|---|---|---|
| FEBio version the file was run with (and FEBio Studio version, if it was exported from it) | Needed. The file does not read in the current FEBio source because of `<laugon>PENALTY</laugon>` | Needed, to pin the reference |
| Whether the file runs as published on a current release, or needs an edit | Is removing the `laugon` line the right fix? | Any known issue on current releases? |
| Linear solver and any FEBio configuration file settings | Pardiso assumed | Pardiso assumed; non-symmetric |
| Cores, memory and wall time of a typical run | To size the machine and set time limits | Same; this is the large one (65,011 nodes) |
| One result to check a run against | A log file from a completed run, or a few numbers from it (final time reached, final contact area, peak first principal stress) | A log file from a completed run, or a few numbers from it (final time reached, peak vessel stress, peak frame displacement) |

The other open files (the dilation valve model, the other five deployments) follow once these two run. Not needed: input files, the frame material law, the modulus load curve, or post-processing scripts. The first three are in the open files. The valve files already write displacement, first principal stress and strain, contact gap and contact area to their own output files, and the harness compares those and other quantities FEBio writes directly. None of the three files read depends on an external input file or a plugin.

**Review asked of Steve Maas and the FEBio team.** See "Open questions". In short: which FEBio build matches a suite commit, whether suite files may be redistributed, and whether small application-derived problems would be welcome in the suite.

**Review asked of Jeff Weiss.** The comparison quantities and tolerances per tier, and suggestions for analytical cases in tier 4.

**Review asked of Ryan Black.** The manifest and comparator design, in particular whether the reference file layout should be usable from the MFEM side as well.

**Asked of the Warp team.** A CUDA device for nightly runs (not yet arranged), and Warp's own practice for FP32 against FP64 tolerances and for tape gradient checks, so ours follow it.

## Tiers

Each tier needs less of FEBio-Warp than the one after it, so verification can start before a full solver exists.

| Tier | What is compared | Needs from FEBio-Warp | Reference comes from |
|---|---|---|---|
| 0. Material point | Cauchy stress and spatial tangent for a given deformation gradient | A material model only | FEBio log output `Fxx` to `Fzz`, `sx` to `sxz`, `cxxxx` and the other 20 tangent components, from problems with uniform deformation |
| 1. Element | Nodal forces for one element or a small patch under fully prescribed nodal displacements | Material and element | FEBio run of the same patch, read as reaction forces |
| 2. Suite problem | Nodal displacements, stresses, reaction forces, contact quantities, at stated times | Reader, elements, loads, contact, solver | FEBio run of the suite problem, cross-checked against the suite's stored value |
| 3. Application | Fields FEBio writes directly: displacement, stress, strain, contact pressure and gap, spring force | Everything | The open FEBio models (projects 134 and 136), run on an official FEBio build |
| 4. Outside FEBio | Analytical solutions; published experimental curves | Varies | Named source, labelled as not a FEBio comparison |

Six points from the FEBio source and from runs shape tiers 0 to 2:

- **Element log variables are averages over the element's integration points** (`FEBioMechData.cpp`). A stress and a deformation gradient from the log are a matched pair only when the deformation is uniform over the element. Tier 0 therefore uses single-element and uniform-deformation problems, not arbitrary suite problems.
- **Log data is written with 12 significant digits; plot files store single precision.** A `format` string on a data record does not raise this: that path writes six digits (`FECore/DataRecord.cpp`; confirmed by a run). So a comparison against FEBio can be no tighter than about 1e-11 relative from the log, and about 1e-7 from a plot file. FP64 comparisons read the log without a format string (question 1).
- **The 21 tangent log variables are read by position, not by name.** `FELogElemElasticity` returns entry n of a `tens4dmm`, which stores the full 6x6 by column. The 21 values are therefore the xx, yy and zz columns in full and three entries of the xy column; the variable named `cyyyy` holds c_xxzz, and the shear-shear entries are not written. Reading them this way gives the agreement above (question 1).
- **A log record addressed by node set reports internal node numbers.** They match the node ids only when ids are contiguous, so the small problems number nodes contiguously.
- **A fixed degree of freedom logs no reaction force.** A reaction is written only for a prescribed one, so where a reaction is the quantity of interest the small problems prescribe zero and do not fix.
- **With every node prescribed, the reaction forces are the element's internal forces.** Tier 1 needs no special output from FEBio for forces. Element stiffness is the open part (question 3).

Tier 0 is the reason to start now: it gives a new material model its first evidence without waiting for FEBio-Warp to solve anything. Tier 4 is where models with no FEBio counterpart, such as superelastic nitinol, are checked.

**Named application cases (tier 3).** In the harness from the start, marked "not yet supported" with the missing features listed until FEBio-Warp can run them:

- **Valve:** `P2Prolapse_Mitral.feb` first, then the dilation model of project 134. Compared on what the files already write (leaflet displacement, first principal stress and strain, contact gap, contact area) and on chordal spring force.
- **Transcatheter pulmonary valve:** `RVOT1_TPV25.feb` first, then the other five open deployments of project 136. Compared on frame and vessel displacement, vessel stress and strain, and contact pressure and gap.

## Parts

- **Case manifest.** One small text entry per case: suite problem and commit, FEBio version, tier, quantities compared, tolerance with its reason, tags (material, element, contact, analysis type), and status. Status is one of passing, known difference with an explanation, not yet supported with the missing feature named, or out of scope. All 549 suite problems have an entry, generated from the survey. A case never disappears silently.
- **Reference generator.** Runs a pinned FEBio build, adds the log file requests a tier needs, and converts the output to a single array file with provenance: FEBio version, suite commit, platform, date, and the suite's stored value for that problem and platform.
- **Reference store.** Outside the main repository, addressed by content hash, fetched by a script. Small tier 0 and tier 1 files can live in git. The contributing guide already asks that large files stay out.
- **Reader.** `pyfebio` for input files and plot files, so the harness and the solver read models the same way.
- **Runner and comparator.** Runs FEBio-Warp on a case and compares against the stored reference with the manifest's tolerance. Tolerances differ by precision (FP64 tight, FP32 looser), take the suite's spread across platforms as a floor, and are never chosen to make a case pass without a recorded reason.
- **Report.** A table per pull request: cases passing, newly passing, newly failing, and the suite coverage numbers above. Stored as a small file so the trend is visible over time.
- **Gradient checks.** For selected cases, a derivative from the Warp tape against finite differences. Agreement with FEBio says nothing about gradients, and differentiability is a goal of the project.
- **Performance.** Tracked by the same runner but reported separately. A speed number is never mixed into a correctness result.

## Continuous integration

- **Every pull request:** tiers 0 and 1, CPU, FP64. Minutes.
- **Nightly:** tier 2 on CPU and on a CUDA device, FP64 and FP32.
- **Weekly or on request:** tier 3.

## Data rules

- Synthetic and published data only. No patient data or anything derived from it that could identify a patient, as the contributing guide states.
- The open application models are public in the FEBio model repository. They are fetched from there by a script and are not stored in this repository.
- Suite files are referenced by commit, not copied, until redistribution is confirmed (question 2).
- Every reference file names the FEBio version that produced it. References are regenerated deliberately, in their own pull request, never as a side effect.
- Agreement with FEBio shows the port is faithful to FEBio. It is not validation against experiment, and reports say which of the two a number is.

## Status and next steps

Done:

- Survey of the suite, and its classification against FEBio-Warp's materials.
- 17 small problems, all running in FEBio, with material point comparisons for the materials FEBio-Warp has.
- Reproduction of the suite's stored values on a source build, for the problems that use those materials.

Next:

1. Agree the ask: an active FEBio build, and the two basic open cases running on it.
2. Repeat the reproduction for all 385 in-scope problems on that build, which then becomes the pinned reference.
3. Generate the manifest for all 549 suite problems from the survey, with class, tags and the stored value per platform.
4. Define the report, then add tier 1 with the first solid element and tier 2 as the reader and solver allow.

None of this needs FEBio-Warp's solver until step 4.

## Open questions

Each was checked against the FEBio source and the suite first; what was found is stated with the question.

1. **Log output.** Four things we found, in case we are misreading them. (a) The element log variables for stress, deformation gradient and tangent are averages over integration points, written with 12 significant digits; is there a per-point or full-precision output we have missed? (b) The tangent variables `cxxxx` to `cxzxz` appear to return the first 21 entries of a 36-entry `tens4dmm`, so the names do not match the values and the shear-shear entries are missing. Our numbers agree with FEBio only when read that way. Is that intended? (c) A `node_data` record given a `node_set` lists internal node numbers and returns zeros when node ids are not contiguous. (d) The log reports zero reaction force at a fixed degree of freedom. None of these blocks us.
2. **Redistribution.** The suite repository has no license file. May its input files, and results derived from them, be redistributed with FEBio-Warp, or should they stay referenced by commit?
3. **Element stiffness.** The tangent diagnostic prints an element stiffness matrix, but only for its built-in scenarios and at about six digits, and no suite problem uses a diagnostic. The global matrix can be written in Harwell-Boeing form from the FEBio prompt. Which route would you recommend for a tier 1 stiffness reference?
4. **Version pinning.** The suite's stored values are updated from the logs of FEBio's GitHub Actions runs, so a suite commit corresponds to a FEBio development build and not to a release. Which FEBio commit or release should references be pinned to, and would you prefer we reuse those CI log artifacts or run FEBio ourselves?
5. **Tolerance.** The suite compares for exact equality per platform, while the platforms differ from each other as measured above. Is that spread a fair guide for our tolerances, and are the eight problems that differ by more than 1 percent across platforms ones to treat as unsuitable references?
6. **New suite problems.** Would the FEBio team want the small application-derived problems listed above added to the suite?

Not asked here: driving a material through a prescribed deformation without a mesh is already raised in the material models draft (the material test diagnostic offers four fixed scenarios; `FEBCPrescribedDeformation` on one element is the alternative). Where the reference store lives is a decision for this project, not a question for reviewers.
