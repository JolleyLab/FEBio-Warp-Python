# FEBio-Warp material models: architecture proposal

Draft v0.3, November 24, 2026. For Steve Maas, Jeff Weiss, Ryan Black, the Warp team and the FEBio-Warp team. Nothing here is merged. This version replaces v0.2 and follows Steve's review of this pull request and Jeff's earlier comments. The earlier pilot code is no longer the starting point; the question now is what the material layer of FEBio-Warp should be.

## For reviewers

**In brief.** Materials follow FEBio's structure (typed parameters with ranges, separate material axes and fibers, solid mixtures, the uncoupled split, history), expressed so that Warp compiles each material tree into one kernel. Four coupled solid models are implemented and checked against FEBio's source and against every parameter set the FEBio test suite uses for them. Added on November 25 for the valve and TPVR inputs: the chordae spring, the linear truss law, and the deviatoric part of uncoupled HGO. Gradients are tested for every model. Everything so far was run on CPU, which is adequate for design and correctness checks at this stage; GPU access for this workflow is being set up for the team, and the timings will be repeated there. The solid models and the spring have since been compared with FEBio runs, through the small problems of the verification draft (pull request 4): stress and tangent, and the spring force, agree to the 12 digits FEBio prints.

**Where to look.**

| To review | Read |
|---|---|
| The design and what was decided | this document |
| What a model looks like next to its FEBio source | `src/warp_febio/materials/models/hgo_unconstrained.py` |
| The interface, parameter ranges, failure rule | `materials/model.py`, `materials/status.py` |
| Mixtures | `materials/compose.py` |
| An uncoupled model (deviatoric part only) | `materials/models/hgo_uncoupled.py` |
| Chordae spring and truss law | `materials/models/nonlinear_spring.py`, `materials/models/linear_truss.py`, `tests/test_material_springs.py` |
| FEBio's fourth-order tensor helpers for 6x6 storage | `materials/voigt.py` |
| The checks every model passes | `materials/conformance.py`, `tests/test_material_interface.py` |
| FEBio closed forms used for comparison (test-only) | `tests/febio_closed_forms.py` |
| Use by an element, and inside `warp.fem` | `tests/test_material_element_consumer.py`, `tests/test_material_warp_fem.py` |

**Most useful feedback.** The questions at the end are grouped by reviewer. Each was checked against the FEBio source, the FEBio test suite, the Warp source and documentation, or the project papers before being asked, and what we found is stated with it. The two that most affect what is built next are 1 (composition at kernel build time) and 2 (geometry fields for fibers).

## Recommendation

Adopt FEBio's way of structuring materials, and keep the list of implemented models short.

- **Structure: follow FEBio.** Parameters of several types with range checks, material axes and fibers as separate properties, nested materials (solid mixture), the uncoupled split, and history. These are what make FEBio's library extensible, and Steve's review shows where a flat design would run out.
- **Mechanism: build the structure when the kernel is compiled.** FEBio composes materials from objects at run time. Warp kernels cannot dispatch at run time, but the same composition can be done once, in Python, before compilation. The result is a single kernel with no branching on material type.
- **Differentiability is tested, not assumed.** Every model is checked for reverse-mode gradients of energy and of stress, with respect to parameters, deformation gradient and geometry. Two errors of ours were found this way and fixed.
- **Catalog: only what the first projects need.** Valve closure and repair, and nitinol device deployment. Fluids, biphasic and multiphasic materials, reactive and growth models stay in FEBio.
- **Use what Warp already has.** The models are plain Warp functions and run unchanged inside `warp.fem`, which already provides meshes, spaces, quadrature and sparse assembly.
- **Evidence: the FEBio test suite.** Every model is tied to FEBio source and to the parameter sets the suite uses. The verification harness is proposed separately because it applies to elements, contact and solvers as well.

## Response to the review

| Steve's point | What this draft does |
|---|---|
| 1. Models exist in several representations | Reference formulas moved to `tests/febio_closed_forms.py`, test-only. The pilot files (`lee_sacks.py`, `neo_hookean.py`, `stvk.py`) have since moved to `pilots/` on `main`; here they serve only as an independent comparison in `tests/test_material_interface.py`. |
| 2. Parameters are floats only | Three kinds of parameter, by what they do. Floats stay in the parameter array, with FEBio's range checks applied when they are packed. Integers, flags and enumerations (for example `pressure_model`) select code, so they are options fixed when the kernel is built. Vectors and frames are geometry (row 5). |
| 3. Density | A property of the `Material` object, read from the input file and available to domains. It never enters a constitutive kernel. |
| 4. Tangent action versus full tensor | Measured, and Steve's concern holds for a representative material. For HGO, stress plus six tangent actions took 191 ms per million points on CPU; one pass that shares intermediates took 89 ms, with identical results. A model can now supply `stress_and_tangent`, returning the Cauchy stress and the 6x6 together, written with helpers that mirror FEBio's `dyad1s` and `dyad4s`. The action stays for matrix-free use and as the fallback, and the two are compared in the tests. The same holds one level up: inside `warp.fem`, a tangent form that uses the stored one-pass result assembled in about 210 to 260 ms, against about 650 ms for a form that calls the action (see Relation to warp.fem). A GPU measurement is still needed. |
| 5. Nested materials and fibers | `solid_mixture` composes models and nests. Fibers are named fields of a per-point geometry vector, separate from the material axes; a model declares the fields it needs, and a material can have several. Continuous fiber distributions are not designed yet. |

## Architecture

### Inputs

A constitutive model is a set of pure functions of four inputs at one material point.

| Input | Holds | FEBio counterpart |
|---|---|---|
| `F` | Deformation gradient | `FEElasticMaterialPoint::m_F` |
| `g` | Geometry vector: material axes, fiber directions and other reference-configuration data, in named fields declared by the model | `mat_axis`, `fiber` and other valuators, evaluated once at setup |
| `p` | Float parameters, FEBio tag names, with ranges | `FEParamDouble` with `FE_RANGE` validators |
| `state` | History variables, committed and trial | Material point data with `Init` and `Update` |

`p` is an array indexed by region, so a value can differ by region or element (FEBio's mapped parameters), can change in time (FEBio's load curves, as one array update per step), and can carry gradients. Material axes from two vectors are built as `FEMat3dVectorMap` does.

### What a solid model returns

The quantities a FEBio material returns, in the current configuration.

```
energy(F, g, p)                     -> scalar       energy per reference volume
cauchy(F, g, p)                     -> 3x3          Cauchy stress
spatial_tangent_action(F, d, g, p)  -> 3x3          c : d for symmetric d
stress_and_tangent(F, g, p)         -> (3x3, 6x6)   optional, one pass, FEBio tens4ds order
```

A model can be written in this form, close to the FEBio source, or from its energy in the reference frame. First Piola stress stays available on every model for gradient checks and for MFEM.

### Composition

- **Solid mixture.** `solid_mixture(name, [(label, model), ...])` returns a model whose energy, stress and tangent are the sums of its children, as `FEElasticMixture` does. Each child reads its own slice of `p` and `g`. Names and parameter ranges are prefixed by label (`ground.E`, `fiber1.fiber`), matching nested XML. Mixtures nest, and keep the one-pass tangent when every child has one.
- **Uncoupled.** An uncoupled model supplies its deviatoric functions only. `volumetric.py` provides `U`, `U'` and `U''` for FEBio's four pressure models. A multi-field element combines them. Following Jeff, they are not summed in a displacement element, and coupled models are the default.
- **Later, same mechanism:** mixture weights, uncoupled mixtures, damage and viscoelastic wrappers.

### Failure inside a kernel

FEBio throws on a non-positive Jacobian and the solver cuts the step. A kernel cannot throw. The rule here (`status.py`): every kernel that evaluates a material writes one status code per point. It checks J before calling the material, so materials may assume J > 0; a non-finite result is reported and zeroed instead of written; the host reads the worst status after the launch and treats any failure as FEBio treats its exception. A failed point does not affect other points in the same launch. Treating a non-finite result like a negative Jacobian matches FEBio, which retries the step when the residual or solution norm is not a number (`NANInResidualDetected` and `NANInSolutionDetected`, caught in `FECore/FEAnalysis.cpp`).

### Model kinds

- **Solid:** a function of F. Used by solid and shell elements.
- **Uniaxial:** axial strain to axial stress, with history: `update(strain, p, committed) -> (stress, tangent, trial)`. Used by the fibers of a beam cross-section, trusses and chordae. The trial array is swapped in on convergence and discarded on a rejected step. The chordae spring and the truss law are uniaxial models without history: for the spring the "strain" is FEBio's deformation measure and the "stress" is the force; for the truss they are the logarithmic stretch and the Kirchhoff stress.

## Library for the first projects

| Model | FEBio type | Used for | Status |
|---|---|---|---|
| Neo-Hookean | `neo-Hookean` | Device (level 1), compression tube | Implemented |
| Isotropic elastic | `isotropic elastic` | Device (level 1, alternative), patch | Implemented |
| Isotropic Lee-Sacks | `isotropic Lee-Sacks` | Valve leaflets | Implemented |
| HGO, unconstrained | `HGO unconstrained` | Outflow tract wall, coupled form | Implemented |
| Nitinol, effective modulus | `isotropic elastic` with a step load curve on `E` | Published TPVR device model | Implemented |
| Nitinol, superelastic, uniaxial | none | Beam-element devices | Shell: interface and tests in place, law to be refined later |
| HGO, uncoupled | `Holzapfel-Gasser-Ogden` | Outflow tract as published | Deviatoric part implemented; needs a three-field element to run |
| Chordae spring | `nonlinear spring` | Valve and patch chordae, tension only | Implemented |
| Linear truss | `linear truss` | Chordae as trusses, medium term | Law implemented; truss element not started |

Deferred until a project needs them: Ogden, anisotropic Lee-Sacks, fiber components and continuous fiber distributions, a three-dimensional superelastic law.

## Nitinol at two levels

**Level 1, effective modulus.** The model in Zelonis et al. (arXiv 2507.06337) and the sensitivity study (arXiv 2601.05410): an elastic solid with a step load curve on Young's modulus between compression and release. With parameters in an array this needs no constitutive code. `effective_modulus_nitinol` pairs an existing elastic model with a schedule. It has no hysteresis and no plateau.

The paper's text gives the law as isotropic elastic and its Table 1 as neo-Hookean, and does not print the load curve. The six open input files in the FEBio model repository (files 3598 to 3603) settle both, and agree with each other:

- The frame is `isotropic elastic` with E = 4.0e7, v = 0.33 and density 6.45e-6. The two crimping tubes are `neo-Hookean` shells with the same constants and no load curve.
- The load curve on E is a STEP curve through (0.998063, 1) and (1, 0.75) with constant extension. By FEBio's step rule (below) E is 4.0e7 through t = 0.998063 and 3.0e7 after it.
- The vessel is `Holzapfel-Gasser-Ogden` (uncoupled) with the published constants, k = 1500, `pressure_model` set to `default`, and material axes given per element as mesh data.

`published_tpvr_frame` returns the frame material exactly as in those files.

**Level 2, superelastic, for beams: a shell for now.** What is in place is the interface a superelastic law plugs into: a uniaxial model with committed and trial history, a consistent tangent, and gradients through the history. It carries a simple law (one internal variable, linear kinetic rules of the kind in Auricchio and Taylor, 1997) so the interface can be exercised. Refining the law is separate, dedicated work and is not attempted here. It should not be used for device results. FEBio has no shape-memory material, and adding one there is deferred, so verification will rest on the primary papers and published uniaxial data.

## What has been checked

52 tests run on CPU: 51 pass, and one GPU test skips without a CUDA device (the count was 59 before the pilot tests moved to `pilots/`). No GPU run has been made yet; GPU access for this workflow is being set up, and CPU is adequate for the design and correctness questions here. The comparisons with FEBio in this pull request are against closed forms transcribed from its source. Comparisons with FEBio runs are in the verification draft (pull request 4): one hexahedron under FEBio's `prescribed deformation` condition, with FEBio's logged stress and tangent stored as the reference. Neo-Hookean, isotropic elastic, isotropic Lee-Sacks, HGO unconstrained and HGO uncoupled (both HGO forms with material axes in and away from the global frame) agree with those runs to about 1e-12 in both, on twelve cases.

**Models against the FEBio source, and their derivatives** (relative error over 66 deformation states, FP64):

| Check | HGO unconstrained | Neo-Hookean | Isotropic elastic | Isotropic Lee-Sacks |
|---|---|---|---|---|
| Cauchy stress vs FEBio closed form | 9.8e-16 | 8.8e-16 | 5.7e-16 | 1.0e-15 |
| 6x6 spatial tangent vs FEBio closed form | 6.4e-16 | 7.4e-16 | 3.2e-16 | 6.9e-16 |
| One-pass tangent vs six tangent actions | 8.0e-17 | 0.0e+00 | n/a | 4.9e-17 |
| Stress vs finite difference of energy | 3.1e-10 | 3.5e-10 | 2.7e-10 | 2.3e-10 |
| Tangent vs finite difference of stress | 1.8e-10 | 3.2e-10 | 2.9e-10 | 2.0e-10 |
| Objectivity | 2.1e-15 | 9.7e-16 | 4.8e-16 | 9.7e-16 |
| Tape: energy gradient w.r.t. parameters | 5.4e-10 | 3.9e-11 | 3.6e-10 | 2.0e-10 |
| Tape: stress gradient w.r.t. parameters | 9.6e-11 | 1.9e-11 | 1.1e-10 | 6.8e-09 |
| Tape: stress gradient w.r.t. F | 3.9e-10 | 1.2e-10 | 1.1e-10 | 1.1e-10 |
| Tape: energy gradient w.r.t. material axes | 1.7e-09 | n/a | n/a | n/a |
| FP32 vs FP64 stress | 4.8e-07 | 3.5e-07 | 2.9e-07 | 2.1e-07 |
| Stress vs pilot implementation | n/a | 1.2e-16 | 0.0e+00 | 4.3e-16 |

**Against the FEBio test suite's own parameter values.** Every distinct parameter set the suite uses for these materials (77 sets: 66 neo-Hookean, 10 isotropic elastic, 1 HGO unconstrained, from 445 uses at suite commit `986005f`) passes the range checks and agrees with the FEBio closed forms for stress and tangent. The suite has no Lee-Sacks problem. The values are stored in `tests/data/febio_suite_materials.json`; the input files are not copied.

**Architecture:**

- A mixture of neo-Hookean and two independent fibers equals the sum of its parts to 1e-13 and passes the same checks as a single model. Mixtures nest and compile in FP32.
- All four volumetric functions have consistent derivatives, zero energy and pressure at J = 1, and stiffness k there.
- Parameter ranges follow FEBio's validators and carry into mixtures under prefixed names.
- Failed points (J <= 0, overflow) are flagged and zeroed, and do not disturb good points in the same launch.
- Two models that share a name do not share a compiled kernel.
- **From an element's point of view.** A single eight-node hexahedron in an updated Lagrangian form, written in NumPy for the test, takes Cauchy stress and the 6x6 tangent from the material layer. Its stiffness (material plus initial-stress terms) equals the finite-difference derivative of its internal force to 1e-7 and is symmetric, for all four models and for a mixture of neo-Hookean and HGO. A rigid motion produces no force. This checks the 6x6 index order and shear convention where an element would use them.
- HGO: fibers carry no compression; the two families are symmetric in the angle; the model is isotropic at kappa = 1/3; its energy and the derivative with respect to k2 are finite at k2 = 0, where FEBio's energy expression divides by zero.

**Chordae spring, truss law and uncoupled HGO** (added November 25):

- **Spring.** `nonlinear spring` with a linear point function, the three deformation measures, and constant or extrapolated extension. The number of points, the measure and the extension are fixed when the kernel is built; the scale and the point coordinates are float parameters. The eight chord groups of the mitral input (100 springs) therefore share one kernel, and the slack length is differentiable. Force and 3x3 stiffness match a transcription of `FENonlinearSpringMaterial` to 1e-13; the stiffness equals the finite-difference derivative of the force; tape gradients with respect to the scale and the points match finite differences.
- **Spring against FEBio runs.** The chord case of the verification draft (elongation and force logged by FEBio through slack and tension) agrees to round-off. Five further one-off runs, covering the three measures, both extensions and a five-point curve, also agreed to round-off; those are not stored.
- **A difference from FEBio in the spring tangent.** FEBio differentiates the point curve numerically (`PointCurve::derive`), which is accurate to about 1e-7 away from a knot and returns the mean slope at one. Here the slope of the active segment is exact.
- **Linear truss.** The law (Kirchhoff stress E ln(lambda)) and the force and stiffness of `FELinearTrussDomain` for one element. The stiffness equals the finite-difference derivative of the force. FEBio's converged displacements for three one-element problems satisfy this force law to 1e-12 (one-off runs, not stored). There is no truss element here yet.
- **Uncoupled HGO.** The deviatoric energy, stress and tangent of `FEHolzapfelGasserOgden`. It passes the same consistency and gradient checks as the coupled models. Against FEBio runs, with the volumetric terms added as `FEUncoupledMaterial` does: three stored cases with the suite's `ma34` constants, and a one-off run with the TPVR vessel constants, `pressure_model` set to `default` and rotated axes (stress and tangent 2e-12, energy 8e-13). In that run, using the global axes would change the stress by 7 percent and using the material's class-default pressure model by 9 percent, so both must come from the input.
- **A property of uncoupled HGO.** Under a pure volume change the deviatoric deformation is a rotation, so both fiber strains are exactly zero, which is where the fibers switch on. The stress is continuous there and the tangent is not. FEBio takes the active branch at zero strain and so does this code; the finite-difference check leaves those states out.

**Defects found by these tests and fixed:**

- Reverse-mode gradients through a history were wrong when the new state was made by copying the committed vector and overwriting one component. Forward values were correct. This is our error, not Warp's: Warp's differentiability guide says a vector component may be assigned only once, and the adjoint of the overwritten value is not cleared. No warning is raised, so it is easy to write. Building the state with a constructor fixed it, a test covers it, and `examples/material_interface/warp_component_write_gradient.py` shows it in a few lines.
- The gradient check itself failed for the suite's smallest moduli (1e-12) because its step was absolute. It is now relative.
- The step load curve was evaluated the wrong way round. FEBio's STEP rule returns the value of the first point whose time is greater than t, so between two points the later value already applies; this code held the earlier value until the next point. A FEBio 4.13 run confirmed FEBio's rule (a STEP curve through (0.25, 1), (0.5, 0.6), (0.75, 0.3) gave 1 at t = 0.25, 0.6 at t = 0.375 and 0.3 at t = 0.5). For the published TPVR frame the difference is whether the modulus drops just after t = 0.998 or at t = 1. Corrected, with those values in the tests.

**Things the tests corrected in this document:** an earlier statement that HGO is nearly isotropic at the published kappa = 0.33 was too strong. Changing the material axes changes the stress by up to 4.5 percent there, against about 90 percent at kappa = 0.1.

**Kernel build cost of composition** (CPU, first call, includes compilation): 1.6 s for a mixture of two components, 3.1 s for eight. Later calls reuse the kernel. Across processes, loading the ten kernels for the four models and one mixture took 12.8 s the first time on two CPU cores and 0.03 s afterwards, from Warp's kernel cache.

**Single precision.** FP32 stress error against FP64 grows as strain shrinks: 2e-4 at a strain of 1e-3, 1e-2 at 1e-5. Taking the displacement gradient as input and forming b - I and J - 1 without cancellation removes this when rotations are small (error 1e-7 at every strain tested). With a 30 degree rotation it helps by only a factor of three, because the small strain is still the difference of order-one numbers. Consequence: FP32 is adequate for soft tissue at finite strain, and is not adequate for stiff parts at small strain with large rotation, such as a nitinol frame meshed with solids. Those need FP64, or beam elements whose strain is formed in a co-rotated frame.

Apart from the single test element above, these are material-point checks. No solve is involved.

## Relation to warp.fem

Warp already ships a finite element module, `warp.fem`. Read from the installed version (1.19.0), it provides hexahedral, tetrahedral and surface meshes; Lagrange, serendipity and B-spline spaces of several degrees; quadrature; block sparse assembly with reuse of the sparsity map; projection of boundary conditions; deformed geometries; and single or double precision. Its examples include mixed nonlinear elasticity, contact between separately meshed bodies, and shape and topology optimization through the tape. It does not appear to include shell elements with thickness, beam elements, or potential-based contact.

Two findings bear on this proposal.

- **The materials work inside `warp.fem` as they are.** A FEBio-Warp model is a set of plain Warp functions, so an integrand can call it with no adapter. `tests/test_material_warp_fem.py` assembles the residual and tangent of a hexahedral grid in double precision with neo-Hookean and with HGO; the tangent matches finite differences of the residual to 1e-8 and is symmetric. `warp.fem` works with reference-configuration gradients, so these forms use first Piola stress and the tangent action, which every model carries.
- **How the tangent is formed matters more than which material is used.** Three ways to form the tangent of the same grid were compared, all in double precision and all agreeing to 1e-16. CPU, two cores, 4,096 hexahedra (32,768 integration points); timings vary by about 20 percent between runs.

| Route | HGO | Neo-Hookean |
|---|---|---|
| A. `warp.fem` form that calls the tangent action | about 650 ms | about 450 ms |
| B. `warp.fem` form on stress and 6x6 stored once per point | 210 to 260 ms | 210 to 230 ms |
| C. Dedicated hexahedron kernel, then sparse build | 55 to 70 ms | 50 to 60 ms |

Route A evaluates the material again for every trial function, which is why the one-pass form helps. Evaluating the material is 6 to 10 ms of route B. Reusing the sparsity map (`construction="mapped"`) did not change A or B on CPU. None of this has been run on a GPU.

A correction: an earlier version of this section compared assembly with the material evaluation alone and reported a gap of about one hundred. That compared unlike things. A first, plain version of the dedicated kernel took about 490 ms, no faster than `warp.fem`; the figure above is after one rewrite of its inner loop.

What follows for the design:

- Keep both forms on every model. The tangent action is what a matrix-free operator consumes and the simplest thing to write in a `warp.fem` form. The one-pass stress and 6x6 tangent is what an assembled element consumes, in `warp.fem` or in a dedicated kernel, and it also serves partial assembly, where the tensors are stored at the points and applied without forming a matrix.
- `warp.fem` is a good basis for the reference tier and for checking elements: it is already written, tested by the Warp team, and differentiable.
- Whether the fast tier needs dedicated element kernels, and whether to assemble the matrix or apply it matrix-free, are element and solver questions. They are taken up in a separate draft on assembly, with the comparison script, so that this proposal stays about materials.

## Coupling to MFEM

Unchanged in substance. The exchange kernel returns Cauchy stress and the 6x6 spatial tangent in FEBio `tens4ds` order into caller-owned memory, which is what the FEBio-MFEM immersed FSI plugin's domains consume, and now uses the one-pass function when a model has one. The volumetric functions match that code's `QUAD` (FEBio `Abaqus`) and `ST91` (FEBio `NIKE3D` and `Abaqus (GOH)`) energies. The MFEM-side shim is not written.

## Decisions so far

- Updated Lagrangian with Cauchy stress as the consumed form (Jeff).
- Coupled models by default; uncoupled only for near-incompressibility and only with a multi-field element (Jeff).
- Verification against the FEBio test suite (Jeff, Steve).
- The pilot implementations are not the basis for the design (MJ, Steve).
- Superelastic nitinol for beams, kept as a shell for now; no shape-memory material in FEBio for now (MJ).
- A non-finite result is handled like a negative Jacobian. FEBio retries the step in both cases, so this needed no question.
- Chordae follow the published valve models (Wu et al. 2022; arXiv 2509.22865): tension-only two-node springs with zero force until the elongation passes a threshold and a linear response beyond it, modeled with FEBio's `nonlinear spring`. The repository's mitral input has eight such groups, with thresholds of 2 to 12 mm. This is now implemented. Trusses for chordae are of interest in the medium term; the linear truss law is in place and the elastic-truss route waits on question 7.

## Questions for reviewers

Each question was checked before being asked: against the FEBio source (develop, commit `552138d`), the FEBio test suite (commit `986005f`), the Warp 1.19.0 source and documentation, the immersed FSI plugin source, or the project papers. What we found is stated with the question. Four questions from the previous version were answered by those sources and have been removed; they are recorded under Decisions or as findings above.

**Steve**

1. Composition when the kernel is built. Tested here: mixtures nest, compile in both precisions, and cost 1.6 to 3.1 s to build once. Is this an acceptable counterpart to FEBio's run-time composition, given that each distinct material tree compiles its own kernel?
2. Geometry for fibers. FEBio's fiber materials hold their own fiber vector (`FEElasticFiberMaterial::m_fiber`), separate from `mat_axis`. Here that is one named three-component field per fiber in a flat per-point vector, tested with two independent fibers in a mixture. Is that enough for what fiber materials need, and what would continuous fiber distributions require beyond it?
3. Build-time options. FEBio's load controllers can drive integer and boolean parameters as well as real ones (`FEModel.cpp`). In the test suite, the material parameters under a load curve are all real-valued (seven names, 29 uses). Is it safe to fix integer, boolean and enumeration parameters when the kernel is built, or is there a case where one changes during a run?
4. Lee-Sacks ranges. `FEIsotropicLeeSacks` and its uncoupled form declare no parameter ranges. We added non-negativity for c0, c1, c2 and k. Is that right? The suite has no Lee-Sacks problem; would one be welcome there?
5. HGO at k2 = 0. In `FEHolzapfelUnconstrained.cpp` the range for k2 allows zero, and the strain energy divides by k2 (stress and tangent do not). Is that worth changing in FEBio? Here the energy uses a form that is finite at k2 = 0.
6. A FEBio reference at a material point. The verification draft (pull request 4) now uses one element under `FEBCPrescribedDeformation` with FEBio's element log variables for the deformation gradient, stress and tangent, which average over the element's integration points; with a uniform deformation those are point values, and the models here agree with them (see What has been checked). FEBio also has a material test diagnostic (`FEBioTest/FEMaterialTest`, uni-axial and simple shear scenarios), which we have not used. Is the prescribed-deformation route the one you would recommend, and is the diagnostic maintained?
7. Trusses for chordae. FEBio has two truss domains. `linear-truss` uses the `linear truss` material (Kirchhoff stress E ln(lambda)) with the scalar stiffness V/l^2 (E - 2 tau). `elastic-truss` evaluates any solid material at F = diag(lambda, lambda^-v, lambda^-v), which is what a hyperelastic chord would need; in `FEElasticTrussDomain::ElementStiffness` the scalar stiffness is `k = E`, with the linear-truss expression commented out. In a one-element test here (neo-Hookean, E = 1000, v = 0.3, about 5 percent stretch, FEBio 4.13 built from source without MKL), with full Newton (`max_ups` 0) the elastic-truss domain took 172 iterations at a cross-sectional area of 1 and did not converge at an area of 0.001, where the linear-truss domain took 30 in both. With the default BFGS settings and tolerances both domains took about 30, and the elastic-truss displacements at the two areas differed in the fourth digit, where they should be equal. The test suite uses `linear-truss` in three problems and `elastic-truss` in none. Is the elastic-truss domain still in progress, and is it what you would recommend for chordae with a hyperelastic material? If so, what should its tangent be?

**Jeff**

8. The published outflow tract model (Zelonis et al., Table 1, and the six open input files) uses uncoupled HGO with c = 200 kPa and k = 1500 kPa, a bulk to shear ratio of 7.5, well under the ratio of about 100 you gave as the threshold. Should that case be run as coupled going forward? Both forms are now implemented at the material level and agree with FEBio runs; the uncoupled one needs a three-field element before it can run.

**Ryan**

9. The plugin's immersed elastic domain calls `Stress` and `SolidTangent`, takes the tangent as a `tens4dmm`, and has switches for secant stress and secant tangent. The exchange kernel here returns Cauchy stress and a full 6x6 in the same pair order (11, 22, 33, 12, 23, 13), stored by rows where `tens4dmm` stores by columns. Is the secant form needed from the Warp side, and is there anything else the domain reads from the material point?

**Warp team (Eric, Kay)**

10. Kernels per material tree. The code generation guide describes building kernels from captured functions, which is what is done here, with one `module="unique"` kernel per material tree. Measured cost is under Kernel build cost above. For a library of this shape, would you keep one module per tree, or group a model's kernels in a named module, and is ahead-of-time compilation the intended route for deployment?
11. Precision. Single precision loses the stress at small strain (table above), so stiff parts need FP64 while tissue does not. In 1.19.0, `wp.Mesh` and `wp.Bvh` accept only `vec3` (FP32) points and bounds, while `wp.HashGrid` accepts FP64. For contact between FP64 bodies, is casting positions to FP32 for the queries the intended pattern, and is mixed precision by domain reasonable to plan for?
12. `warp.fem` for nonlinear solids. With the material evaluated once per point and stored, `fem.integrate` assembled the tangent in 210 to 260 ms on CPU against 55 to 70 ms for a dedicated hexahedron kernel (table above); no GPU numbers yet. Is storing point data and indexing it by `s.qp_index` the intended way to avoid re-evaluating a material per trial function, and is there a faster route within `warp.fem` that we missed?
13. Shells in `warp.fem`. The valve models use four-node quadrilateral shells with thickness (0.396 mm), and FEBio's elastic shell domain carries a second displacement per node for the back face (`sx`, `sy`, `sz`) with integration through the thickness. `warp.fem` 1.19.0 has surface meshes embedded in 3D (`Quadmesh3D`, `Trimesh3D`) but no shell abstraction that we could find: no thickness, director or through-thickness rule. In a short check outside this PR, a `Quadmesh3D` with two `vec3d` fields, the surface gradient, the surface normal and a hand-written two-point rule through the thickness compiled and integrated in FP64. Is that the route you would take for shells, with the shell kinematics written in the integrand, or is a shell element type planned or better placed in `warp.fem` itself? Is there shell work elsewhere (Newton, for example) that we should build on instead?

## Next steps

1. Verification harness (separate draft).
2. Assembly: assembled against matrix-free, `warp.fem` against dedicated kernels (separate draft).
3. A three-field element, so the uncoupled HGO can run (element work).
4. A discrete spring domain and, after question 7, a truss element (element work).
5. A GPU run of the tests and of the timings, once GPU access is in place.
6. Later, as dedicated work: the superelastic law.
