# FEBio-Warp material models: architecture proposal

Draft v0.3, November 24, 2026. For Steve Maas, Jeff Weiss, Ryan Black, the Warp team and the FEBio-Warp team. Nothing here is merged. This version replaces v0.2 and follows Steve's review of this pull request and Jeff's earlier comments. The earlier pilot code is no longer the starting point; the question now is what the material layer of FEBio-Warp should be.

## For reviewers

**In brief.** Materials follow FEBio's structure (typed parameters with ranges, separate material axes and fibers, solid mixtures, the uncoupled split, history), expressed so that Warp compiles each material tree into one kernel. Four solid models are implemented and checked against FEBio's source and against every parameter set the FEBio test suite uses for them. Gradients are tested for every model. Nothing has been run on a GPU or against a FEBio run yet.

**Where to look.**

| To review | Read |
|---|---|
| The design and what was decided | this document |
| What a model looks like next to its FEBio source | `src/warp_febio/materials/models/hgo_unconstrained.py` |
| The interface, parameter ranges, failure rule | `materials/model.py`, `materials/status.py` |
| Mixtures | `materials/compose.py` |
| FEBio's fourth-order tensor helpers for 6x6 storage | `materials/voigt.py` |
| The checks every model passes | `materials/conformance.py`, `tests/test_material_interface.py` |
| FEBio closed forms used for comparison (test-only) | `tests/febio_closed_forms.py` |
| Use by an element, and inside `warp.fem` | `tests/test_material_element_consumer.py`, `tests/test_material_warp_fem.py` |

**Most useful feedback.** The questions at the end are grouped by reviewer. The three that most affect what is built next are 1 (composition at kernel build time), 2 (geometry fields for fibers) and 14 (`warp.fem` or dedicated element kernels).

## Recommendation

Adopt FEBio's way of structuring materials, and keep the list of implemented models short.

- **Structure: follow FEBio.** Parameters of several types with range checks, material axes and fibers as separate properties, nested materials (solid mixture), the uncoupled split, and history. These are what make FEBio's library extensible, and Steve's review shows where a flat design would run out.
- **Mechanism: build the structure when the kernel is compiled.** FEBio composes materials from objects at run time. Warp kernels cannot dispatch at run time, but the same composition can be done once, in Python, before compilation. The result is a single kernel with no branching on material type.
- **Differentiability is tested, not assumed.** Every model is checked for reverse-mode gradients of energy and of stress, with respect to parameters, deformation gradient and geometry. Two defects were found this way and fixed.
- **Catalog: only what the first projects need.** Valve closure and repair, and nitinol device deployment. Fluids, biphasic and multiphasic materials, reactive and growth models stay in FEBio.
- **Use what Warp already has.** The models are plain Warp functions and run unchanged inside `warp.fem`, which already provides meshes, spaces, quadrature and sparse assembly.
- **Evidence: the FEBio test suite.** Every model is tied to FEBio source and to the parameter sets the suite uses. The verification harness is proposed separately because it applies to elements, contact and solvers as well.

## Response to the review

| Steve's point | What this draft does |
|---|---|
| 1. Models exist in several representations | Reference formulas moved to `tests/febio_closed_forms.py`, test-only. The pilot files (`lee_sacks.py`, `neo_hookean.py`, `stvk.py`) are still used as an independent comparison and should be removed once their callers are migrated. |
| 2. Parameters are floats only | Three kinds of parameter, by what they do. Floats stay in the parameter array, with FEBio's range checks applied when they are packed. Integers, flags and enumerations (for example `pressure_model`) select code, so they are options fixed when the kernel is built. Vectors and frames are geometry (row 5). |
| 3. Density | A property of the `Material` object, read from the input file and available to domains. It never enters a constitutive kernel. |
| 4. Tangent action versus full tensor | Measured, and Steve's concern holds for a representative material. For HGO, stress plus six tangent actions took 191 ms per million points on CPU; one pass that shares intermediates took 89 ms, with identical results. A model can now supply `stress_and_tangent`, returning the Cauchy stress and the 6x6 together, written with helpers that mirror FEBio's `dyad1s` and `dyad4s`. The action stays for matrix-free use and as the fallback, and the two are compared in the tests. A GPU measurement is still needed. |
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

FEBio throws on a non-positive Jacobian and the solver cuts the step. A kernel cannot throw. The rule here (`status.py`): every kernel that evaluates a material writes one status code per point. It checks J before calling the material, so materials may assume J > 0; a non-finite result is reported and zeroed instead of written; the host reads the worst status after the launch and treats any failure as FEBio treats its exception. A failed point does not affect other points in the same launch.

### Model kinds

- **Solid:** a function of F. Used by solid and shell elements.
- **Uniaxial:** axial strain to axial stress, with history: `update(strain, p, committed) -> (stress, tangent, trial)`. Used by the fibers of a beam cross-section, trusses and chordae. The trial array is swapped in on convergence and discarded on a rejected step.

## Library for the first projects

| Model | FEBio type | Used for | Status |
|---|---|---|---|
| Neo-Hookean | `neo-Hookean` | Device (level 1), compression tube | Implemented |
| Isotropic elastic | `isotropic elastic` | Device (level 1, alternative), patch | Implemented |
| Isotropic Lee-Sacks | `isotropic Lee-Sacks` | Valve leaflets | Implemented |
| HGO, unconstrained | `HGO unconstrained` | Outflow tract wall, coupled form | Implemented |
| Nitinol, effective modulus | elastic law with a load curve on `E` | Published TPVR device model | Implemented |
| Nitinol, superelastic, uniaxial | none | Beam-element devices | Shell: interface and tests in place, law to be refined later |
| HGO, uncoupled, with three-field element | `Holzapfel-Gasser-Ogden` | Outflow tract as published | Next, with the element |
| Chordae | to be confirmed | Valve chordae, under the uniaxial kind | Next |

Deferred until a project needs them: Ogden, anisotropic Lee-Sacks, fiber components and continuous fiber distributions, a three-dimensional superelastic law.

## Nitinol at two levels

**Level 1, effective modulus.** The model in Zelonis et al. (arXiv 2507.06337) and the sensitivity study (arXiv 2601.05410): an elastic solid with a step load curve on Young's modulus between compression and release. With parameters in an array this needs no constitutive code. `effective_modulus_nitinol` pairs an existing elastic model with a schedule. It has no hysteresis and no plateau. One point to settle from the published input files: the paper's text gives the law as isotropic elastic, its Table 1 as neo-Hookean. Both are available until that is checked. The load curve values are not printed in the paper and must come from the input files.

**Level 2, superelastic, for beams: a shell for now.** What is in place is the interface a superelastic law plugs into: a uniaxial model with committed and trial history, a consistent tangent, and gradients through the history. It carries a simple law (one internal variable, linear kinetic rules of the kind in Auricchio and Taylor, 1997) so the interface can be exercised. Refining the law is separate, dedicated work and is not attempted here. It should not be used for device results. FEBio has no shape-memory material, and adding one there is deferred, so verification will rest on the primary papers and published uniaxial data.

## What has been checked

43 tests pass on CPU. Two GPU tests skip without a CUDA device. No GPU run has been made, and FEBio itself has not been run: comparisons with FEBio are against closed forms transcribed from its source.

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

**Defects found by these tests and fixed:**

- Reverse-mode gradients through a history were wrong when the new state was made by copying the committed vector and writing its components (Warp 1.19). Forward values were correct. Building the state with a constructor fixed it.
- The gradient check itself failed for the suite's smallest moduli (1e-12) because its step was absolute. It is now relative.

**Things the tests corrected in this document:** an earlier statement that HGO is nearly isotropic at the published kappa = 0.33 was too strong. Changing the material axes changes the stress by up to 4.5 percent there, against about 90 percent at kappa = 0.1.

**Kernel build cost of composition** (CPU, first call, includes compilation): 1.6 s for a mixture of two components, 3.1 s for eight. Later calls reuse the kernel.

**Single precision.** FP32 stress error against FP64 grows as strain shrinks: 2e-4 at a strain of 1e-3, 1e-2 at 1e-5. Taking the displacement gradient as input and forming b - I and J - 1 without cancellation removes this when rotations are small (error 1e-7 at every strain tested). With a 30 degree rotation it helps by only a factor of three, because the small strain is still the difference of order-one numbers. Consequence: FP32 is adequate for soft tissue at finite strain, and is not adequate for stiff parts at small strain with large rotation, such as a nitinol frame meshed with solids. Those need FP64, or beam elements whose strain is formed in a co-rotated frame.

Apart from the single test element above, these are material-point checks. No solve is involved.

## Relation to warp.fem

Warp already ships a finite element module, `warp.fem`. Read from the installed version (1.19.0), it provides hexahedral, tetrahedral and surface meshes; Lagrange, serendipity and B-spline spaces of several degrees; quadrature; block sparse assembly with reuse of the sparsity map; projection of boundary conditions; deformed geometries; and single or double precision. Its examples include mixed nonlinear elasticity, contact between separately meshed bodies, and shape and topology optimization through the tape. It does not appear to include shell elements with thickness, beam elements, or potential-based contact.

Two findings bear on this proposal.

- **The materials work inside `warp.fem` as they are.** A FEBio-Warp model is a set of plain Warp functions, so an integrand can call it with no adapter. `tests/test_material_warp_fem.py` assembles the residual and tangent of a hexahedral grid in double precision with neo-Hookean and with HGO; the tangent matches finite differences of the residual to 1e-8 and is symmetric. `warp.fem` works with reference-configuration gradients, so these forms use first Piola stress and the tangent action, which every model carries.
- **General assembly costs far more than the material.** On CPU, for 4,096 hexahedra (32,768 integration points) with HGO, assembling the tangent took 3.6 s with the "generic" strategy and 0.65 s with "dispatch"; the residual took 46 ms. Evaluating stress and the full tangent once per integration point, as a dedicated element kernel would, took 5 ms. With neo-Hookean the assembly times were 3.1 s and 0.5 s, so most of the cost is assembly, not the material. The option that reuses the sparsity map was not tested, and none of this was run on a GPU.

What follows for the design:

- Keep both forms on every model. The tangent action is what a `warp.fem` bilinear form consumes, and what a matrix-free operator consumes. The one-pass stress and 6x6 tangent is what a dedicated element kernel consumes.
- `warp.fem` is a good basis for the reference tier and for checking elements: it is already written, tested by the Warp team, and differentiable. A dedicated element kernel is likely needed for the fast tier. That choice belongs to the element design and is a question for the Warp team below.

## Coupling to MFEM

Unchanged in substance. The exchange kernel returns Cauchy stress and the 6x6 spatial tangent in FEBio `tens4ds` order into caller-owned memory, which is what the FEBio-MFEM immersed FSI plugin's domains consume, and now uses the one-pass function when a model has one. The volumetric functions match that code's `QUAD` (FEBio `Abaqus`) and `ST91` (FEBio `NIKE3D` and `Abaqus (GOH)`) energies. The MFEM-side shim is not written.

## Decisions so far

- Updated Lagrangian with Cauchy stress as the consumed form (Jeff).
- Coupled models by default; uncoupled only for near-incompressibility and only with a multi-field element (Jeff).
- Verification against the FEBio test suite (Jeff, Steve).
- The pilot implementations are not the basis for the design (MJ, Steve).
- Superelastic nitinol for beams, kept as a shell for now; no shape-memory material in FEBio for now (MJ).

## Questions for reviewers

**Steve**

1. Is composing materials when the kernel is built an acceptable counterpart to FEBio's run-time composition, given that each distinct material tree compiles its own kernel?
2. Is a flat per-point geometry vector with named fields enough for what fiber materials need, including several fibers and, later, continuous distributions?
3. Which FEBio integer and enumeration parameters, if any, need to change within one model instance at run time, so that build-time options would not be enough?
4. Lee-Sacks has no parameter ranges in FEBio. We added non-negativity. Is that right, and would a Lee-Sacks problem be useful in the FEBio suite, which has none?
5. HGO: FEBio's energy divides by zero at k2 = 0 while its range allows it. Is that worth changing in FEBio?
6. Can FEBio drive a material through a prescribed deformation history without a mesh, and write F and stress at integration points to the log? Either would let us compare against FEBio runs and not only its source.

**Jeff**

7. The published outflow tract model used uncoupled HGO at a bulk to shear ratio of 7.5. Should that case be treated as coupled going forward, which the unconstrained model here now allows?
8. Chordae: which constitutive law should the uniaxial kind carry for the valve problems?
9. Failure rule: is zeroing a failed point and cutting the step, as FEBio does for a negative Jacobian, also right for a non-finite stress?

**Ryan**

10. For MFEM, is Cauchy stress with the 6x6 tangent per quadrature point what the immersed solid should receive, and is the one-pass form useful for partial assembly, or would you want the action only?

**Warp team (Eric, Kay)**

11. Is the wrong reverse-mode gradient after writing components of a copied vector argument a known limitation? A short reproducer is available.
12. Is building kernels from captured functions, one kernel per material tree, the recommended way to specialize, and are there limits on how many such kernels a session should hold?
13. Precision: given the single-precision results above, is mixed precision by domain (FP64 for stiff parts, FP32 for tissue) reasonable to plan for?
14. `warp.fem`: for solid elements, would you build on `fem.integrate` and tune it (dispatch assembly, reused sparsity map), or write dedicated element kernels that evaluate the material once per integration point? The CPU timings above suggest the second for speed, but we have no GPU numbers.

**Seda**

15. Which pilot callers still depend on `lee_sacks.py`, `neo_hookean.py` and `stvk.py`, so they can be migrated and the files removed?

## Next steps

1. Verification harness (separate draft).
2. HGO, uncoupled, with a three-field element.
3. Chordae under the uniaxial kind.
4. A GPU run of the tests and of the tangent timing.
5. Remove the pilot material files once their callers are migrated.
6. Later, as dedicated work: the superelastic law.
