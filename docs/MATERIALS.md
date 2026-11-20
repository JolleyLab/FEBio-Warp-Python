# FEBio-Warp material interface: proposal for discussion

Draft, November 20, 2026, revised the same day after comments from Jeff Weiss. For Steve Maas, Jeff Weiss, Ryan Black and the FEBio-Warp team. Nothing here is merged. The three existing materials have been converted to this form on the draft branch `materials-interface-proposal`, and the results are in the section "Results for the three existing models".

**What changed in this revision.** The first draft had elements consume first Piola stress in the reference frame. Jeff asked what advantage that gave, and there was none specific to this project: it came from the pilot code and from MFEM's interface. Elements now consume Cauchy stress and the spatial tangent (updated Lagrangian), as FEBio does. Jeff's guidance on uncoupled models and on verification problems is also adopted; see items 1 and 5 and "Verification problems".

## Why now

`FEBio-Warp-Python` has three materials (isotropic Lee-Sacks, neo-Hookean, St. Venant-Kirchhoff) written with three different naming schemes and hard-coded FP64 scalar parameters. Five more are planned: HGO, Ogden, anisotropic Lee-Sacks, an initial nitinol law, and a superelastic nitinol law. We would like one agreed shape before writing them, and we want that shape to stay usable from MFEM.

## What we took from FEBio, MFEM and iFSI

| Source | What it does | What we keep |
|---|---|---|
| FEBio `FEElasticMaterial` | `Stress` (Cauchy), `Tangent` (spatial), `StrainEnergyDensity`, all reading a material point | The three quantities; the spatial form as a supported way to write a model |
| FEBio `FEUncoupledMaterial` | `DevStress`, `DevTangent` plus `U(J)` chosen by `pressure_model`; the element decides how to integrate the two parts | The deviatoric/volumetric split as a declared property of the model, with the four FEBio `U(J)` options in one shared file |
| FEBio `FEMaterialPoint` | Carries F, J, local frame Q, and history with `Init`/`Update` | F and Q as explicit inputs; history as explicit committed and trial arrays |
| FEBio `FEParamDouble` | Parameters can vary by material point | Parameters as an array indexed by region or element |
| MFEM `HyperelasticModel` | `EvalW(F)`, `EvalP(F)` (first Piola), `AssembleH` | The reference-frame pair (energy, first Piola stress) kept on every model for MFEM and for gradient checks |
| iFSI native materials (`MFEMFSI/ImmersedFSI/Materials`) | `MaterialType` with `EvalPK2Stress`, `EvalCauchyStress`, `EvalElasticityTensor(i,j,k,l)` assembled to Voigt 6x6; total or updated Lagrangian; ordered `mat_consts`; a nearly incompressible base class holding the volumetric energy while derived classes supply the isochoric part | Ordered parameter vector; the same dev/vol split; a full spatial tangent available as well as the action |
| FEBio-MFEM-iFSI plugin (`MFEMImSolid`) | The immersed solid is an FEBio domain: it sets F at each FEBio material point and calls `Stress`, `SolidTangent`, and for three-field `DevStress`, `DevTangent`, `UJ`, `UJJ` | Cauchy stress and spatial tangent in FEBio conventions as the exchange format with MFEM |
| MFEM `miniapps/hooke` | Material is a plain struct with `stress` and `action_of_gradient`, the derivative source selectable (symbolic, AD, finite difference) | Tangent as an action on a direction; finite difference kept as the test, not the production path |

## Proposal

**1. One form consumed by elements.** Three pure functions per model, in the current configuration (updated Lagrangian), the same quantities a FEBio material returns:

```
energy(F, Q, p)                     -> scalar   energy per reference volume
cauchy(F, Q, p)                     -> 3x3      Cauchy stress (symmetric)
spatial_tangent_action(F, d, Q, p)  -> 3x3      c : d for symmetric d (c is FEBio's spatial tangent)
```

F is the deformation gradient, Q the local frame (FEBio `mat_axis`), p a fixed-length parameter vector. No element, mesh or solver knowledge enters a material. Reasons for this form: element residual and stiffness can be compared with FEBio term by term; contact and boundary conditions that depend on the current geometry see the same stress measure; stress and tangent are symmetric (6 and 6x6), which is less to store and move on a GPU; and it is what the MFEM plugin consumes. The full 6x6 tangent is the action applied to the six symmetric unit tensors.

**2. Two ways to write a model.** Either write the three functions directly, which lets a FEBio material be transcribed nearly line for line, or write the model in the reference frame from its energy (first Piola stress and its directional derivative) and let one shared adapter produce the consumed form. The adapters are the standard relations P = J sigma F^-T and d(tau) = J c:d + l tau + tau l^T. Both sets of functions are available on every model, so the reference-frame pair remains for gradient checks and for MFEM's `HyperelasticModel` convention.

Checked: neo-Hookean and isotropic Lee-Sacks are written spatially from the FEBio source; isotropic elastic is written in the reference form. All three reproduce FEBio's Cauchy stress and spatial tangent to rounding (table below). This also confirms the FEBio Lee-Sacks tangent is the exact one when `tangent_scale` is 1.

**3. A small descriptor per model.** Name, the FEBio `type` string it reproduces (so a `.feb` reader can map to it), ordered parameter names, coupled or uncoupled, whether it uses Q, and the number of history variables. A registry maps FEBio type strings to models.

**4. Precision and parameters.** Functions are generic in the float type, so one source gives FP32 and FP64. Parameters are passed as an array, which allows per-element values and gradients.

Checked: the same Lee-Sacks source compiles in both precisions (FP32 agrees with FP64 to about 3e-7), and the gradient of total energy with respect to c0, c1, c2, k from the Warp tape matches finite differences to nine digits.

**5. Coupled first; uncoupled only with a multi-field element.** Following Jeff: an uncoupled model exists to simulate near-incompressibility, and it needs an appropriate multi-field element to avoid locking. If near-incompressibility is not the aim (bulk to shear ratio below about 100), use the corresponding coupled (unconstrained) model with a displacement-based element. So new models are written in their coupled form first: Ogden unconstrained and HGO unconstrained. An uncoupled model supplies its deviatoric functions, takes `U(J)` from one shared file with the four FEBio pressure models, and is only run together with a three-field element. We will not sum the two parts in a displacement element, which the first draft offered as a shortcut. One case to decide: the published TPVR RVOT model used uncoupled HGO with k = 1500 kPa and c = 200 kPa, a ratio of 7.5; reproducing those results needs the uncoupled model and three-field element, while new work at those values would use the coupled model.

**6. Path-dependent models (superelastic nitinol).** One extra function, `update(F, Q, p, committed) -> (sigma, trial)`, with history held in two arrays that are swapped on a converged step and discarded on a rejected one. This mirrors FEBio's `Init`/`Update` and the `commit`/`rollback` already in `contracts.py`. Not yet prototyped.

**7. Comments that point back to FEBio.** Each model file opens with the FEBio type string, links to the FEBio source file and the FEBio feature manual page, the strain energy written out, and the parameter order with units. Parameters use the FEBio XML tag names (for example `E` and `v`, converted to Lame parameters inside the model as FEBio does). Functions say which FEBio method they mirror, and any deliberate difference from FEBio is stated where it occurs.

**8. One conformance test for every model.** Zero energy and stress at F = I and under rigid rotation, symmetric Cauchy stress, objectivity, stress equal to the energy derivative and tangent equal to the stress derivative by finite difference, agreement of the spatial and reference-frame forms, FP32 against FP64, CPU against GPU, and the FEBio material test problem where one exists.

## Results for the three existing models

The three materials already in the repository were rewritten against the interface as a test. Neo-Hookean and isotropic Lee-Sacks were transcribed from the FEBio source in spatial form. Isotropic elastic (St. Venant-Kirchhoff) was written from its energy in the reference form on purpose, so that both ways of writing a model are exercised. The previous files are untouched and serve as the independent comparison.

All figures are relative errors over 66 deformation states (uniaxial, biaxial, volumetric, simple shear and 60 random states), FP64 on CPU unless stated. Parameters: Lee-Sacks c0 = 200, c1 = 2968.4, c2 = 0.2661, k = 5000 (the mitral valve deck); neo-Hookean E = 1e6, v = 0.45; isotropic elastic E = 1e6, v = 0.3.

| Check | neo-Hookean (written spatially) | isotropic elastic (written in reference form) | isotropic Lee-Sacks (written spatially) |
|---|---|---|---|
| Cauchy stress vs FEBio closed form | 8.8e-16 | 5.7e-16 | 1.0e-15 |
| 6x6 spatial tangent vs FEBio closed form | 7.4e-16 | 3.2e-16 | 6.9e-16 |
| Cauchy stress asymmetry | 0.0e+00 | 7.1e-17 | 0.0e+00 |
| 6x6 tangent asymmetry | 0.0e+00 | 8.0e-17 | 4.9e-17 |
| Cauchy stress vs finite difference of energy | 3.5e-10 | 2.7e-10 | 2.3e-10 |
| Spatial tangent vs finite difference of stress | 3.2e-10 | 2.9e-10 | 2.0e-10 |
| Objectivity, sigma(RF) = R sigma(F) R^T | 9.7e-16 | 4.8e-16 | 9.7e-16 |
| Stress at F = I and at a rigid rotation | 4.4e-18 | 8.3e-17 | 1.2e-18 |
| First Piola stress consistent with Cauchy stress | 3.5e-16 | 2.6e-16 | 2.9e-16 |
| First Piola stress vs previous implementation | 1.2e-16 | 0.0e+00 | 4.3e-16 |
| First Piola tangent action vs previous implementation | 4.0e-16 | 0.0e+00 | 4.4e-16 |
| FP32 vs FP64 stress | 3.5e-07 | 2.9e-07 | 2.1e-07 |
| Parameter gradient, tape vs finite difference | 8.1e-11 | 3.6e-10 | 3.3e-10 |

What this shows:

- The conversion changes nothing: each model agrees with its previous implementation to rounding.
- A model written spatially and a model written in the reference form both reproduce FEBio's Cauchy stress and spatial tangent to rounding. For isotropic elastic this is the stronger check, since FEBio's `b`-based expressions were never typed into the Warp model.
- The Cauchy form that elements now consume and the first Piola form of the first draft are the same physics to rounding, so the change of consumed form costs nothing in accuracy.
- Stress, tangent and parameter gradients are mutually consistent, and single precision compiles from the same source.

What this does not show: these are material-point comparisons against closed forms transcribed from the FEBio source, not against a FEBio run. No element, no solve and no GPU execution is involved. Running the FEBio material test problems listed below is the next level of evidence.

Reproduce with `python examples/material_interface/run.py`; the numbers are stored in `validation/material_interface/results.json`, and `tests/test_material_interface.py` asserts on them.

## Coupling to MFEM

Ryan Black's `FEBioMFEMiFSI` plugin already couples MFEM to FEBio materials, and it sets the terms for how FEBio-Warp should couple.

**How the plugin does it today.** The immersed solid is an FEBio solid domain living inside the MFEM solve. At each quadrature point it writes F (evaluated at the generalized-alpha intermediate time) into the FEBio material point and calls the material's virtual `Stress` and `SolidTangent`; the three-field domain calls `DevStress`, `DevTangent`, `UJ` and `UJJ`. Any FEBio elastic material therefore runs in iFSI unchanged. The plugin registers neo-Hookean, incompressible neo-Hookean, St. Venant-Kirchhoff, Fung orthotropic and uncoupled Ogden. It runs on CPU: the device option lists CPU only, and per-point virtual calls on stateful material objects do not move to GPU as written.

**What that means for FEBio-Warp.**

- The exchange format with MFEM is Cauchy stress and the spatial tangent in FEBio conventions, because that is what the plugin's domains consume. A model written in the spatial form already has both. One generic exchange kernel serves every model: it returns the model's Cauchy stress and the full spatial tangent as a symmetric 6x6 in FEBio `tens4ds` order (11, 22, 33, 12, 23, 13), and writes into caller-owned memory without a copy. Since elements now consume the same quantities, the kernel adds no mechanics of its own.

  Checked: the kernel's output matches the closed forms in the FEBio source to rounding for all three models, written directly into buffers the caller allocated (CPU; the GPU pointer path is the same call but has not been run).
- The material must be a pure function of the F it is handed, since the caller chooses the time level. The proposal already requires this.
- The dev/vol split lines up. iFSI's `QUAD` volumetric energy is FEBio's `Abaqus` pressure model, and its `ST91` gives the same pressure as FEBio's `NIKE3D` and `Abaqus (GOH)` models. One shared volumetric file can serve FEBio-Warp, the plugin's three-field domain and native iFSI.
- iFSI passes fiber directions as a vector coefficient; FEBio passes a local frame plus an angle. We take the frame as input and derive fiber vectors inside the model, so either source can feed it.

**Three routes, in order of effort.**

1. **Through FEBio (works today).** A model that exists in FEBio runs in iFSI through the plugin. This fits the agreed direction that constitutive exploration stays in FEBio: a new law such as anisotropic Lee-Sacks or superelastic nitinol is added to FEBio once and is then available to MFEM, with FEBio-Warp carrying a transcription checked against it.
2. **Batched GPU evaluation (when iFSI moves to GPU).** The quadrature loop in the plugin's stress update and material stiffness is the seam. MFEM fills a device array of F for all points, a FEBio-Warp kernel returns Cauchy stress and spatial tangent, or the tangent action for partial assembly, into MFEM-owned memory without a copy. The Warp half of this is built (the exchange kernel above). The MFEM half is a thin C++ shim in the plugin that we have not written.
3. **Single-source C++ kernels (later, if needed).** Material math in a header that MFEM includes directly and Warp calls through `wp.func_native`. No Python on the MFEM side, but adjoints for Warp must be written by hand, so only for models that have stabilized.

## Verification problems

Jeff pointed us to the material problems in the FEBio test suite (`ma*.feb`; https://github.com/febiosoftware/TestSuite and https://repo.febio.org/permalink/project/34). Searching the suite (commit of November 11, 2026) for our material types:

| Material | Problems in the suite |
|---|---|
| Ogden unconstrained | `ma02.feb`, also `co38.feb` to `co40.feb` |
| Ogden (uncoupled) | `ma01.feb`, `ma16.feb`, `co28.feb` and others |
| Holzapfel-Gasser-Ogden (uncoupled) | `ma34.feb`, `fs07.feb` |
| HGO unconstrained | `mp10.feb` |
| neo-Hookean | many (317 files use it) |
| isotropic elastic | 12 files |
| isotropic Lee-Sacks | none found |

We have not run these yet. There is no Lee-Sacks problem in the suite, coupled or uncoupled; we could contribute a single-element one if that is useful.

## Answered so far

- **Consumed form.** Jeff: the typical choices are total Lagrangian with second Piola stress or updated Lagrangian with Cauchy stress, with updated Lagrangian having some advantage for contact and for constraints and boundary conditions that depend on the current geometry. Adopted: updated Lagrangian with Cauchy stress (item 1).
- **Uncoupled models.** Jeff: use them only for near-incompressibility and only with a multi-field element; otherwise use the coupled model with a displacement element. Adopted (item 5).
- **Verification.** Jeff: the `ma*.feb` problems in the test suite. Adopted (previous section).

## Open questions

1. The TPVR RVOT model used uncoupled HGO at a bulk to shear ratio of 7.5. Should that case be treated as coupled going forward?
2. `tangent_scale` in isotropic Lee-Sacks: keep it as an optional solver aid, or leave it out and rely on the exact tangent?
3. Fiber models: is a per-point frame Q plus an angle, as in `FEHolzapfelGasserOgden`, the right general input, or would you rather pass fiber vectors directly?
4. History: is one flat vector of doubles per point enough for what you would expect of a superelastic law, or do you see a need for typed state?
5. Would a single-element Lee-Sacks problem be a useful addition to the FEBio test suite?

## Question for Ryan

For the GPU version of the immersed solid, would you rather receive Cauchy stress and a full spatial tangent per quadrature point, or a tangent action for partial assembly, and at which time level (the generalized-alpha intermediate F, as now)?

## Next steps if agreed

Done in the draft branch: the three existing materials in this form with no change in results, and the conformance test. Next, if the structure is agreed: run the neo-Hookean and isotropic elastic test problems against FEBio, retire the previous files, then Ogden unconstrained (`ma02.feb`) and HGO unconstrained (`mp10.feb`) as the first new models.
