# FEBio output definitions

Source version is exactly `4.9.0.6dcb43939` from patch.log. Source files fetched at commit `6dcb43939`; archived source evidence in `sources/`. URL template: `https://github.com/febiosoftware/FEBio/blob/6dcb43939/<path>`.

One CSV exists: `reference/febio/patch_output.csv`. Despite suffix, this is whitespace-delimited FEBio logfile element_data, not a spreadsheet export. `patch.feb` explicitly requests `<element_data data="Ux;Uy;Uz;s1;E1" file="patch_output.csv" elem_set="patch"/>`. Each block has 1863 rows with exact original element IDs 1..1863 (all verified); each row has ID and 5 values. The ElementSet `patch` includes those elements, all quad4. No nodal data in this CSV.

|File|Column|Exact quantity|Location/reduction/frame|Units (confirmed mm-N-s)|Source|PatchWarp equivalence|
|---|---|---|---|---|---|---|
|patch_output.csv|first integer|Original FEBio element ID|Original quad4 element; not node ID|none|patch.feb element_data and ElementSet; blocks verified|Requires mesh mapping, not same ID assumed|
|patch_output.csv|Ux|U_xx where U=sqrt(F^T F), right stretch|Arithmetic unweighted mean of values at all 8 shell integration points, global reference Cartesian axes|dimensionless|FEBioMechModule.cpp:969; FEBioMechData.cpp:628; FEElasticMaterialPoint.cpp:163|Derived from saved full 3D GP strain: U=sqrt(I+2E), then diagonal and mean|
|patch_output.csv|Uy|U_yy|Same|dimensionless|FEBioMechData.cpp:642 and registry immediately after Ux|Same derivation as Ux|
|patch_output.csv|Uz|U_zz|Same|dimensionless|FEBioMechData.cpp:656 and registry immediately after Ux|Same derivation as Ux|
|patch_output.csv|s1|Mean of largest principal Cauchy stress per integration point|Unweighted average of per-point lambda_max(sigma); objective scalar from spatial Cauchy tensor; NOT max eigenvalue of mean stress|MPa=N/mm²|FEBioMechModule.cpp:1016; FEBioMechData.cpp:1333; FECore/mat3d.cpp:147|Derived from saved stress_ip_kpa: convert to MPa, largest eigenvalue at each GP, then mean|
|patch_output.csv|E1|Mean of largest principal Green-Lagrange strain per integration point, E=(F^T F-I)/2|Unweighted average of per-point lambda_max(E); objective scalar of referential strain; NOT engineering/log strain; NOT eigenvalue of average E|dimensionless|FEBioMechModule.cpp:960; FEBioMechData.cpp:474; FEElasticMaterialPoint.cpp:261; FECore/mat3d.cpp:147|Derived from saved strain_ip: largest eigenvalue at each GP, then mean|

No nodal extrapolation, spatial smoothing, area/volume weighting or thickness-surface selection occurs in these CSV reductions. All shell integration points (in-plane and through thickness) enter. `exact_eigen` l[0] is largest for its nondegenerate analytical branch; isotropic fallback has three equal diagonal values, so ordering is immaterial here. Zero state values 1,1,1,0,0 independently corroborate definitions but were not used to infer them.

Timestamps s: 0, 0.001, 0.0038, 0.00804, 0.013432, 0.0197456, 0.02679648, 0.034437184, 0.0425497472, 0.0510397978, 0.0598318382, 0.0688654706, 0.0780923765, 0.0874739012, 0.0969791209, 0.1. Blocks Step 0..15. Pressure at a block is 0.0133*(t/0.1) MPa. Last state 0.1 s is end of ramp, not post-ramp hold/equilibrium.

## Additional XPLT quantities

`patch.feb` requests displacement, stress, Lagrange strain, relative volume. These are distinct from CSV quantities.

- displacement: global x/y/z DOF vector, nodal, mm, for shell top/reference nodes. FEBioMechPlot.cpp:85 FEPlotNodeDisplacement.
- stress: arithmetic mean tensor of Cauchy stress at all element integration points, global spatial Cartesian frame, MPa. FEBioMechPlot.cpp:882 FEPlotElementStress; FECore/writeplot.h:154. Not CSV s1 (which averages eigenvalues first).
- Lagrange strain: arithmetic mean tensor E at integration points for this default elastic-shell, global reference Cartesian frame, dimensionless. FEBioMechPlot.cpp:2505; writeplot.h:104. Not CSV E1.
- relative volume: arithmetic mean J=det(F) over integration points, dimensionless. FEBioMechPlot.cpp:2184 default shell branch. No GP data in these reductions, so original per-GP max principal data cannot generally be recovered from these mean tensors.

## Physical interpretation / source confirmations

- `isotropic elastic` maps to FEIsotropicElastic (FEBioMechModule.cpp:350). It is St Venant-Kirchhoff, not neo-Hookean: FEIsotropicElastic.cpp PK2Stress returns lambda tr(E) I + 2 mu E; StrainEnergyDensity returns lambda tr(E)^2/2 + mu E:E. E=1 MPa, nu=0.3, density=1e-9 tonne/mm³ =1000 kg/m³.
- quad4 defaults FE_SHELL_QUAD4G8 (FEModelBuilder.cpp:496), NEW_SHELL default (line79), coupled material -> elastic-shell in FESolidDomainFactory.cpp:138-148. Eight integration points, not EAS or ANS.
- Geometry coordinates r0 are shell top/reference surface: FESSIShellDomain.cpp:387 uses r0 - D*(1-eta)/2, top eta=+1, bottom eta=-1. Thickness 0.5 mm. Not a midsurface with ±h/2 offsets.
- `zero displacement` only x,y,z (FEFixedDisplacement.cpp Init), not shell bottom variables. FEFixedBC.cpp simply fixes listed DOFs; FEElasticShellDomain.cpp:113 activates both ordinary and shell DOFs. Thus supplied perimeter fixes top displacement, bottom shell displacement remains free (no zero shell displacement BC present).
- positive pressure on top (`shell_bottom=0`) applies traction -p*n_current, FEPressureLoad.cpp LoadVector. Follower because linear=0; symmetric_stiffness=1 symmetrizes load stiffness.
- Dynamics: supplied DYNAMIC, beta=.25 gamma=.5 alpha=1 rhoi=-2; adaptive 15 accepted timesteps after zero state, ends ramp at .1 s. No inference of equilibrium.
- Runtime from log: solve 1.64429 s (including IO .317263 s); total elapsed 1.69116 s; CPU Pardiso; not comparable hardware/workload to current Warp timings.

## Unresolved / constraints

All CSV column meanings are established with exact-version source. Matching native reductions have been implemented in compare.py. Spatial transfer between different meshes adds an area-averaging convention and is diagnostic, not identical material-point sampling. Full tensor averages in XPLT do not uniquely reconstruct CSV principal-first reductions. No clinical material validation implied. Units were not encoded explicitly in input; user confirmed mm-N-s/MPa in this session.

## Block metadata and output availability

`*Step` is the accepted FEBio state/step index (0 is initialization), `*Time` is physical simulation time in seconds, and `*Data` lists the five element quantities. Neither metadata field is a nodal/elemental physical field. The first integer on every data row is the original element ID. See `febio_extracted.npz` for all states and IDs. Only one CSV exists in the reference directory; all its columns are covered above.

Current CardioFEM-backed PatchWarp state stores 8-point tensors only for its final state. XPLT mean stress/strain and J have equivalents obtained by arithmetic GP reduction of those saved tensors/J. XPLT nodal displacement has a direct quantity equivalent in `(positions-rest)*1000` mm. Different meshes, material laws and histories prevent these from constituting a matched validation case.

## Primary source links

- [Registry and output names](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEBioMechModule.cpp)
- [CSV reductions](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEBioMechData.cpp)
- [Strain/stretch definitions](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEElasticMaterialPoint.cpp)
- [Isotropic elastic constitutive law](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEIsotropicElastic.cpp)
- [XPLT quantities](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEBioMechPlot.cpp)
- [Pressure implementation](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEPressureLoad.cpp)
- [Shell kinematics](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FESSIShellDomain.cpp)
- [Fixed translations](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FEBioMech/FEFixedDisplacement.cpp)

Source line references above refer to the archived exact-version files, not current upstream HEAD. Units are supplied by the user's explicit mm–N–s/MPa confirmation; FEBio itself does not infer these units.

XPLT symmetric-tensor component order is xx, yy, zz, xy, yz, xz, established by [FEDataStream.h:54](https://github.com/febiosoftware/FEBio/blob/6dcb43939/FECore/FEDataStream.h#L54). Full Frobenius comparisons count both symmetric off-diagonal entries. Missing XPLT units are overridden explicitly by the confirmed unit system; the reader’s fallback dimensionless labels are not used.
