# Historical synthetic FEBio tension target

`solid_tension.feb` is the original synthetic 30-node, 8-hex8 job input. It prescribes
2 units of x extension on a 20 x 10 x 1 block; neo-Hookean E=1, nu=.45, static
analysis and a linear load curve. The intended interpretation is mm/N/MPa, but the
XML has no explicit unit declaration. Retained density=1 is inactive in this static case.

Run with a separately installed FEBio executable from this directory:
`febio4 -i solid_tension.feb`. Final archived nodal displacement is in
`validation/solid_reference/` relative to repository root.

No corresponding CardioFEM global solid solver was found. Reproducing that side
requires recovering the original implementation or building a verified volume
solver. The new affine linear-elastic FEM example is a separate verification,
not a reproduction of this finite-strain neo-Hookean reference.

The matching 3D material-point law is available in
[`materials/neo_hookean.py`](../../src/warp_febio/materials/neo_hookean.py).
Finite-strain element assembly and the equilibrium driver remain planned.


[Case settings and comparison records](febio_cases.json) also documents the
earlier 288-Tet4 leaflet pressurization study. That study used Newton/Warp with
a different constitutive response from the FEBio Neo-Hookean input. Its archived
comparison is diagnostic. Reproducing it requires the original benchmark runner
and deck referenced in the record.
