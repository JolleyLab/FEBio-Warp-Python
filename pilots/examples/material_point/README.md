# CardioFEM material-point example

From the root: `python examples/material_point/run.py --device cpu`.
The deformation gradient is prescribed; the response contains energy, PK1 stress,
Cauchy stress and J with explicit units. No spatial mesh or equilibrium solve occurs.
The synthetic parameters are not calibrated tissue values. Run the inherited
material tests with `python -m unittest discover -s tests -v`.
The original CardioFEM repository is not needed for this isolated example.
