# CardioFEM valve pressurization and closure

[Case settings and comparison records](febio_cases.json) keeps the original
valve's material coefficients, thickness summary, boundary conditions, chord
force curves, pressure history, contact settings, time integration and solver
controls. It also records the archived first-step and later comparisons,
including each comparison's convergence and agreement status.

The original CardioFEM convention is mm–kg–s, with force in mN and stress in
kPa. Its Lee–Sacks coefficients and solver tangent scaling belong to that
archived valve case. The standalone material-point example uses its own SI
inputs. Each JSON record identifies the original path and file hash.

Run this case in the original CardioFEM prototype. The archived anatomical
geometry, thickness field, mesh connectivity and full simulation outputs
remain in that project. The archived comparisons did not establish full
valve agreement.
