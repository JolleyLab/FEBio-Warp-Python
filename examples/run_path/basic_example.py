"""The run path on the basic example: one hex8, one material, prescribed
nodal displacements with free lateral unknowns.

    python examples/run_path/basic_example.py [device]     device: cpu (default) or cuda:0

Each numbered block is one stage of the path. Template code, for discussion.
"""

import os
import sys

import numpy as np
import warp as wp

from warp_febio.analysis.steps import build_system, run
from warp_febio.io import read_feb, write_xplt
from warp_febio.sensitivity.adjoint import parameter_gradient

wp.config.quiet = True
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"  # for a GPU: cuda:0
HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "..", "..", "references", "single_element")
NAME = "hex8_neo_hookean"

# 1. Read the FEBio input into the Model (pull request 5).
model = read_feb(os.path.join(REF, NAME + ".feb"))

# 2. Build the system: a field table and a list of terms, on the device.
system = build_system(model, device=DEVICE)
print(f"entries (node, variable): {system.n}, 3x3 blocks; free components: {int(system.fields.free.sum())}")
for term in system.terms:
    print(
        f"term: {term.name}  {term.febio_type}  {term.element_type}  "
        f"{term.material.febio_type}  parameters {term.parameter_names}"
    )

# 3. Run the step: time stepping, Newton iterations, linear solves.
results, states = run(system, log=print, keep_states=True)


# 4. Compare with FEBio's log data for the same input.
def febio_log(path, n, columns):
    rows = [line.split() for line in open(path) if not line.startswith("*")]
    return np.array(rows, dtype=float)[:, 1:].reshape(n, -1, columns)


n = len(results.states)
u_ref = febio_log(os.path.join(REF, "febio", NAME + "_nodes.txt"), n, 3)
u = np.stack([s.displacement for s in results.states])
print(f"largest relative difference from FEBio, displacement: {abs(u - u_ref).max() / abs(u_ref).max():.1e}")

# 5. Write the plot file FEBio Studio reads.
out = os.path.join(HERE, NAME + "_run_path.xplt")
write_xplt(model, results, out)
print("wrote", out)

# 6. Sensitivity: d(output)/d(material parameters) at the final state, from
#    one transpose solve and one reverse pass of the residual.
final = states[-1]
free = system.fields.free
u_final = final.u.numpy()
gradients, _ = parameter_gradient(system, final, dJ_du=u_final * free)
domain = system.terms[0]
print("output J = 1/2 |free displacements|^2 =", 0.5 * (u_final**2 * free).sum())
for pname, g in zip(domain.parameter_names, gradients[domain.name + ".material"][0]):
    print(f"  dJ/d{pname} = {g:.6e}")
