# Single-element references

Two FEBio spec 4.0 inputs, each one element, one material and prescribed nodal
displacements, used to exercise the whole path: read the input, build the
model, solve in Warp, write results. This follows Steve Maas's suggestion to
start from a small end-to-end example.

| Input | Element | Unknowns |
| --- | --- | --- |
| `hex8_neo_hookean.feb` | one hex8, `neo-Hookean`, mildly distorted | lateral displacement of the moved face (8) |
| `quad4_shell_neo_hookean.feb` | one quad4 `elastic-shell`, `neo-Hookean`, thickness 0.1 | in-plane y and back-face displacement of the moved edge (8) |

Both use `max_ups` 0 (full Newton) and tolerances tighter than FEBio's defaults,
so each code converges well below the comparison tolerance.

## FEBio results in `febio/`

Produced by running these inputs unchanged in FEBio 4.13.0, built from source
at commit `552138d` with GCC, without Intel MKL (skyline linear solver). This is
a provisional build, not an official release; regenerate on a release build
before relying on these beyond this draft.

- `*.xplt`: FEBio's plot file (single precision).
- `*_nodes.txt`, `*_elements.txt`: FEBio log data, displacement and
  element-average stress, at the 12 significant digits FEBio prints.

To regenerate: `febio4 -i hex8_neo_hookean.feb` and the same for the shell, in
this folder, then move the outputs into `febio/`.
