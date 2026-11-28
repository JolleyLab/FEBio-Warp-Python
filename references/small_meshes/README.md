# Small multi-element references

Two inputs that extend the single-element references to several elements, so
that assembly across elements and the averaging of shell directors at shared
nodes are exercised.

| Input | Mesh |
| --- | --- |
| `hex_block.feb` | 2 x 2 x 2 hex8 block with a displaced interior node; bottom fixed, top moved in z |
| `shell_patch.feb` | 3 x 3 quad4 shell patch on a curved surface, thickness 0.1; one edge clamped, the opposite edge moved |

The inputs were written by `warp_febio.io.write_feb` from models built in code.
The FEBio results in `febio/` come from the same provisional FEBio 4.13.0 source
build described in `../single_element/README.md`.
