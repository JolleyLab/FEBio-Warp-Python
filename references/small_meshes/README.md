# Small multi-element references

Two inputs that extend the single-element references to several elements, so
that assembly across elements and the averaging of shell directors at shared
nodes are exercised.

| Input | Mesh |
| --- | --- |
| `hex_block.feb` | 2 x 2 x 2 hex8 block with a displaced interior node; bottom fixed, top moved in z |
| `shell_patch.feb` | 3 x 3 quad4 shell patch on a curved surface, thickness 0.1; one edge clamped, the opposite edge moved |
| `two_material_pressure.feb` | two hex8 domains of four elements each, stacked in z, one neo-Hookean and one isotropic elastic; bottom fixed, follower pressure on top. Results include nodal reaction forces |
| `shell_patch_truss_chords.feb` | the shell patch of `shell_patch.feb`, clamped on one edge, with its opposite edge held by four `linear truss` chords from one fixed point; lifted by a nodal load on every node that is not clamped. Trusses sharing nodes with shells |
| `tet4_block.feb` | the distorted cube of `../single_element/hex8_neo_hookean.feb` split into six tet4 elements; bottom fixed, top moved in z |

The inputs were written by `warp_febio.io.write_feb` from models built in code.
The FEBio results in `febio/` come from the same provisional FEBio 4.13.0 source
build described in `../single_element/README.md`.

`tet4_block.feb` was written by hand from the single hex8 input, and its FEBio
results are from the build at commit `e4df3f8` made by `tools/build_febio.sh`.

`two_material_pressure.feb` was written by hand; its FEBio results are from the
same build at `e4df3f8`.

`shell_patch_truss_chords.feb` was derived from `shell_patch.feb` by hand; its
FEBio results (the two log data files) are from the same build at `e4df3f8`.
