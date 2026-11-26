"""Generate the small FEBio inputs used as verification cases.

FEBio 4.0 input files for features the FEBio test suite covers thinly for
the valve and device applications (see docs/design/verification.md). Every
one is synthetic: generated geometry, with parameter values taken from the
FEBio test suite or from the published valve model in this repository. Each
is an ordinary FEBio input file, so it can also be offered to the FEBio team
for the suite.

Usage::

    python references/small_problems/generate.py     # writes inputs/*.feb

Groups:

``mp_*``  Material point. One hex8 element with every node driven by FEBio's
          "prescribed deformation" boundary condition, x = F X, scaled by a
          ramp. The deformation is uniform, so FEBio's element log variables
          (averages over integration points) are point values.
          Logged: F (9), Cauchy stress (6), spatial tangent (21).
``ch_*``  A chord: one nonlinear spring taken through slack, tension and back.
``bk_*``  Small solid problems: a three-field HGO block, a damped block, a
          block whose modulus follows a load curve, a contact potential
          patch, and a ring crimped in a tube.
``sh_*``  Shell problems: a Lee-Sacks patch under pressure and self-contact
          of a folded strip.

FEBio input format: https://help.febio.org/docs/FEBioUser-4-7/UM47-Chapter-3.html
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "inputs"

# --------------------------------------------------------------------------
# Materials. Names are FEBio type strings. Parameter values and their source.
# --------------------------------------------------------------------------
MATERIALS = {
    # Valve leaflet values of the mitral model in
    # references/febio/originals/cardiofem/valve_pressurization (kPa, mm).
    # tangent_scale is left at its default of 1 so the logged tangent is the
    # true one; the valve model sets it to 10 as a solver aid.
    "lee_sacks": (
        "isotropic Lee-Sacks",
        {"density": 1.04e-6, "c0": 200, "c1": 2968.4, "c2": 0.2661, "k": 5000},
        "",
    ),
    # FEBio suite co62 (contact potential problem).
    "neo_hookean": ("neo-Hookean", {"density": 1, "E": 1, "v": 0.3}, ""),
    # FEBio suite ma34 (uncoupled HGO).
    "hgo_uncoupled": (
        "Holzapfel-Gasser-Ogden",
        {
            "density": 1,
            "k": 100,
            "c": 0.00764,
            "k1": 0.9966,
            "k2": 524.6,
            "kappa": 0.226,
            "gamma": 49.98,
        },
        '<mat_axis type="vector"><a>1,0,0</a><d>0,1,0</d></mat_axis>',
    ),
    # Same fiber parameters with the coupled form FEBio-Warp implements.
    "hgo_unconstrained": (
        "HGO unconstrained",
        {
            "density": 1,
            "c": 0.00764,
            "k1": 0.9966,
            "k2": 524.6,
            "kappa": 0.226,
            "gamma": 49.98,
            "k": 100,
        },
        '<mat_axis type="vector"><a>1,0,0</a><d>0,1,0</d></mat_axis>',
    ),
    # The two HGO forms again with material axes away from the global frame, so
    # that the construction of the axes from a and d is part of the comparison.
    "hgo_uncoupled_rotated": (
        "Holzapfel-Gasser-Ogden",
        {
            "density": 1,
            "k": 100,
            "c": 0.00764,
            "k1": 0.9966,
            "k2": 524.6,
            "kappa": 0.226,
            "gamma": 49.98,
        },
        '<mat_axis type="vector"><a>1,2,0.5</a><d>-1,0.3,2</d></mat_axis>',
    ),
    "hgo_unconstrained_rotated": (
        "HGO unconstrained",
        {
            "density": 1,
            "c": 0.00764,
            "k1": 0.9966,
            "k2": 524.6,
            "kappa": 0.226,
            "gamma": 49.98,
            "k": 100,
        },
        '<mat_axis type="vector"><a>1,2,0.5</a><d>-1,0.3,2</d></mat_axis>',
    ),
    # Device frame of the published TPVR model (Zelonis et al., Table 1; kPa, mm).
    "isotropic_elastic": (
        "isotropic elastic",
        {"density": 6.45e-6, "E": 4.0e7, "v": 0.33},
        "",
    ),
}


def _rot(axis, deg):
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    C = 1 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def _matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


# Target deformation gradients at the end of the ramp. The applied gradient at
# ramp value s is I + s (F - I), so each file gives ten states along a line.
PATHS = {
    "uniaxial": [[1.30, 0, 0], [0, 0.92, 0], [0, 0, 0.92]],
    "biaxial": [[1.20, 0, 0], [0, 1.20, 0], [0, 0, 0.75]],
    "shear": [[1, 0.40, 0], [0, 1, 0], [0, 0, 1]],
    # A stretch with all components, then a rigid rotation: tests objectivity
    # and every tangent component at once.
    "general": _matmul(
        _rot((1, 2, 3), 25.0),
        [[1.15, 0.10, 0.05], [0.10, 0.95, -0.08], [0.05, -0.08, 1.05]],
    ),
}

# Material point cases. Kept to the models the applications need and the suite
# covers thinly or not at all, plus one neo-Hookean case as a control, since
# that model is in 220 in-scope suite problems.
MATERIAL_POINT_CASES = [
    ("lee_sacks", "uniaxial"),  # no suite problem uses this model
    ("lee_sacks", "biaxial"),  # the leaflet loading state
    ("lee_sacks", "shear"),
    ("lee_sacks", "general"),
    ("hgo_uncoupled", "biaxial"),  # one suite problem (ma34)
    ("hgo_uncoupled", "general"),
    ("hgo_unconstrained", "biaxial"),  # no in-scope suite problem
    ("hgo_unconstrained", "general"),
    ("neo_hookean", "general"),  # control
    ("hgo_uncoupled_rotated", "general"),  # axes away from the global frame
    ("hgo_unconstrained_rotated", "general"),
    ("isotropic_elastic", "general"),  # device frame, no earlier case
]

# The fiber term of the ma34 parameters (k2 = 524.6) stiffens very quickly, so
# the HGO cases use a quarter of each path: about 5 percent strain.
AMPLITUDE = {
    "hgo_uncoupled": 0.25,
    "hgo_unconstrained": 0.25,
    "hgo_uncoupled_rotated": 0.25,
    "hgo_unconstrained_rotated": 0.25,
}


def target(mat_key, path_key):
    """Deformation gradient at the end of the ramp for one material point case."""
    a = AMPLITUDE.get(mat_key, 1.0)
    F = PATHS[path_key]
    return [[(i == j) + a * (F[i][j] - (i == j)) for j in range(3)] for i in range(3)]


F_VARS = "Fxx;Fxy;Fxz;Fyx;Fyy;Fyz;Fzx;Fzy;Fzz"
S_VARS = "sx;sy;sz;sxy;syz;sxz"
# FEBio's 21 tangent log variables (FEBioMech/FEBioMechModule.cpp). Read the
# values by position, not by name: FELogElemElasticity returns entry n of a
# tens4dmm, which stores the full 6x6 column by column (FECore/tens4dmm.hpp),
# so value n is entry (n % 6, n // 6) in the order xx, yy, zz, xy, yz, xz.
# The 21 values are therefore columns xx, yy, zz in full and the first three
# rows of column xy. The shear-shear entries are not written.
C_VARS = (
    "cxxxx;cxxyy;cyyyy;cxxzz;cyyzz;czzzz;cxxxy;cyyxy;czzxy;cxyxy;cxxyz;"
    "cyyyz;czzyz;cxyyz;cyzyz;cxxxz;cyyxz;czzxz;cxyxz;cyzxz;cxzxz"
)


# --------------------------------------------------------------------------
# Mesh and XML helpers
# --------------------------------------------------------------------------
class Grid:
    """A structured block of hex8 elements with FEBio node ordering."""

    def __init__(self, n, size, origin=(0.0, 0.0, 0.0), first_node=1, first_elem=1,mapping=None):  # fmt: skip
        """mapping, if given, takes the block coordinates to the real ones."""
        self.n, self.size, self.origin = n, size, origin
        self.first_node, self.first_elem = first_node, first_elem
        nx, ny, nz = n
        self.nodes = {}
        for k, j, i in itertools.product(range(nz + 1), range(ny + 1), range(nx + 1)):
            self.nodes[self.nid(i, j, k)] = (
                origin[0] + size[0] * i / nx,
                origin[1] + size[1] * j / ny,
                origin[2] + size[2] * k / nz,
            )
            if mapping:
                self.nodes[self.nid(i, j, k)] = mapping(*self.nodes[self.nid(i, j, k)])
        self.elems = {}
        e = first_elem
        for k, j, i in itertools.product(range(nz), range(ny), range(nx)):
            self.elems[e] = [
                self.nid(i, j, k), self.nid(i + 1, j, k),
                self.nid(i + 1, j + 1, k), self.nid(i, j + 1, k),
                self.nid(i, j, k + 1), self.nid(i + 1, j, k + 1),
                self.nid(i + 1, j + 1, k + 1), self.nid(i, j + 1, k + 1),
            ]  # fmt: skip
            e += 1

    def nid(self, i, j, k):
        nx, ny, _ = self.n
        return self.first_node + i + (nx + 1) * (j + (ny + 1) * k)

    def face_nodes(self, axis, side):
        """Node ids on the min (side 0) or max (side 1) face along an axis."""
        lim = self.n[axis] if side else 0
        out = []
        for k, j, i in itertools.product(*(range(m + 1) for m in reversed(self.n))):
            if (i, j, k)[axis] == lim:
                out.append(self.nid(i, j, k))
        return out

    def face_quads(self, axis, side):
        """Outward-facing quad4 facets on the min (0) or max (1) face of an axis."""
        u, v = [a for a in range(3) if a != axis]
        lim = self.n[axis] if side else 0
        quads = []
        for jv, iu in itertools.product(range(self.n[v]), range(self.n[u])):

            def node(du, dv):
                ijk = [0, 0, 0]
                ijk[axis], ijk[u], ijk[v] = lim, iu + du, jv + dv
                return self.nid(*ijk)

            q = [node(0, 0), node(1, 0), node(1, 1), node(0, 1)]
            # (u, v, axis) is right-handed for axis 0 and 2, left-handed for 1.
            outward = (side == 1) == (axis != 1)
            quads.append(q if outward else q[::-1])
        return quads

    def nodes_xml(self, name):
        rows = "\n".join(
            f'\t\t\t<node id="{i}">{x:.17g},{y:.17g},{z:.17g}</node>'
            for i, (x, y, z) in self.nodes.items()
        )
        return f'\t\t<Nodes name="{name}">\n{rows}\n\t\t</Nodes>'

    def elems_xml(self, name):
        rows = "\n".join(
            f'\t\t\t<elem id="{i}">{",".join(map(str, c))}</elem>'
            for i, c in self.elems.items()
        )
        return f'\t\t<Elements type="hex8" name="{name}">\n{rows}\n\t\t</Elements>'


def nodeset(name, ids):
    return f'\t\t<NodeSet name="{name}">{",".join(map(str, ids))}</NodeSet>'


def surface(name, quads):
    rows = "\n".join(
        f'\t\t\t<quad4 id="{i + 1}">{",".join(map(str, q))}</quad4>'
        for i, q in enumerate(quads)
    )
    return f'\t\t<Surface name="{name}">\n{rows}\n\t\t</Surface>'


def material_xml(mid, name, key, overrides=None):
    ftype, params, extra = MATERIALS[key]
    params = {**params, **(overrides or {})}
    body = "".join(f"\n\t\t\t<{k}>{v}</{k}>" for k, v in params.items())
    if extra:
        body += f"\n\t\t\t{extra}"
    return (
        f'\t\t<material id="{mid}" name="{name}" type="{ftype}">{body}\n\t\t</material>'
    )


def load_curve(cid, points, interpolate="LINEAR"):
    pts = "".join(f"\n\t\t\t\t<pt>{t:g},{v:g}</pt>" for t, v in points)
    return (
        f'\t\t<load_controller id="{cid}" type="loadcurve">\n'
        f"\t\t\t<interpolate>{interpolate}</interpolate>\n"
        f"\t\t\t<extend>CONSTANT</extend>\n"
        f"\t\t\t<points>{pts}\n\t\t\t</points>\n"
        f"\t\t</load_controller>"
    )


def control(analysis="STATIC", steps=10, dt=0.1, extra=""):
    return (
        "\t<Control>\n"
        f"\t\t<analysis>{analysis}</analysis>\n"
        f"\t\t<time_steps>{steps}</time_steps>\n"
        f"\t\t<step_size>{dt:g}</step_size>\n"
        '\t\t<solver type="solid">\n'
        "\t\t\t<symmetric_stiffness>symmetric</symmetric_stiffness>\n"
        "\t\t\t<max_refs>25</max_refs>\n"
        "\t\t\t<dtol>1e-7</dtol>\n"
        "\t\t\t<etol>1e-12</etol>\n"
        "\t\t\t<rtol>0</rtol>\n"
        "\t\t\t<lstol>0.9</lstol>\n"
        f"{extra}"
        '\t\t\t<qn_method type="BFGS">\n'
        "\t\t\t\t<max_ups>0</max_ups>\n"
        "\t\t\t</qn_method>\n"
        "\t\t</solver>\n"
        "\t</Control>"
    )


def deck(comment, sections):
    body = "\n".join(s for s in sections if s)
    return (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        f"<!--\n{comment.strip()}\n\nGenerated by references/small_problems/generate.py "
        "(FEBio-Warp verification harness). Synthetic.\n-->\n"
        '<febio_spec version="4.0">\n\t<Module type="solid"/>\n'
        f"{body}\n</febio_spec>\n"
    )


def section(tag, *rows):
    return f"\t<{tag}>\n" + "\n".join(rows) + f"\n\t</{tag}>"


def output(*records):
    plot = (
        '\t\t<plotfile type="febio">\n'
        '\t\t\t<var type="displacement"/>\n\t\t\t<var type="stress"/>\n'
        '\t\t\t<var type="reaction forces"/>\n\t\t</plotfile>'
    )
    log = (
        "\t\t<logfile>\n"
        + "\n".join(f"\t\t\t{r}" for r in records)
        + "\n\t\t</logfile>"
    )
    return section("Output", plot, log)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------
def material_point(mat_key, path_key):
    grid = Grid((1, 1, 1), (1, 1, 1))
    F = target(mat_key, path_key)
    fstr = ",".join(f"{F[i][j]:.17g}" for i in range(3) for j in range(3))
    ftype = MATERIALS[mat_key][0]
    comment = (
        f"Material point check: {ftype}, {path_key} deformation.\n"
        "One hex8 element, every node driven by x = F X scaled by a linear ramp,\n"
        "so the deformation is uniform and the logged element values are point\n"
        "values. Log record 1: deformation gradient. 2: Cauchy stress.\n"
        "3: spatial tangent, first 21 entries of the 6x6 stored by column."
    )
    return deck(
        comment,
        [
            control(steps=10, dt=0.1),
            section("Material", material_xml(1, "m", mat_key)),
            section(
                "Mesh",
                grid.nodes_xml("Object1"),
                grid.elems_xml("Part1"),
                nodeset("all", list(grid.nodes)),
            ),
            section("MeshDomains", '\t\t<SolidDomain name="Part1" mat="m"/>'),
            section(
                "Boundary",
                '\t\t<bc name="F" node_set="all" type="prescribed deformation">\n'
                '\t\t\t<scale lc="1">1</scale>\n'
                f"\t\t\t<F>{fstr}</F>\n\t\t</bc>",
            ),
            section("LoadData", load_curve(1, [(0, 0), (1, 1)])),
            output(
                f'<element_data data="{F_VARS}" delim=","/>',
                f'<element_data data="{S_VARS}" delim=","/>',
                f'<element_data data="{C_VARS}" delim=","/>',
            ),
        ],
    )


def chord():
    """One nonlinear spring between a fixed block and a translated block.

    Force curve and scale are those of chord A1 in the mitral model: slack up
    to 4 mm of elongation, then 60 N/mm. The upper block is translated rigidly
    (so it carries no stress) by 0 -> 6 -> 0 mm; the spring force is the only
    load on it and is read as "discrete element force" and as the reaction.
    Expected: force = 60 * max(0, elongation - 4).
    """
    lower = Grid((1, 1, 1), (1, 1, 1))
    upper = Grid((1, 1, 1), (1, 1, 1), origin=(0, 0, 11), first_node=9, first_elem=2)
    a, b = lower.nid(0, 0, 1), upper.nid(0, 0, 0)  # 10 mm apart along z
    return deck(
        chord.__doc__,
        [
            control(steps=24, dt=0.5),
            section("Material", material_xml(1, "block", "neo_hookean")),
            section(
                "Mesh",
                lower.nodes_xml("Lower"),
                upper.nodes_xml("Upper"),
                lower.elems_xml("PartLower"),
                upper.elems_xml("PartUpper"),
                nodeset("lower", list(lower.nodes)),
                nodeset("upper", list(upper.nodes)),
                f'\t\t<DiscreteSet name="chord">\n\t\t\t<delem>{a},{b}</delem>\n\t\t</DiscreteSet>',
            ),
            section(
                "MeshDomains",
                '\t\t<SolidDomain name="PartLower" mat="block"/>',
                '\t\t<SolidDomain name="PartUpper" mat="block"/>',
            ),
            section(
                "Boundary",
                '\t\t<bc name="fix" node_set="lower" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n\t\t</bc>",
                '\t\t<bc name="fixxy" node_set="upper" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>\n\t\t</bc>",
                '\t\t<bc name="pull" node_set="upper" type="prescribed displacement">\n'
                '\t\t\t<dof>z</dof>\n\t\t\t<value lc="1">1</value>\n'
                "\t\t\t<relative>0</relative>\n\t\t</bc>",
            ),
            section(
                "Discrete",
                '\t\t<discrete_material id="1" name="chord" type="nonlinear spring">\n'
                "\t\t\t<scale>60</scale>\n\t\t\t<measure>elongation</measure>\n"
                '\t\t\t<force type="point">\n'
                "\t\t\t\t<interpolate>linear</interpolate>\n"
                "\t\t\t\t<extend>extrapolate</extend>\n"
                "\t\t\t\t<points>\n\t\t\t\t\t<pt>0,0</pt>\n\t\t\t\t\t<pt>4,0</pt>\n"
                "\t\t\t\t\t<pt>5,1</pt>\n\t\t\t\t</points>\n\t\t\t</force>\n"
                "\t\t</discrete_material>",
                '\t\t<discrete dmat="1" discrete_set="chord"/>',
            ),
            section("LoadData", load_curve(1, [(0, 0), (6, 6), (12, 0)])),
            output(
                f'<node_data data="uz;Rz" delim=",">{b}</node_data>',
                '<element_data data="discrete element elongation;discrete element force" '
                'delim="," elem_set="chord"/>',
            ),
        ],
    )


def _block_bcs(grid):
    """Symmetry planes on x=0, y=0, z=0 so a block deforms uniformly."""
    rows = []
    for axis, name in enumerate("xyz"):
        flags = "".join(f"<{d}_dof>{int(d == name)}</{d}_dof>" for d in "xyz")
        rows.append(nodeset(f"{name}0", grid.face_nodes(axis, 0)))
        rows.append(
            f'\t\t<bc name="sym{name}" node_set="{name}0" type="zero displacement">\n'
            f"\t\t\t{flags}\n\t\t</bc>"
        )
    return rows[0::2], rows[1::2]


def hgo_three_field_block():
    """Uncoupled HGO block in a three-field domain, pulled along the fiber mean.

    A 2x2x2 hex8 block with the material of suite problem ma34, far smaller
    than ma34 itself. Symmetry planes on three faces, x displacement
    prescribed on the fourth to 20 percent stretch. The response is uniform.
    """
    grid = Grid((2, 2, 2), (1, 1, 1))
    sets, bcs = _block_bcs(grid)
    return deck(
        hgo_three_field_block.__doc__,
        [
            control(steps=10, dt=0.1),
            section("Material", material_xml(1, "m", "hgo_uncoupled")),
            section(
                "Mesh",
                grid.nodes_xml("Object1"),
                grid.elems_xml("Part1"),
                *sets,
                nodeset("x1", grid.face_nodes(0, 1)),
            ),
            section(
                "MeshDomains",
                '\t\t<SolidDomain name="Part1" mat="m" type="three-field-solid">\n'
                "\t\t\t<laugon>1</laugon>\n\t\t\t<atol>0.01</atol>\n"
                "\t\t\t<minaug>0</minaug>\n\t\t\t<maxaug>10</maxaug>\n"
                "\t\t</SolidDomain>",
            ),
            section(
                "Boundary",
                *bcs,
                '\t\t<bc name="pull" node_set="x1" type="prescribed displacement">\n'
                '\t\t\t<dof>x</dof>\n\t\t\t<value lc="1">0.2</value>\n'
                "\t\t\t<relative>0</relative>\n\t\t</bc>",
            ),
            section("LoadData", load_curve(1, [(0, 0), (1, 1)])),
            output(
                f'<element_data data="{F_VARS}" delim=","/>',
                f'<element_data data="{S_VARS};J" delim=","/>',
                '<node_data data="ux;uy;uz;Rx" delim=","/>',
            ),
        ],
    )


def damped_block():
    """Mass damping on a solid element.

    One neo-Hookean hex8 (E = 1, density = 1) with symmetry planes, a pressure
    of -0.05 (tension) stepped onto the top face at t = 0, dynamic analysis,
    mass damping C = 2. The top displacement overshoots and decays to the
    static value. The suite exercises mass damping only on truss elements
    (dy17); this is the same load on a solid.
    """
    grid = Grid((1, 1, 1), (1, 1, 1))
    sets, bcs = _block_bcs(grid)
    return deck(
        damped_block.__doc__,
        [
            control("DYNAMIC", steps=200, dt=0.1),
            section("Material", material_xml(1, "m", "neo_hookean")),
            section(
                "Mesh",
                grid.nodes_xml("Object1"),
                grid.elems_xml("Part1"),
                *sets,
                surface("top", grid.face_quads(2, 1)),
            ),
            section("MeshDomains", '\t\t<SolidDomain name="Part1" mat="m"/>'),
            section("Boundary", *bcs),
            section(
                "Loads",
                '\t\t<surface_load name="pull" surface="top" type="pressure">\n'
                '\t\t\t<pressure lc="1">-0.05</pressure>\n'
                "\t\t\t<symmetric_stiffness>1</symmetric_stiffness>\n"
                "\t\t\t<linear>0</linear>\n\t\t</surface_load>",
                '\t\t<body_load elem_set="Part1" type="mass damping">\n'
                "\t\t\t<C>2</C>\n\t\t</body_load>",
            ),
            section("LoadData", load_curve(1, [(0, 1), (20, 1)])),
            output(
                f'<node_data data="uz;vz" delim=",">{grid.nid(1, 1, 1)}</node_data>',
                f'<element_data data="{S_VARS}" delim=","/>',
            ),
        ],
    )


def modulus_step_block():
    """A material parameter on a load curve: the effective-modulus device law.

    One neo-Hookean hex8 under a constant tension of 0.05 on the top face.
    Young's modulus follows a load curve: 1 until t = 0.5, then 0.25, the
    step in modulus used for the frame in Zelonis et al. (arXiv 2507.06337).
    The stretch grows when the modulus drops.
    """
    grid = Grid((1, 1, 1), (1, 1, 1))
    sets, bcs = _block_bcs(grid)
    return deck(
        modulus_step_block.__doc__,
        [
            control(steps=10, dt=0.1),
            section(
                "Material",
                material_xml(1, "m", "neo_hookean", {"E": "@LC2@"}).replace(
                    "<E>@LC2@</E>", '<E lc="2">1</E>'
                ),
            ),
            section(
                "Mesh",
                grid.nodes_xml("Object1"),
                grid.elems_xml("Part1"),
                *sets,
                surface("top", grid.face_quads(2, 1)),
            ),
            section("MeshDomains", '\t\t<SolidDomain name="Part1" mat="m"/>'),
            section("Boundary", *bcs),
            section(
                "Loads",
                '\t\t<surface_load name="pull" surface="top" type="pressure">\n'
                '\t\t\t<pressure lc="1">-0.05</pressure>\n'
                "\t\t\t<symmetric_stiffness>1</symmetric_stiffness>\n"
                "\t\t\t<linear>0</linear>\n\t\t</surface_load>",
            ),
            section(
                "LoadData",
                load_curve(1, [(0, 0), (0.2, 1), (1, 1)]),
                load_curve(2, [(0, 1), (0.5, 1), (0.6, 0.25), (1, 0.25)]),
            ),
            output(
                f'<node_data data="ux;uy;uz" delim=",">{grid.nid(1, 1, 1)}</node_data>',
                f'<element_data data="{F_VARS}" delim=","/>',
                f'<element_data data="{S_VARS}" delim=","/>',
            ),
        ],
    )


def contact_potential_patch():
    """Two blocks brought together under the contact potential.

    The setup of suite problem co62 on a small generated mesh: a fixed lower
    block (3x3x1 hex8) and an upper block (2x2x1, so the surface meshes do not
    match) lowered onto it. Contact potential parameters are those of co62.
    Logged: the reaction force on the driven face and the top block's stress.
    Node ids are kept contiguous: a log record addressed by node set reports
    internal node numbers, which match the ids only when they are contiguous.
    Self-contact of a single folded surface is the next case to add.
    """
    lower = Grid((3, 3, 1), (3, 3, 1))
    upper = Grid(
        (2, 2, 1), (2, 2, 1), origin=(0.5, 0.5, 1.6), first_node=33, first_elem=10
    )
    return deck(
        contact_potential_patch.__doc__,
        [
            control(
                steps=20,
                dt=0.05,
                extra="\t\t\t<reform_each_time_step>1</reform_each_time_step>\n",
            ),
            section(
                "Material",
                material_xml(1, "bottom", "neo_hookean"),
                material_xml(2, "top", "neo_hookean"),
            ),
            section(
                "Mesh",
                lower.nodes_xml("Lower"),
                upper.nodes_xml("Upper"),
                lower.elems_xml("PartLower"),
                upper.elems_xml("PartUpper"),
                nodeset("lower_base", lower.face_nodes(2, 0)),
                nodeset("upper_top", upper.face_nodes(2, 1)),
                surface("primary", upper.face_quads(2, 0)),
                surface("secondary", lower.face_quads(2, 1)),
                '\t\t<SurfacePair name="pair">\n\t\t\t<primary>primary</primary>\n'
                "\t\t\t<secondary>secondary</secondary>\n\t\t</SurfacePair>",
            ),
            section(
                "MeshDomains",
                '\t\t<SolidDomain name="PartLower" mat="bottom"/>',
                '\t\t<SolidDomain name="PartUpper" mat="top"/>',
            ),
            section(
                "Boundary",
                '\t\t<bc name="fix" node_set="lower_base" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n\t\t</bc>",
                '\t\t<bc name="fixxy" node_set="upper_top" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>\n\t\t</bc>",
                '\t\t<bc name="push" node_set="upper_top" type="prescribed displacement">\n'
                '\t\t\t<dof>z</dof>\n\t\t\t<value lc="1">-0.5</value>\n'
                "\t\t\t<relative>0</relative>\n\t\t</bc>",
            ),
            section(
                "Contact",
                '\t\t<contact name="potential" surface_pair="pair" type="contact potential">\n'
                "\t\t\t<kc>1e-2</kc>\n\t\t\t<p>4</p>\n\t\t\t<R_in>0.3</R_in>\n"
                "\t\t\t<R_out>0.5</R_out>\n\t\t\t<R0_min>0</R0_min>\n"
                "\t\t\t<w_tol>0</w_tol>\n\t\t</contact>",
            ),
            section("LoadData", load_curve(1, [(0, 0), (1, 1)])),
            output(
                '<node_data data="uz;Rz" delim="," node_set="upper_top"/>',
                f'<element_data data="{S_VARS}" delim="," elem_set="PartUpper"/>',
            ),
        ],
    )


# --------------------------------------------------------------------------
# Shell and device cases
# --------------------------------------------------------------------------
THICKNESS = 0.396  # mm, leaflet thickness of the mitral model


class Sheet:
    """A structured sheet of quad4 shell elements, mapped into 3D."""

    def __init__(self, n, mapping):
        self.n = n
        nu, nv = n
        self.nodes = {
            self.nid(i, j): mapping(i / nu, j / nv)
            for j, i in itertools.product(range(nv + 1), range(nu + 1))
        }
        self.elems = {}
        for e, (j, i) in enumerate(itertools.product(range(nv), range(nu)), start=1):
            self.elems[e] = [self.nid(i, j), self.nid(i + 1, j),
                             self.nid(i + 1, j + 1), self.nid(i, j + 1)]  # fmt: skip

    def nid(self, i, j):
        return 1 + i + (self.n[0] + 1) * j

    def edge(self, pred):
        nu, nv = self.n
        return [
            self.nid(i, j) for j in range(nv + 1) for i in range(nu + 1) if pred(i, j)
        ]

    def mesh_xml(self, part):
        nodes = "\n".join(
            f'\t\t\t<node id="{i}">{x:.17g},{y:.17g},{z:.17g}</node>'
            for i, (x, y, z) in self.nodes.items()
        )
        elems = "\n".join(
            f'\t\t\t<elem id="{i}">{",".join(map(str, c))}</elem>'
            for i, c in self.elems.items()
        )
        return (
            f'\t\t<Nodes name="Object1">\n{nodes}\n\t\t</Nodes>\n'
            f'\t\t<Elements type="quad4" name="{part}">\n{elems}\n\t\t</Elements>'
        )

    def domain_xml(self, part, mat):
        """Shell domain and thickness data, written as in the mitral model."""
        domain = section(
            "MeshDomains",
            f'\t\t<ShellDomain name="{part}" mat="{mat}" type="elastic-shell">\n'
            "\t\t\t<shell_thickness>0</shell_thickness>\n"
            "\t\t\t<shell_normal_nodal>1</shell_normal_nodal>\n\t\t</ShellDomain>",
        )
        t = ",".join([f"{THICKNESS:g}"] * 4)
        rows = "\n".join(f'\t\t\t<e lid="{i}">{t}</e>' for i in self.elems)
        data = section(
            "MeshData",
            f'\t\t<ElementData type="shell thickness" elem_set="{part}">\n{rows}\n\t\t</ElementData>',
        )
        return domain, data


def _dynamic_control(steps, dt):
    """Implicit dynamics with the solver settings of the mitral model."""
    return control(
        "DYNAMIC", steps=steps, dt=dt,
        extra="\t\t\t<reform_each_time_step>1</reform_each_time_step>\n",
    )  # fmt: skip


def lee_sacks_shell_patch():
    """A Lee-Sacks shell patch under pressure, with the settings of the valve model.

    A flat 10 mm square of 6x6 quad4 elastic-shell elements, thickness
    0.396 mm, Lee-Sacks leaflet parameters, edges pinned. Pressure of 13.3 kPa
    (100 mmHg) ramped over 5 ms and held to 20 ms; implicit dynamics with mass
    damping C = 3000, as in the mitral model in this repository. The suite has
    no problem that puts this material on shells.
    """
    n = 6
    sheet = Sheet((n, n), lambda u, v: (10 * u, 10 * v, 0.0))
    edge = sheet.edge(lambda i, j: i in (0, n) or j in (0, n))
    center = sheet.nid(n // 2, n // 2)
    domain, data = sheet.domain_xml("Part1", "m")
    return deck(
        lee_sacks_shell_patch.__doc__,
        [
            _dynamic_control(steps=100, dt=2e-4),
            section("Material", material_xml(1, "m", "lee_sacks")),
            section(
                "Mesh",
                sheet.mesh_xml("Part1"),
                nodeset("edge", edge),
                surface("face", list(sheet.elems.values())),
            ),
            domain,
            data,
            section(
                "Boundary",
                '\t\t<bc name="pin" node_set="edge" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n\t\t</bc>",
            ),
            section(
                "Loads",
                '\t\t<surface_load name="p" surface="face" type="pressure">\n'
                '\t\t\t<pressure lc="1">-13.3</pressure>\n'
                "\t\t\t<symmetric_stiffness>1</symmetric_stiffness>\n"
                "\t\t\t<linear>0</linear>\n\t\t\t<shell_bottom>0</shell_bottom>\n"
                "\t\t</surface_load>",
                '\t\t<body_load elem_set="Part1" type="mass damping">\n'
                "\t\t\t<C>3000</C>\n\t\t</body_load>",
            ),
            section("LoadData", load_curve(1, [(0, 0), (0.005, 1), (0.02, 1)])),
            output(
                f'<node_data data="ux;uy;uz" delim=",">{center}</node_data>',
                f'<element_data data="{S_VARS}" delim=","/>',
            ),
        ],
    )


def shell_self_contact():
    """Self-contact of one shell surface under the contact potential.

    One strip of quad4 shells folded into a U: two flat flaps 2 mm apart
    joined by a half cylinder. The lower flap's free edge is fixed and the
    upper flap's free edge is lowered by 1.8 mm in 30 steps, so the flaps come
    within the potential's range (0.5 mm) over the last five steps. The surface pair names the same surface as
    primary and secondary, as the leaflet contact in the mitral model does,
    with that model's potential parameters. The suite's one contact potential
    problem (co62) has two separate solid bodies.
    """
    n_flap, n_fold, width = 6, 6, 3
    length, gap = 6.0, 1.0  # flap length, half the distance between flaps
    arc = math.pi * gap
    total = 2 * length + arc

    def mapping(u, v):
        s = u * total
        if s <= length:  # upper flap, from its free edge towards the fold
            return (length - s, 3.0 * v, gap)
        if s <= length + arc:
            a = (s - length) / gap
            return (-gap * math.sin(a), 3.0 * v, gap * math.cos(a))
        return (s - length - arc, 3.0 * v, -gap)

    # Uneven spacing along the strip so each region has whole elements.
    nu = 2 * n_flap + n_fold
    breaks = (
        [length * i / n_flap for i in range(n_flap)]
        + [length + arc * i / n_fold for i in range(n_fold)]
        + [length + arc + length * i / n_flap for i in range(n_flap + 1)]
    )
    sheet = Sheet((nu, width), lambda u, v: mapping(breaks[round(u * nu)] / total, v))
    upper_edge = sheet.edge(lambda i, j: i == 0)
    lower_edge = sheet.edge(lambda i, j: i == nu)
    domain, data = sheet.domain_xml("Part1", "m")
    return deck(
        shell_self_contact.__doc__,
        [
            control(
                steps=30,
                dt=1 / 30,
                extra="\t\t\t<reform_each_time_step>1</reform_each_time_step>\n",
            ),
            section("Material", material_xml(1, "m", "lee_sacks")),
            section(
                "Mesh",
                sheet.mesh_xml("Part1"),
                nodeset("upper_edge", upper_edge),
                nodeset("lower_edge", lower_edge),
                surface("leaflet", list(sheet.elems.values())),
                '\t\t<SurfacePair name="self">\n\t\t\t<primary>leaflet</primary>\n'
                "\t\t\t<secondary>leaflet</secondary>\n\t\t</SurfacePair>",
            ),
            domain,
            data,
            section(
                "Boundary",
                '\t\t<bc name="fix" node_set="lower_edge" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n\t\t</bc>",
                '\t\t<bc name="fixxy" node_set="upper_edge" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>0</z_dof>\n\t\t</bc>",
                '\t\t<bc name="close" node_set="upper_edge" type="prescribed displacement">\n'
                '\t\t\t<dof>z</dof>\n\t\t\t<value lc="1">-1.8</value>\n'
                "\t\t\t<relative>0</relative>\n\t\t</bc>",
            ),
            section(
                "Contact",
                '\t\t<contact name="self" surface_pair="self" type="contact potential">\n'
                "\t\t\t<kc>2</kc>\n\t\t\t<p>4</p>\n\t\t\t<R_in>0.2</R_in>\n"
                "\t\t\t<R_out>0.5</R_out>\n\t\t\t<R0_min>0</R0_min>\n"
                "\t\t\t<w_tol>0.6</w_tol>\n\t\t</contact>",
            ),
            section("LoadData", load_curve(1, [(0, 0), (1, 1)])),
            output(
                f'<node_data data="uz;Rz" delim=",">{",".join(map(str, upper_edge))}</node_data>',
                f'<node_data data="ux;uy;uz" delim=",">{sheet.nid(n_flap // 2, 1)},{sheet.nid(nu - n_flap // 2, 1)}</node_data>',
            ),
        ],
    )


def ring_in_tube():
    """A ring crimped by a tube and released with a stiffer modulus.

    The reduced form of the device simulations in Zelonis et al.
    (arXiv 2507.06337): a quarter of a neo-Hookean ring (radii 9 and 10 mm)
    inside a tube (inner radius 10.6 mm), plane strain, hex8 elements,
    contact potential between them, non-matching meshes. The tube is driven
    radially (prescribed deformation, x = F X with F = diag(a, a, 1)): it
    contracts to 85 percent by t = 1, holds while the ring's Young's modulus
    is raised from 0.25 to 1 by a load curve (t = 1 to 1.5), then opens to
    95 percent by t = 2.5. Every tube node is prescribed, so the tube acts as
    a moving wall. The quantity of interest is the ring's hoop force, read as
    the reaction on its symmetry plane, before and after the modulus step.
    """
    quarter = math.pi / 2

    def polar(r, t, z):
        return (r * math.cos(t), r * math.sin(t), z)

    ring = Grid((1, 12, 1), (1.0, quarter, 1.0), origin=(9.0, 0.0, 0.0), mapping=polar)
    tube = Grid((1, 9, 1), (1.0, quarter, 1.0), origin=(10.6, 0.0, 0.0),
                first_node=len(ring.nodes) + 1, first_elem=len(ring.elems) + 1,
                mapping=polar)  # fmt: skip
    ring_mat = material_xml(1, "ring", "neo_hookean", {"E": "@"}).replace(
        "<E>@</E>", '<E lc="2">1</E>'
    )
    return deck(
        ring_in_tube.__doc__,
        [
            control(
                steps=25,
                dt=0.1,
                extra="\t\t\t<reform_each_time_step>1</reform_each_time_step>\n",
            ),
            section("Material", ring_mat, material_xml(2, "tube", "neo_hookean")),
            section(
                "Mesh",
                ring.nodes_xml("Ring"),
                tube.nodes_xml("Tube"),
                ring.elems_xml("PartRing"),
                tube.elems_xml("PartTube"),
                nodeset("ring_all", list(ring.nodes)),
                nodeset("ring_y0", ring.face_nodes(1, 0)),
                nodeset("ring_x0", ring.face_nodes(1, 1)),
                nodeset("tube_all", list(tube.nodes)),
                surface("ring_outer", ring.face_quads(0, 1)),
                surface("tube_inner", tube.face_quads(0, 0)),
                '\t\t<SurfacePair name="pair">\n\t\t\t<primary>ring_outer</primary>\n'
                "\t\t\t<secondary>tube_inner</secondary>\n\t\t</SurfacePair>",
            ),
            section(
                "MeshDomains",
                '\t\t<SolidDomain name="PartRing" mat="ring"/>',
                '\t\t<SolidDomain name="PartTube" mat="tube"/>',
            ),
            section(
                "Boundary",
                '\t\t<bc name="plane" node_set="ring_all" type="zero displacement">\n'
                "\t\t\t<x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof>\n\t\t</bc>",
                # Prescribed to zero, not fixed: FEBio logs no reaction for a
                # fixed degree of freedom, and the hoop force is read here.
                '\t\t<bc name="symy" node_set="ring_y0" type="prescribed displacement">\n'
                "\t\t\t<dof>y</dof>\n\t\t\t<value>0</value>\n"
                "\t\t\t<relative>0</relative>\n\t\t</bc>",
                '\t\t<bc name="symx" node_set="ring_x0" type="zero displacement">\n'
                "\t\t\t<x_dof>1</x_dof><y_dof>0</y_dof><z_dof>0</z_dof>\n\t\t</bc>",
                '\t\t<bc name="crimp" node_set="tube_all" type="prescribed deformation">\n'
                '\t\t\t<scale lc="1">1</scale>\n'
                "\t\t\t<F>0,0,0,0,0,0,0,0,1</F>\n\t\t</bc>",
            ),
            section(
                "Contact",
                '\t\t<contact name="potential" surface_pair="pair" type="contact potential">\n'
                "\t\t\t<kc>1e-2</kc>\n\t\t\t<p>4</p>\n\t\t\t<R_in>0.3</R_in>\n"
                "\t\t\t<R_out>0.5</R_out>\n\t\t\t<R0_min>0</R0_min>\n"
                "\t\t\t<w_tol>0</w_tol>\n\t\t</contact>",
            ),
            section(
                "LoadData",
                # Scale s gives x = X + s (F - I) X = (1 - s) X in the plane.
                load_curve(1, [(0, 0), (1, 0.15), (1.5, 0.15), (2.5, 0.05)]),
                load_curve(2, [(0, 0.25), (1, 0.25), (1.5, 1), (2.5, 1)]),
            ),
            output(
                f'<node_data data="ux;uy" delim=",">{ring.nid(1, 0, 0)}</node_data>',
                '<node_data data="Ry" delim="," node_set="ring_y0"/>',
                f'<element_data data="{S_VARS}" delim="," elem_set="PartRing"/>',
            ),
        ],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    for mat, path in MATERIAL_POINT_CASES:
        cases[f"mp_{mat}_{path}"] = (material_point(mat, path), 0, [MATERIALS[mat][0]])
    cases["ch_chord_slack_tension"] = (chord(), 1, ["nonlinear spring"])
    cases["bk_hgo_three_field"] = (
        hgo_three_field_block(), 2, ["Holzapfel-Gasser-Ogden", "three-field-solid"])  # fmt: skip
    cases["bk_mass_damping"] = (damped_block(), 2, ["mass damping", "dynamic"])
    cases["bk_modulus_load_curve"] = (
        modulus_step_block(), 2, ["material load curve"])  # fmt: skip
    cases["bk_contact_potential_patch"] = (
        contact_potential_patch(), 2, ["contact potential"])  # fmt: skip

    cases["sh_lee_sacks_pressure"] = (
        lee_sacks_shell_patch(), 2, ["isotropic Lee-Sacks", "elastic-shell", "pressure", "mass damping"])  # fmt: skip
    cases["sh_self_contact_fold"] = (
        shell_self_contact(), 2, ["contact potential", "self-contact", "elastic-shell"])  # fmt: skip
    cases["bk_ring_in_tube"] = (
        ring_in_tube(), 3, ["contact potential", "material load curve", "prescribed deformation"])  # fmt: skip

    index = []
    for name, (text, tier, tags) in cases.items():
        (OUT / f"{name}.feb").write_text(text)
        index.append({"case": name, "tier": tier, "tags": tags})
    (OUT.parent / "cases.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(cases)} input files to {OUT}")


if __name__ == "__main__":
    main()
