"""Contact search and enforcement for leaflets and devices.

Point-based contact: point sets, pair search, the Kamensky volume potential,
and the PhysicsTerm adapter. See docs/design/contact.md.

FEBio counterpart: FEBioMech contact interfaces.
Status (November 2026): draft, in pull request 1. See the development areas table in the README.
"""

from .points import (
    ContactPoints,
    boundary_triangles,
    node_points,
    quad_facet_points,
    segment_points,
    tri_facet_points,
)
from .potential import ContactPotential, PotentialParams, from_febio_block
from .search import PairSearch
from .enforcement import ContactPotentialTerm, ContactTangent
from .interface import ContactInterface, ContactInterfaceBase, ContactSet

__all__ = [
    "ContactPoints",
    "boundary_triangles",
    "node_points",
    "quad_facet_points",
    "segment_points",
    "tri_facet_points",
    "ContactPotential",
    "PotentialParams",
    "from_febio_block",
    "PairSearch",
    "ContactPotentialTerm",
    "ContactTangent",
    "ContactInterface",
    "ContactInterfaceBase",
    "ContactSet",
]
