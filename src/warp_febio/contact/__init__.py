"""Point-based contact: point sets, pair search, the Kamensky volume potential,
and the PhysicsTerm adapter. See docs/CONTACT.md."""

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
