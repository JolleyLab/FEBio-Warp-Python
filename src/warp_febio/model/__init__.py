"""The central model abstraction (host side).

FEBio counterpart: FEModel and FEMesh.
Status (November 2026): draft. See docs/design/model.md.
"""

from .model import (
    ELEMENT_NODES,
    FACET_NODES,
    Domain,
    ElementBlock,
    Issue,
    LoadController,
    Material,
    Mesh,
    Model,
    Record,
    Step,
    Surface,
)

__all__ = [
    "ELEMENT_NODES",
    "FACET_NODES",
    "Domain",
    "ElementBlock",
    "Issue",
    "LoadController",
    "Material",
    "Mesh",
    "Model",
    "Record",
    "Step",
    "Surface",
]
