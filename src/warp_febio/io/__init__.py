"""Reading FEBio input and reading and writing results.

FEBio counterpart: FEBioXML and FEBioPlot.
Status (November 2026): draft. See docs/design/model.md.

    read_feb, write_feb        FEBio input files, spec 4.0          (feb.py)
    read_xplt, write_xplt      FEBio plot files, version 0x0035     (xplt.py)
    write_vtkhdf               results as VTKHDF, double precision  (vtkhdf.py)
    vtk_adapter                Model and results as VTK data, for 3D Slicer

The first two need only NumPy and the standard library.
"""

from .feb import FebFormatError, read_feb, read_feb_string, write_feb
from .xplt import PlotFile, read_xplt, write_xplt


def write_vtkhdf(model, results, path):
    from .vtkhdf import write_vtkhdf as _write

    _write(model, results, path)


__all__ = [
    "FebFormatError",
    "PlotFile",
    "read_feb",
    "read_feb_string",
    "read_xplt",
    "write_feb",
    "write_vtkhdf",
    "write_xplt",
]
