"""Import check for the package and its component namespaces."""

import importlib
import unittest

SUBPACKAGES = (
    "analysis",
    "assembly",
    "constraints",
    "contact",
    "discretization",
    "elements",
    "io",
    "loads",
    "materials",
    "mesh",
    "sensitivity",
    "solvers.linear",
    "solvers.nonlinear",
)


class PackageImportTest(unittest.TestCase):
    def test_subpackages_import(self):
        for name in SUBPACKAGES:
            with self.subTest(subpackage=name):
                importlib.import_module(f"warp_febio.{name}")


if __name__ == "__main__":
    unittest.main()
