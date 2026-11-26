"""Constitutive models behind a common material interface.

FEBio counterpart: FEBioMech materials.
Status (November 2026): draft, in pull request 2. See the development areas table in the README.

    model.py      the interface a constitutive model implements, and the registry
    compose.py    solid mixtures
    febio.py      from a FEBio <material> of the central model to a material here;
                  the one binding compile, the coverage report and the tests use
    status.py     failure codes written by kernels, shared with the elements
"""
