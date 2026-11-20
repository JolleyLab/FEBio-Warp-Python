# FEBio Valve Pressurization Reference

This directory contains the reference FEBio model and associated validation
outputs for CardioFEM Version 1.

## Reference Policy

The FEBio model defines the target physical problem.

The reference model must not be modified merely to improve agreement with the
Warp/CardioFEM implementation.

Do not change:

- geometry
- mesh
- material parameters
- pressure
- chord properties
- boundary conditions
- contact parameters
- load curves
- solver settings

unless the change corrects a documented interpretation or unit error, or is
explicitly approved as a new reference case.

## Validation Principle

If CardioFEM and FEBio disagree, investigate the cause in the implementation,
formulation, solver, precision, or post-processing before changing the
reference model.

Parameter fitting to force agreement is not validation.

## Reference Files

Planned contents:

- target `.feb` model
- FEBio solver log
- nodal coordinates
- connectivity
- displacement field
- reaction forces
- stress / strain outputs
- chord forces
- contact outputs
- load / time information

## Provenance

Record:

- FEBio version
- model creation date
- units
- run command / analysis settings
- output extraction method

before using this model for final validation.
