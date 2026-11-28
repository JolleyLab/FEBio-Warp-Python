

`mitral_p2prolapse.feb` is the open mitral input (FEBio model repository, file
3572): 7,200 quad4 shells, 100 chord springs in 8 discrete sets, a pressure
written inside the step, mass damping, leaflet self-contact by the contact
potential, dynamic with `rhoi` 0.

Status in FEBio-Warp, November 28, 2026. The input runs to the end, through
leaflet closure, with FEBio's line search and FEBio's default for excluding
neighboring elements from contact. `examples/valve/mitral_closure.py` runs it
and compares it with the FEBio results kept here: contact starts at the same
time, the closed contact area agrees to one percent, and the first principal
stress of the leaflet elements to half a percent in the median
(`docs/design/contact_in_run.md`). The FEBio
run stored in `febio_results/` (FEBio 4.9.0) itself ends with an error
termination at time 0.0135 of 0.02, so a comparison with it is possible up to
that time only. Whether a current FEBio build completes this input is an open
question for the FEBio team.
