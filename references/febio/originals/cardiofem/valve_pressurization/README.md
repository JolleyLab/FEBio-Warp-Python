

`mitral_p2prolapse.feb` is the open mitral input (FEBio model repository, file
3572): 7,200 quad4 shells, 100 chord springs in 8 discrete sets, a pressure
written inside the step, mass damping, leaflet self-contact by the contact
potential, dynamic with `rhoi` 0.

Status in FEBio-Warp, November 28, 2026. The coverage report accepts the
input. With its contact removed it runs all 200 steps. With contact, the
first template stops converging as the leaflets close; what that needs (a line
search and a step limit first) is in `docs/design/contact_in_run.md`. The FEBio
run stored in `febio_results/` (FEBio 4.9.0) itself ends with an error
termination at time 0.0135 of 0.02, so a comparison with it is possible up to
that time only. Whether a current FEBio build completes this input is an open
question for the FEBio team.
