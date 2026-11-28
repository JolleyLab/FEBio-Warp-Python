# Where this code differs from FEBio

For Steve Maas and Jeff Weiss. November 28, 2026. Nothing here is settled.

**The rule followed.** FEBio decides the inputs, the formulations, the names, the sign conventions, the defaults and the outputs. The internals are arranged for Warp: flat arrays on the device, one kernel per call, a fixed block pattern, and parameters as device arrays that carry gradients. Where the result of a run, or the path Newton takes to it, differs from FEBio's, it is listed here with the reason and, where there is one, the setting that gives FEBio's behavior back.

**Scope.** Trusses and springs, nodal loads, dynamics and mass damping, the linear solve, and the first template of contact in a run: the work of November 27 and 28. Differences in the material layer, the model and input and output are in their own notes ([materials.md](materials.md), [model.md](model.md)).

## 1. Differences that change a result or the path to it

| # | Area | FEBio | Here | Why | To get FEBio's behavior | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Mass damping, Newton tangent | Stiffness C/dt per unit mass, which is not the derivative of the damping force, C gamma/(beta dt). The shell domain adds it with the opposite sign of the solid and truss domains | The derivative, for every domain | One tangent for Newton and for the adjoint. Same solution in far fewer iterations: 325 against 1400 on the shell reference case, 416 against 902 on the solid one | `System.damping_tangent = "febio"`: FEBio's form per domain. It reproduces FEBio's iteration counts (902 of 902; 1400 against 1397) | D4, [dynamics.md](dynamics.md) |
| 2 | Quasi-Newton and line search | BFGS with `max_ups` updates; a line search (`lstol`, `lsmin`, `lsiter`) | Full Newton every iteration, no line search | Not a choice: neither is built yet. Where FEBio runs full Newton and its line search stays at 1, iteration counts match (suite `dy16`, `dy07`, `bt01`, `di01`, `di02`). Where its line search acts they diverge (suite `sh58` from step 15), and with them the automatic step sizes | None yet | L6, [linear_solve.md](linear_solve.md) |
| 3 | `tangent_scale` on `isotropic Lee-Sacks` | Multiplies `c0` in the tangent only, to make Newton more robust (10 in the open mitral input) | Not applied: the exact tangent is used, and the coverage report says so | It does not change what Newton converges to. It may matter for robustness at leaflet closure, which is not yet reached | None yet | T14, [materials.md](materials.md) |
| 4 | Slope of a spring's point curve | A central difference (`PointCurve::derive`) | The exact slope of the active segment | An exact tangent, for Newton and for gradients. The two agree to about 1e-7 away from a knot | None | [chordae.md](chordae.md) |
| 5 | Contact potential, `w_tol` | The normal test is applied per element pair | Per point pair (the contact draft's search) | Not a choice: the element-pair form is not built. On the self-contact reference case the two give the same result to 4e-11 | None yet | [contact_in_run.md](contact_in_run.md) |
| 6 | Contact potential, step rule | None of this kind | A step is rejected and cut if a node moves farther than `R_out` in one iterate | Beyond `R_out` a point can pass through a surface unseen by the potential (the contact note) | Not needed where steps are small | [contact_in_run.md](contact_in_run.md) |
| 7 | Linear solver | Pardiso, skyline, FGMRES and others, chosen in the input | Warp's CG, BiCGSTAB or GMRES on the device; a dense or sparse LU on the host; optionally NVIDIA cuDSS on the device. FEBio's linear solver settings are reported as having no effect | What runs on a GPU. The methods agree with each other to 3e-13 on the shell patch | Not applicable | L2, [linear_solve.md](linear_solve.md) |
| 8 | A truss of zero length | Divides by the length | Writes a status code; the step is rejected | A kernel cannot throw | Not applicable | [chordae.md](chordae.md) |

## 2. Additions that FEBio does not have

Each is marked in the code and, where it can appear in an input, in the coverage report ("not a FEBio material: FEBio-Warp only").

| # | Addition | What it is | Relation to FEBio | Decision |
| --- | --- | --- | --- | --- |
| 9 | Rest length as a parameter array, trusses and springs | The rest length of each element is a device array that defaults to the distance between the reference nodes | The default is FEBio's value, so forward results are unchanged. Changing it makes chord length a design variable without moving nodes | C3, [chordae.md](chordae.md) |
| 10 | `tension-only linear truss` | FEBio's `linear truss` with no stress in compression, active at a stretch of exactly 1 | Names, parameters and the rule at the slack point follow `tension-only linear spring`. Equal to `linear truss` when every chord is in tension | C1 |
| 11 | `smooth tension-only linear spring` | FEBio's `tension-only linear spring` with the corner rounded over a band `smoothing_length` around the slack length | Equal to FEBio's law outside the band and for a width of zero. Inside it the force differs by at most E w / 4. For optimization, where the corner makes gradients one-sided | C9 |
| 12 | Axial force of a truss as an output | The chord tension per element | FEBio writes `stress` and `truss stretch` and has no axial force variable | C8 |
| 13 | Chord force on a node set | The resultant force the chords exert on a set of nodes (a papillary tip), as a vector and as an objective | The sum FEBio's `discrete element force` gives over the springs at those nodes | [chordae.md](chordae.md) |
| 14 | Gradients | Adjoint gradients through one equilibrium solve or one dynamic step | FEBio has no adjoint. Checked against central differences of this code | [sensitivity.md](sensitivity.md) |

## 3. Same result, different arrangement

These do not change a result. They are listed so that a reader of the code knows where the structure departs from FEBio's and why.

| # | Area | FEBio | Here | Why |
| --- | --- | --- | --- | --- |
| 15 | Velocity and acceleration | On the nodes | Arrays of the State, one 3-vector per entry, updated by one kernel | A step is a function from one State to the next, which an adjoint over steps replays (D2) |
| 16 | Mass | Assembled into the global matrix by each domain | One number per stiffness block, computed once; the inertial force and the mass in the tangent go through the blocks that are already there | No second matrix pattern. Consistent mass, as FEBio (D1) |
| 17 | Mass damping | A body force integrated by each domain's element code | C times the mass blocks of the part times the velocity | The same integral, with no element code of its own. Limited to a whole part for now |
| 18 | Springs | One domain per `discrete` entry | One term per spring law, holding every discrete set with that law; sets keep their FEBio names | One launch for the chordae of a valve (C7) |
| 19 | Rest length | From the reference node positions, with the code that computes the current length | Computed on the device by a kernel with the arithmetic of the update kernels | So that an undeformed element has a stretch of exactly 1 on every device, as in FEBio. From NumPy on the host, one tension-only element in twenty started slack on CUDA |
| 20 | Contact tangent | Added to the global matrix; the matrix profile is rebuilt with the pairs | A matrix of its own, added to the structural matrix before the solve | The structural pattern and its map stay fixed for the life of the system |
| 21 | Loads written inside a step | Activated when the step starts | With one step, built as loads of the model | The same for one step. Activation by step is reserved for several |
| 22 | A failed integration point | Throws | A status code per point, read once per update | A kernel cannot throw |
| 23 | Mass and damping coefficients | Evaluated from the material and geometry whenever a matrix is built | Computed once on the host when the system is built | The mass is constant. It does not follow later changes of density or truss area, and carries no gradient yet |
| 24 | `discrete` entries | `type="spring"` (or none) and `type="wire"` | Only springs; a `wire` entry, or settings inside `discrete`, is refused by the coverage report | Not built |

## 4. Checked and the same as FEBio

Points a reviewer might expect to differ, and that were checked against the source at `e4df3f8` and against runs.

| Point | Check |
| --- | --- |
| Time integration parameters for `rhoi` -1, in [0, 1], and otherwise; the velocity and acceleration update; the start of a step | `FESolidSolver2::Init`, `PrepStep`, `UpdateKinematics`; `bk_mass_damping` to 6e-14 over 200 steps |
| Initial accelerations from M a = F when a load acts at time zero, for solids and for both faces of a shell | `InitAccelerations`; `bk_mass_damping`, and step 1 of suite `sh58` |
| Consistent mass of hex8, quad4 shell and truss; the inertial force on alpha_m a + (1 - alpha_m) a_n; the mass scaled by alpha_m / (beta dt^2) | The domains' `MassMatrix` and `InertialForces`; suite `dy16` and `dy07` reproduce FEBio's iteration counts exactly |
| Nodal reaction forces: from the internal forces at free components; at prescribed components also the inertial force and mass damping, which FEBio's domains assemble element by element; zero at fixed components. Contact forces are not counted (the self-contact reference agrees without them). Whether surface loads are counted at prescribed components is not yet checked | `FESolidSolver2::Residual`, `FEResidualVector::Assemble` |
| Default density of an elastic material: 1 | `FESolidMaterial` |
| Truss and spring force and stiffness; `tension-only linear spring` active at exactly the slack length; the guard for a spring of zero length | [chordae.md](chordae.md), section 4 |
| Sign of the residual, external minus internal, for every term including contact | The run path note |

## 5. Questions these raise

1. Is the sign of the body force stiffness in `FEElasticShellDomain::ElementBodyForceStiffness` intended? It is opposite to `FEElasticSolidDomain::BodyForceStiffness` and `FELinearTrussDomain::BodyForceStiffness`, and with mass damping on shells it roughly doubles the iteration count (row 1).
2. Would the derivative of the mass damping force, C gamma/(beta dt), be welcome as FEBio's stiffness for it?
3. Would a tension-only truss, and an axial force plot variable for trusses, be of use in FEBio (rows 10 and 12)?
4. Is the per-element-pair form of `w_tol` essential to the contact potential's results on valve models, or a cost saving (row 5)?
