"""Coverage report: which features of a model FEBio-Warp can run.

Every feature in a Model is looked up by its FEBio name before any device
array is allocated. A model with a feature that is not yet supported does not
run; the report lists what is missing. Nothing is skipped silently.

Status values
    implemented         used as FEBio uses it
    no effect here      a setting FEBio-Warp reads but that does not change the
                        converged solution of this draft (the note says why)
    not written         an output request that is not produced; the solution is
                        unaffected
    not yet supported   the model cannot run until this is added
"""

from dataclasses import dataclass, field

IMPLEMENTED = "implemented"
NO_EFFECT = "no effect here"
NOT_WRITTEN = "not written"
NOT_SUPPORTED = "not yet supported"


@dataclass
class Entry:
    category: str
    febio_name: str
    status: str
    where: str = ""
    note: str = ""


@dataclass
class CoverageReport:
    entries: list = field(default_factory=list)

    def add(self, category, febio_name, status, where="", note=""):
        self.entries.append(Entry(category, str(febio_name), status, where, note))

    @property
    def runnable(self) -> bool:
        return not any(e.status == NOT_SUPPORTED for e in self.entries)

    def missing(self) -> list:
        return [e for e in self.entries if e.status == NOT_SUPPORTED]

    def table(self) -> str:
        rows = [("Category", "FEBio name", "Status", "Where", "Note")] + [
            (e.category, e.febio_name, e.status, e.where, e.note) for e in self.entries
        ]
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        lines = []
        for k, r in enumerate(rows):
            lines.append("  ".join(r[i].ljust(widths[i]) for i in range(4)) + ("  " + r[4] if r[4] else ""))
            if k == 0:
                lines.append("  ".join("-" * w for w in widths))
        verdict = "runnable" if self.runnable else "not yet runnable"
        return "\n".join(lines) + f"\n\nVerdict: {verdict}"


class CoverageError(RuntimeError):
    """Raised by compile when the model needs a feature that is not supported."""

    def __init__(self, report: CoverageReport):
        self.report = report
        missing = ", ".join(f"{e.category} {e.febio_name!r}" for e in report.missing())
        super().__init__(f"model is not yet runnable in FEBio-Warp: {missing}\n\n{report.table()}")


# Control settings and what this draft does with them. Anything not listed is
# reported as not yet supported, so an unfamiliar setting cannot pass unseen.
# FEBio definitions: FEAnalysis, FENewtonSolver, FESolidSolver2
# (https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEAnalysis.cpp,
#  https://github.com/febiosoftware/FEBio/blob/develop/FECore/FENewtonSolver.cpp,
#  https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESolidSolver2.cpp).
CONTROL_IMPLEMENTED = {"analysis", "time_steps", "step_size"}
CONTROL_NO_EFFECT = {
    "plot_zero_state": "the initial state is always written, as FEBio does with plot_range 0,-1",
    "output_level": "no log file is written",
    "plot_stride": "all converged steps are kept",
    "output_stride": "no log file is written",
    "adaptor_re_solve": "no mesh adaptor",
}
SOLVER_IMPLEMENTED = {
    "dtol",
    "etol",
    "rtol",
    "min_residual",
    "max_refs",
    "lstol",
    "lsmin",
    "lsiter",
    "ls_check_jacobians",
}
SOLVER_NO_EFFECT = {
    "symmetric_stiffness": "linear solver settings; this draft uses Warp iterative solvers or a dense host solve",
    "equation_scheme": "linear solver settings; this draft uses Warp iterative solvers or a dense host solve",
    "equation_order": "linear solver settings; this draft uses Warp iterative solvers or a dense host solve",
    "optimize_bw": "linear solver settings; this draft uses Warp iterative solvers or a dense host solve",
    "check_zero_diagonal": "not checked",
    "zero_diagonal_tol": "not checked",
    "force_partition": "single partition",
    "reform_each_time_step": "the stiffness is reformed every iteration",
    "reform_augment": "no augmentations",
    "diverge_reform": "the stiffness is reformed every iteration",
    "max_residual": "not checked",
    "rhoi": "static analysis",
    "alpha": "static analysis",
    "beta": "static analysis",
    "gamma": "static analysis",
    "logSolve": "not used",
    "arc_length": "0 only",
    "arc_length_scale": "arc length is off",
    "qn_method": "full Newton every iteration; same converged solution within the tolerances, iteration counts can differ",
}
