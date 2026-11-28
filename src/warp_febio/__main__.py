"""Command line: inspect or run a FEBio input file.

    python -m warp_febio inspect model.feb
    python -m warp_febio run model.feb [-o model.xplt] [--vtkhdf model.vtkhdf]

`inspect` prints the coverage report: every feature of the file and whether
FEBio-Warp implements it. `run` refuses a file that is not yet runnable and
prints the same report. Both go through the run path (analysis/steps.py), so
the report covers the loads and element types the run path can build.
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="warp_febio")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        p = sub.add_parser(name)
        p.add_argument("input")
        if name == "run":
            p.add_argument(
                "-o",
                "--output",
                help="plot file to write (default: input name with .xplt)",
            )
            p.add_argument("--vtkhdf", help="also write results as VTKHDF")
            p.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from .io import read_feb, write_vtkhdf, write_xplt

    model = read_feb(args.input)
    if args.command == "inspect":
        # Importing the run path imports Warp. The model and the file layers
        # above do not need it.
        from .analysis.steps import inspect_model

        report = inspect_model(model)
        print(report.table())
        return 0 if report.runnable else 1
    from .analysis.steps import build_system, run
    from .coverage import CoverageError

    try:
        system = build_system(model, device=args.device)
    except CoverageError as err:
        print(err.report.table())
        return 1
    results = run(system, log=print)
    out = args.output or str(Path(args.input).with_suffix(".xplt"))
    write_xplt(model, results, out)
    print(f"wrote {out}")
    if args.vtkhdf:
        write_vtkhdf(model, results, args.vtkhdf)
        print(f"wrote {args.vtkhdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
