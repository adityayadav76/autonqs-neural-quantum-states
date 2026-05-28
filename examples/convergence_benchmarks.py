from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.regression import RegressionCase, run_regression_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configurable convergence benchmarks with pass/fail energy thresholds.")
    parser.add_argument("--case", action="append", choices=["h", "he", "lih", "be", "h2", "n2", "h2o"])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden-density", type=int, default=2)
    parser.add_argument("--optimizer", default="natural", choices=["adam", "natural", "sr", "diag-natural", "kfac"])
    parser.add_argument("--max-error", type=float, default=None, help="Override max absolute energy error in Hartree.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    names = args.case or ["h2"]
    cases = [
        RegressionCase(
            name,
            steps=args.steps,
            walkers=args.walkers,
            hidden=args.hidden,
            hidden_density=args.hidden_density,
            optimizer=args.optimizer,
            max_abs_error_hartree=args.max_error,
        )
        for name in names
    ]
    result = run_regression_suite(cases, device=args.device)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
