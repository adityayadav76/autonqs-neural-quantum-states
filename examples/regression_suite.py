from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.regression import RegressionCase, default_smoke_cases, reference_table, run_regression_suite, validate_reference_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic regression checks against known benchmark systems.")
    parser.add_argument("--case", action="append", choices=["h", "he", "lih", "be", "h2", "n2", "h2o"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--references", action="store_true", help="Print reference table and exit.")
    args = parser.parse_args()
    validate_reference_table()
    if args.references:
        print(json.dumps(reference_table(), indent=2))
        return
    if args.case:
        cases = [RegressionCase(name, steps=args.steps, walkers=args.walkers) for name in args.case]
    else:
        cases = default_smoke_cases()
    result = run_regression_suite(cases, device=args.device)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
