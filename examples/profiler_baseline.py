from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.profiler import compare_profile, profile_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or compare autonqs profiler baselines.")
    parser.add_argument("--molecule", default="h2")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--walkers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/profiler/h2_profile.json")
    parser.add_argument("--baseline", default="", help="Existing profile JSON to compare against.")
    parser.add_argument("--max-slowdown", type=float, default=1.25)
    args = parser.parse_args()
    current = profile_training(args.molecule, args.steps, args.walkers, args.device, args.output)
    payload = {"current": current}
    exit_code = 0
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        comparison = compare_profile(current, baseline, args.max_slowdown)
        payload["comparison"] = comparison
        exit_code = 0 if comparison["passed"] else 1
    print(json.dumps(payload, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
