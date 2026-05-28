from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.analysis import compare_reference
from autonqs.benchmarks import get_benchmark, list_benchmarks
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a short VMC result against a built-in reference.")
    parser.add_argument("--benchmark", choices=list_benchmarks(), default="h2")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    bench = get_benchmark(args.benchmark)
    result = train(
        bench.molecule,
        TrainConfig(steps=args.steps, walkers=args.walkers, hidden=16, layers=1, hidden_density=1, burn_in=1, mcmc_steps=1, device=args.device),
    )
    result.pop("model")
    print(json.dumps({"result_energy": result["analysis"]["energy_blocks"]["mean"], "comparison": compare_reference(args.benchmark, result["analysis"]["energy_blocks"]["mean"])}, indent=2))


if __name__ == "__main__":
    main()
