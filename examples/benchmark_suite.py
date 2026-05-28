from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.benchmarks import get_benchmark, list_benchmarks
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoNQS benchmark smoke/convergence jobs.")
    parser.add_argument("--benchmark", choices=["all", *list_benchmarks()], default="h2")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden-density", type=int, default=4)
    parser.add_argument("--orbital-reference", action="store_true")
    parser.add_argument("--no-backflow", action="store_true")
    parser.add_argument("--optimizer", default="sr", choices=["adam", "natural", "sr", "diag-natural", "kfac"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--energy-metric", default="final", choices=["block-mean", "final", "best"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in list_benchmarks():
            bench = get_benchmark(name)
            print(f"{name:6s} reference={bench.reference_energy_hartree: .6f} Ha tolerance={bench.tolerance_hartree}")
        return

    names = list_benchmarks() if args.benchmark == "all" else [args.benchmark]
    cfg = TrainConfig(
        steps=args.steps,
        walkers=args.walkers,
        hidden=args.hidden,
        hidden_density=args.hidden_density,
        orbital_reference=args.orbital_reference,
        backflow=not args.no_backflow,
        layers=2,
        optimizer=args.optimizer,
        lr=args.lr,
        device=args.device,
    )
    for name in names:
        bench = get_benchmark(name)
        result = train(bench.molecule, cfg)
        result.pop("model")
        if args.energy_metric == "block-mean":
            calculated = result["analysis"]["energy_blocks"]["mean"]
        elif args.energy_metric == "best":
            calculated = result["best_energy"]
        else:
            calculated = result["final_energy"]
        result["reference_energy_hartree"] = bench.reference_energy_hartree
        result["calculated_energy_hartree"] = calculated
        result["energy_metric"] = args.energy_metric
        result["error_hartree"] = calculated - bench.reference_energy_hartree
        result["abs_error_hartree"] = abs(result["error_hartree"])
        result["reference_source"] = bench.source
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
