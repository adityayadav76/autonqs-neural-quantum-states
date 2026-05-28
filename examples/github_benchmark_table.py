from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.benchmarks import get_benchmark, list_benchmarks
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmarks and emit a GitHub-ready Markdown table.")
    parser.add_argument("--benchmark", choices=["all", *list_benchmarks()], default="all")
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
    parser.add_argument("--output", default="runs/benchmarks/github_benchmark_results.json")
    args = parser.parse_args()

    names = list_benchmarks() if args.benchmark == "all" else [args.benchmark]
    cfg = TrainConfig(
        steps=args.steps,
        walkers=args.walkers,
        hidden=args.hidden,
        layers=2,
        hidden_density=args.hidden_density,
        orbital_reference=args.orbital_reference,
        backflow=not args.no_backflow,
        optimizer=args.optimizer,
        lr=args.lr,
        burn_in=50,
        mcmc_steps=10,
        device=args.device,
    )
    rows = []
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
        error = calculated - bench.reference_energy_hartree
        rows.append(
            {
                "system": name,
                "calculated_energy_hartree": calculated,
                "reference_energy_hartree": bench.reference_energy_hartree,
                "error_hartree": error,
                "abs_error_hartree": abs(error),
                "within_tolerance": abs(error) <= bench.tolerance_hartree,
                "tolerance_hartree": bench.tolerance_hartree,
                "best_checkpoint_energy_hartree": result["best_energy"],
                "block_mean_energy_hartree": result["analysis"]["energy_blocks"]["mean"],
                "block_stderr_hartree": result["analysis"]["energy_blocks"]["stderr"],
                "seconds": result["seconds"],
            }
        )
        print(f"finished {name}: final={calculated:.6f} ref={bench.reference_energy_hartree:.6f}", flush=True)

    payload = {
        "settings": {
            "steps": args.steps,
            "walkers": args.walkers,
            "hidden": args.hidden,
            "hidden_density": args.hidden_density,
            "optimizer": args.optimizer,
            "device": args.device,
            "energy_for_table": args.energy_metric,
            "energy_metric": args.energy_metric,
        },
        "rows": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n```markdown")
    print("| System | Calculated Energy (Ha) | Reference Energy (Ha) | Error (Ha) | Abs. Error (Ha) |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['system']} | {row['calculated_energy_hartree']:.6f} | "
            f"{row['reference_energy_hartree']:.6f} | {row['error_hartree']:.6f} | "
            f"{row['abs_error_hartree']:.6f} |"
        )
    print("```")


if __name__ == "__main__":
    main()
