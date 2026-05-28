"""Run AutoNQS ground-state demos for the 20 requested molecules.

Large transition-metal and biological complexes use active-electron model specs
so the same PyTorch VMC machinery remains runnable on workstation GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.molecules import get_molecule, list_molecules
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=["all", *list_molecules()], default="h2")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--hidden-density", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, name in enumerate(list_molecules(), 1):
            mol = get_molecule(name)
            print(f"{i:02d}. {name:16s} electrons={mol.electron_count:3d} {mol.description}")
        return

    names = list_molecules() if args.molecule == "all" else [args.molecule]
    cfg = TrainConfig(
        steps=args.steps,
        walkers=args.walkers,
        hidden=args.hidden,
        layers=2,
        hidden_density=args.hidden_density,
        device=args.device,
    )
    results = []
    for name in names:
        result = train(get_molecule(name), cfg)
        result.pop("model")
        results.append(result)
        print(json.dumps(result, indent=2))
    if len(results) > 1:
        print(json.dumps({"ran": len(results), "mean_steps_per_second": sum(r["steps_per_second"] for r in results) / len(results)}, indent=2))


if __name__ == "__main__":
    main()
