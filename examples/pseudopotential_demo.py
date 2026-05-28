from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.config import RunConfig
from autonqs.network import NetworkConfig
from autonqs.optim import OptimizerConfig
from autonqs.training import train_from_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local pseudopotential example.")
    parser.add_argument("--molecule", choices=["h2o", "ch2s"], default="h2o")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pseudos = {
        "h2o": ("BFD-O", "BFD-H", "BFD-H"),
        "ch2s": ("BFD-C", "BFD-S", "BFD-H", "BFD-H"),
    }[args.molecule]
    cfg = RunConfig(
        molecule=args.molecule,
        steps=args.steps,
        walkers=args.walkers,
        burn_in=1,
        mcmc_steps=1,
        device=args.device,
        pseudopotentials=pseudos,
        network=NetworkConfig(hidden=16, pair_hidden=8, layers=1, hidden_density=1),
        optimizer=OptimizerConfig(name="adam", lr=2e-4),
    )
    result = train_from_run_config(cfg)
    result.pop("model")
    print(json.dumps({"pseudopotentials": pseudos, "result": result}, indent=2))


if __name__ == "__main__":
    main()
