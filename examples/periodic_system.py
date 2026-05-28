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
    parser = argparse.ArgumentParser(description="Run a periodic minimum-image supercell example.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--cell", type=float, default=8.0, help="Cubic cell length in Angstrom.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = RunConfig(
        molecule="h2",
        steps=args.steps,
        walkers=args.walkers,
        burn_in=1,
        mcmc_steps=1,
        device=args.device,
        periodic_cell=((args.cell, 0.0, 0.0), (0.0, args.cell, 0.0), (0.0, 0.0, args.cell)),
        network=NetworkConfig(hidden=16, pair_hidden=8, layers=1, hidden_density=1),
        optimizer=OptimizerConfig(name="adam", lr=2e-4),
    )
    result = train_from_run_config(cfg)
    result.pop("model")
    print(json.dumps({"periodic_cell_angstrom": cfg.periodic_cell, "result": result}, indent=2))


if __name__ == "__main__":
    main()
