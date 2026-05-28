from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.config import RunConfig
from autonqs.excited import ExcitedStateConfig
from autonqs.network import NetworkConfig
from autonqs.optim import OptimizerConfig
from autonqs.training import train_from_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ground-state checkpoint, then an excited-state penalty run.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs/examples/excited_states") / str(int(time.time()))
    ground_ckpt = str(run_dir / "ground.pt")
    net = NetworkConfig(hidden=16, pair_hidden=8, layers=1, hidden_density=1)
    opt = OptimizerConfig(name="natural", lr=2e-4)
    ground = train_from_run_config(
        RunConfig(
            molecule="h2",
            steps=args.steps,
            walkers=args.walkers,
            burn_in=1,
            mcmc_steps=1,
            device=args.device,
            checkpoint_path=ground_ckpt,
            checkpoint_every=1,
            network=net,
            optimizer=opt,
        )
    )
    ground.pop("model")
    excited = train_from_run_config(
        RunConfig(
            molecule="h2",
            steps=args.steps,
            walkers=args.walkers,
            burn_in=1,
            mcmc_steps=1,
            device=args.device,
            network=net,
            optimizer=opt,
            excited=ExcitedStateConfig(state_index=1, overlap_penalty=5.0, variance_penalty=0.01, previous_checkpoints=(ground_ckpt,)),
        )
    )
    excited.pop("model")
    print(json.dumps({"ground_checkpoint": ground_ckpt, "ground": ground, "excited": excited}, indent=2))


if __name__ == "__main__":
    main()
