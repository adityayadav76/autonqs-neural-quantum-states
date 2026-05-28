from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.config import RunConfig
from autonqs.distributed import DistributedConfig
from autonqs.network import NetworkConfig
from autonqs.optim import OptimizerConfig
from autonqs.training import train_from_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or print a torchrun distributed autonqs example.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    command = "conda run -n pytorch torchrun --standalone --nproc_per_node=1 examples\\distributed_training.py"
    if args.print_command:
        print(command)
        return
    cfg = RunConfig(
        molecule="h2",
        steps=args.steps,
        walkers=args.walkers,
        burn_in=1,
        mcmc_steps=1,
        device=args.device,
        distributed=DistributedConfig(
            enabled="RANK" in os.environ,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        ),
        network=NetworkConfig(hidden=16, pair_hidden=8, layers=1, hidden_density=1),
        optimizer=OptimizerConfig(name="natural", lr=2e-4),
    )
    result = train_from_run_config(cfg)
    result.pop("model")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
