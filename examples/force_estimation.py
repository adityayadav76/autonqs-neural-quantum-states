from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.molecules import get_molecule
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train briefly and report nuclear force estimates.")
    parser.add_argument("--molecule", default="h2")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(
        get_molecule(args.molecule),
        TrainConfig(
            steps=args.steps,
            walkers=args.walkers,
            hidden=16,
            layers=1,
            hidden_density=1,
            burn_in=1,
            mcmc_steps=1,
            estimate_forces=True,
            device=args.device,
        ),
    )
    result.pop("model")
    print(json.dumps(result["forces"], indent=2))


if __name__ == "__main__":
    main()
