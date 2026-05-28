from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.molecules import get_molecule
from autonqs.training import TrainConfig, train


def _run(steps: int, checkpoint: str, log_dir: str, device: str) -> dict:
    result = train(
        get_molecule("h2"),
        TrainConfig(
            steps=steps,
            walkers=8,
            hidden=16,
            layers=1,
            hidden_density=1,
            burn_in=1,
            mcmc_steps=1,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            log_dir=log_dir,
            log_every=1,
            device=device,
        ),
    )
    result.pop("model")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate checkpoint creation and resume.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs/examples/checkpoint_resume") / str(int(time.time()))
    checkpoint = str(run_dir / "checkpoint.pt")
    first = _run(2, checkpoint, str(run_dir), args.device)
    resumed = _run(4, checkpoint, str(run_dir), args.device)
    print(json.dumps({"checkpoint": checkpoint, "first_steps": len(first["history"]), "resumed_final_step": resumed["history"][-1]["step"], "resumed": resumed}, indent=2))


if __name__ == "__main__":
    main()
