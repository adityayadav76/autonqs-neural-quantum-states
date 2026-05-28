from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.analysis import analyze_history, load_metrics_jsonl
from autonqs.molecules import get_molecule
from autonqs.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Create metric logs and analyze them.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs/examples/logging_analysis") / str(int(time.time()))
    result = train(
        get_molecule("h2"),
        TrainConfig(
            steps=args.steps,
            walkers=8,
            hidden=16,
            layers=1,
            hidden_density=1,
            burn_in=1,
            mcmc_steps=1,
            log_dir=str(run_dir),
            log_every=1,
            device=args.device,
        ),
    )
    result.pop("model")
    metrics_path = run_dir / "metrics.jsonl"
    analysis = analyze_history(load_metrics_jsonl(metrics_path), "h2", block_size=2)
    print(json.dumps({"metrics": str(metrics_path), "analysis": analysis}, indent=2))


if __name__ == "__main__":
    main()
