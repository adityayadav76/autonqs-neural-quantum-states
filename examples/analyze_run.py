from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonqs.analysis import analyze_history, load_metrics_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze autonqs metric logs.")
    parser.add_argument("metrics_jsonl")
    parser.add_argument("--molecule", default="")
    parser.add_argument("--block-size", type=int, default=5)
    args = parser.parse_args()
    history = load_metrics_jsonl(args.metrics_jsonl)
    print(json.dumps(analyze_history(history, args.molecule, args.block_size), indent=2))


if __name__ == "__main__":
    main()
