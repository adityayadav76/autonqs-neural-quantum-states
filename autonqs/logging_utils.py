from __future__ import annotations

import csv
import json
from pathlib import Path


class MetricLogger:
    def __init__(self, path: str = ""):
        self.path = Path(path) if path else None
        self._csv_path = None
        self._fields = None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self._csv_path = self.path / "metrics.csv"

    def write(self, metrics: dict) -> None:
        if not self.path:
            return
        json_path = self.path / "metrics.jsonl"
        with json_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._csv_path:
            fields = list(metrics.keys())
            write_header = not self._csv_path.exists()
            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)
