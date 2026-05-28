from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .benchmarks import BENCHMARKS


def block_statistics(values: list[float], block_size: int = 5) -> dict:
    if not values:
        return {"mean": math.nan, "stderr": math.nan, "blocks": 0}
    blocks = [values[i : i + block_size] for i in range(0, len(values), block_size) if values[i : i + block_size]]
    means = torch.tensor([sum(b) / len(b) for b in blocks], dtype=torch.float64)
    mean = float(means.mean())
    stderr = float(means.std(unbiased=False) / math.sqrt(max(1, len(means))))
    return {"mean": mean, "stderr": stderr, "blocks": len(blocks), "block_size": block_size}


def compare_reference(molecule: str, energy: float) -> dict:
    key = molecule.lower().replace("-", "_")
    if key not in BENCHMARKS:
        return {"molecule": molecule, "has_reference": False}
    bench = BENCHMARKS[key]
    error = energy - bench.reference_energy_hartree
    return {
        "molecule": molecule,
        "has_reference": True,
        "reference_energy_hartree": bench.reference_energy_hartree,
        "energy_hartree": energy,
        "error_hartree": error,
        "abs_error_hartree": abs(error),
        "within_tolerance": abs(error) <= bench.tolerance_hartree,
        "tolerance_hartree": bench.tolerance_hartree,
        "source": bench.source,
    }


def analyze_history(history: list[dict], molecule: str = "", block_size: int = 5) -> dict:
    energies = [float(item["energy"]) for item in history if "energy" in item]
    best = min(energies) if energies else math.nan
    final = energies[-1] if energies else math.nan
    out = {
        "points": len(energies),
        "best_energy": best,
        "final_energy": final,
        "energy_blocks": block_statistics(energies, block_size),
    }
    if molecule:
        comparison_energy = out["energy_blocks"]["mean"] if energies else final
        out["reference"] = compare_reference(molecule, comparison_energy)
    return out


def load_metrics_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
