from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import torch

from .molecules import get_molecule
from .training import TrainConfig, train


def profile_training(
    molecule: str = "h2",
    steps: int = 5,
    walkers: int = 16,
    device: str = "cuda",
    output: str | Path | None = None,
) -> dict:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    result = train(
        get_molecule(molecule),
        TrainConfig(
            steps=steps,
            walkers=walkers,
            hidden=24,
            layers=1,
            hidden_density=2,
            burn_in=2,
            mcmc_steps=2,
            seed=7,
            deterministic=True,
            device=device,
        ),
    )
    total = perf_counter() - start
    result.pop("model")
    cuda_memory = None
    if device == "cuda" and torch.cuda.is_available():
        cuda_memory = {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    payload = {
        "molecule": molecule,
        "steps": steps,
        "walkers": walkers,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": total,
        "train_seconds": result["seconds"],
        "steps_per_second": result["steps_per_second"],
        "parameters": result["parameters"],
        "cuda_memory": cuda_memory,
        "result": result,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_profile(current: dict, baseline: dict, max_slowdown: float = 1.25) -> dict:
    base = float(baseline["steps_per_second"])
    now = float(current["steps_per_second"])
    ratio = now / base if base else 0.0
    return {
        "passed": ratio >= 1.0 / max_slowdown,
        "current_steps_per_second": now,
        "baseline_steps_per_second": base,
        "speed_ratio": ratio,
        "max_slowdown": max_slowdown,
    }
