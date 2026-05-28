from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .network import NetworkConfig
from .optim import OptimizerConfig
from .excited import ExcitedStateConfig
from .distributed import DistributedConfig


@dataclass
class RunConfig:
    molecule: str = "h2"
    steps: int = 200
    walkers: int = 128
    mcmc_steps: int = 10
    burn_in: int = 50
    step_size: float = 0.04
    target_acceptance: float = 0.55
    adapt_mcmc: bool = True
    electron_wise_mcmc: bool = True
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 7
    deterministic: bool = False
    log_every: int = 0
    energy_batch_size: int = 0
    data_parallel: bool = False
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    periodic_cell: tuple[tuple[float, float, float], ...] | None = None
    pseudopotentials: tuple[str | None, ...] | None = None
    excited: ExcitedStateConfig = field(default_factory=ExcitedStateConfig)
    estimate_forces: bool = False
    analysis_block_size: int = 5
    checkpoint_path: str = ""
    checkpoint_every: int = 0
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def _filter_dataclass(cls, values: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in names})


def run_config_from_dict(values: dict[str, Any]) -> RunConfig:
    data = dict(values)
    if isinstance(data.get("network"), dict):
        data["network"] = _filter_dataclass(NetworkConfig, data["network"])
    if isinstance(data.get("optimizer"), dict):
        data["optimizer"] = _filter_dataclass(OptimizerConfig, data["optimizer"])
    if isinstance(data.get("excited"), dict):
        data["excited"] = _filter_dataclass(ExcitedStateConfig, data["excited"])
    if isinstance(data.get("distributed"), dict):
        data["distributed"] = _filter_dataclass(DistributedConfig, data["distributed"])
    if isinstance(data.get("periodic_cell"), list):
        data["periodic_cell"] = tuple(tuple(float(v) for v in row) for row in data["periodic_cell"])
    if isinstance(data.get("pseudopotentials"), list):
        data["pseudopotentials"] = tuple(data["pseudopotentials"])
    return _filter_dataclass(RunConfig, data)


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        return run_config_from_dict(json.load(f))


def save_config(config: RunConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(config), f, indent=2)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
