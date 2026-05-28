from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0


def config_from_env(enabled: bool = False, backend: str = "nccl") -> DistributedConfig:
    return DistributedConfig(
        enabled=enabled,
        backend=backend,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def init_distributed(config: DistributedConfig) -> torch.device | None:
    if not config.enabled:
        return None
    world_size = int(os.environ.get("WORLD_SIZE", str(config.world_size)))
    rank = int(os.environ.get("RANK", str(config.rank)))
    local_rank = int(os.environ.get("LOCAL_RANK", str(config.local_rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(config.backend, init_method=config.init_method, world_size=world_size, rank=rank)
    return torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


def is_primary() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def average_tensor(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def wrap_distributed(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    if torch.cuda.is_available():
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    return torch.nn.parallel.DistributedDataParallel(model)
