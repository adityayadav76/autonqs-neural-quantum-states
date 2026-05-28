from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExcitedStateConfig:
    state_index: int = 0
    overlap_penalty: float = 0.0
    variance_penalty: float = 0.0
    previous_checkpoints: tuple[str, ...] = ()


@torch.no_grad()
def normalized_overlap(logpsi_a: torch.Tensor, sign_a: torch.Tensor, logpsi_b: torch.Tensor, sign_b: torch.Tensor) -> torch.Tensor:
    max_a = logpsi_a.max()
    max_b = logpsi_b.max()
    a = sign_a * torch.exp(logpsi_a - max_a)
    b = sign_b * torch.exp(logpsi_b - max_b)
    num = (a * b).mean()
    den = (a.square().mean().sqrt() * b.square().mean().sqrt()).clamp_min(1e-12)
    return num / den


def overlap_penalty(current_model, previous_models: list[torch.nn.Module], positions: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0 or not previous_models:
        return torch.zeros((), device=positions.device, dtype=positions.dtype)
    sign, logabs = current_model.slog_psi(positions)
    penalties = []
    for prev in previous_models:
        with torch.no_grad():
            p_sign, p_logabs = prev.slog_psi(positions)
        max_cur = logabs.max()
        max_prev = p_logabs.max()
        cur = sign * torch.exp(logabs - max_cur)
        old = p_sign * torch.exp(p_logabs - max_prev)
        ov = (cur * old).mean() / (cur.square().mean().sqrt() * old.square().mean().sqrt()).clamp_min(1e-12)
        penalties.append(ov.square())
    return weight * torch.stack(penalties).sum()


def variance_penalty(local_energy: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0:
        return torch.zeros((), device=local_energy.device, dtype=local_energy.dtype)
    return weight * local_energy.var(unbiased=False)
