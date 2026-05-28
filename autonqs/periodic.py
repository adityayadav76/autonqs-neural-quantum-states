from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PeriodicCell:
    vectors: tuple[tuple[float, float, float], ...]
    k_points: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    minimum_image: bool = True
    madelung: float = 0.0

    def tensor(self, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.tensor(self.vectors, device=device, dtype=dtype or torch.get_default_dtype())


def fractional_coordinates(x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(cell.t(), x.reshape(-1, 3).t()).t().reshape_as(x)


def cartesian_coordinates(frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    return frac.matmul(cell)


def wrap_positions(x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac = fractional_coordinates(x, cell)
    return cartesian_coordinates(frac - torch.floor(frac), cell)


def minimum_image_displacement(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac = fractional_coordinates(delta, cell)
    frac = frac - torch.round(frac)
    return cartesian_coordinates(frac, cell)


def reciprocal_vectors(cell: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.pi * torch.linalg.inv(cell).t()
