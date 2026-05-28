from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Pseudopotential:
    symbol: str
    core_electrons: int
    local_radius: float
    local_strength: float = 1.0


PSEUDOPOTENTIALS: dict[str, Pseudopotential] = {
    "BFD-H": Pseudopotential("H", 0, 0.20),
    "BFD-C": Pseudopotential("C", 2, 0.35),
    "BFD-N": Pseudopotential("N", 2, 0.35),
    "BFD-O": Pseudopotential("O", 2, 0.35),
    "BFD-S": Pseudopotential("S", 10, 0.50),
    "BFD-Fe": Pseudopotential("Fe", 18, 0.70),
    "BFD-Mn": Pseudopotential("Mn", 18, 0.70),
    "BFD-Ir": Pseudopotential("Ir", 60, 0.90),
}


def get_pseudopotential(name: str) -> Pseudopotential:
    if name not in PSEUDOPOTENTIALS:
        raise KeyError(f"Unknown pseudopotential {name!r}. Options: {', '.join(PSEUDOPOTENTIALS)}")
    return PSEUDOPOTENTIALS[name]


def effective_charges(charges: torch.Tensor, specs: tuple[Pseudopotential | None, ...] | None) -> torch.Tensor:
    if specs is None:
        return charges
    vals = []
    for charge, spec in zip(charges, specs):
        vals.append(charge if spec is None else torch.clamp(charge - spec.core_electrons, min=1.0))
    return torch.stack(vals)


def local_pseudopotential_energy(x: torch.Tensor, nuclei: torch.Tensor, charges: torch.Tensor, specs: tuple[Pseudopotential | None, ...] | None) -> torch.Tensor:
    if specs is None:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    rel = x[:, :, None, :] - nuclei[None, None, :, :]
    r = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    terms = []
    for i, spec in enumerate(specs):
        if spec is None or spec.core_electrons <= 0:
            terms.append(torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype))
            continue
        rc = torch.as_tensor(spec.local_radius, device=x.device, dtype=x.dtype)
        core = torch.as_tensor(spec.core_electrons * spec.local_strength, device=x.device, dtype=x.dtype)
        terms.append(core * torch.exp(-(r[:, :, i] / rc).square()) / r[:, :, i])
    return torch.stack(terms, dim=-1).sum(dim=(1, 2))
