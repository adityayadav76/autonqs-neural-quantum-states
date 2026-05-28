from __future__ import annotations

import torch

from .hamiltonian import potential_energy


def estimate_forces(model, positions: torch.Tensor) -> torch.Tensor:
    """Estimate nuclear forces from the Coulomb local potential derivative.

    This is the Hellmann-Feynman part of the VMC force estimator. Pulay terms
    require geometry-differentiated wavefunction training and are reported as a
    future high-accuracy extension rather than silently folded in.
    """

    base = model.module if hasattr(model, "module") else model
    nuclei = base.nuclei.detach().clone().requires_grad_(True)
    charges = base.charges.detach()
    cell = getattr(base, "cell", None)
    pseudopotentials = getattr(base, "pseudopotentials", None)
    energy = potential_energy(positions.detach(), nuclei, charges, cell, pseudopotentials).mean()
    grad = torch.autograd.grad(energy, nuclei)[0]
    return -grad


def force_report(model, positions: torch.Tensor) -> dict:
    forces = estimate_forces(model, positions)
    return {
        "forces": forces.detach().cpu().tolist(),
        "max_force": float(forces.norm(dim=-1).max().detach().cpu()),
        "rms_force": float(forces.square().sum(dim=-1).mean().sqrt().detach().cpu()),
        "estimator": "hellmann_feynman_local_potential",
    }
