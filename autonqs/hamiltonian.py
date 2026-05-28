from __future__ import annotations

import torch

from .periodic import minimum_image_displacement
from .pseudopotentials import effective_charges, local_pseudopotential_energy


def potential_energy(
    x: torch.Tensor,
    nuclei: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor | None = None,
    pseudopotentials: tuple | None = None,
) -> torch.Tensor:
    eff_charges = effective_charges(charges, pseudopotentials)
    rel_en = x[:, :, None, :] - nuclei[None, None, :, :]
    if cell is not None:
        rel_en = minimum_image_displacement(rel_en, cell)
    r_en = torch.linalg.norm(rel_en, dim=-1).clamp_min(1e-6)
    v_en = -(eff_charges[None, None, :] / r_en).sum(dim=(1, 2))

    rel_ee = x[:, :, None, :] - x[:, None, :, :]
    if cell is not None:
        rel_ee = minimum_image_displacement(rel_ee, cell)
    r_ee = torch.linalg.norm(rel_ee, dim=-1).clamp_min(1e-6)
    iu = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
    v_ee = (1.0 / r_ee[:, iu[0], iu[1]]).sum(dim=-1)

    rel_nn = nuclei[:, None, :] - nuclei[None, :, :]
    if cell is not None:
        rel_nn = minimum_image_displacement(rel_nn, cell)
    r_nn = torch.linalg.norm(rel_nn, dim=-1).clamp_min(1e-6)
    ju = torch.triu_indices(nuclei.shape[0], nuclei.shape[0], offset=1, device=x.device)
    v_nn = (eff_charges[ju[0]] * eff_charges[ju[1]] / r_nn[ju[0], ju[1]]).sum()
    v_pp = local_pseudopotential_energy(x, nuclei, charges, pseudopotentials)
    return v_en + v_ee + v_nn + v_pp


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def local_energy(model, x: torch.Tensor) -> torch.Tensor:
    base_model = _unwrap(model)
    x_req = x.detach().clone().requires_grad_(True)
    logpsi = model(x_req)
    grad = torch.autograd.grad(logpsi.sum(), x_req, create_graph=True)[0]
    lap = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    flat_grad = grad.reshape(x.shape[0], -1)
    flat_x = x_req.reshape(x.shape[0], -1)
    for i in range(flat_x.shape[1]):
        gi = flat_grad[:, i].sum()
        second = torch.autograd.grad(gi, x_req, create_graph=True)[0].reshape(x.shape[0], -1)[:, i]
        lap = lap + second
    kinetic = -0.5 * (lap + grad.square().sum(dim=(1, 2)))
    cell = getattr(base_model, "cell", None)
    pseudopotentials = getattr(base_model, "pseudopotentials", None)
    return kinetic + potential_energy(x_req, base_model.nuclei, base_model.charges, cell, pseudopotentials)


def local_energy_batched(model, x: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    if batch_size is None or batch_size <= 0 or x.shape[0] <= batch_size:
        return local_energy(model, x)
    chunks = [local_energy(model, chunk) for chunk in x.split(batch_size, dim=0)]
    return torch.cat(chunks, dim=0)
