from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .periodic import minimum_image_displacement


def inverse_softplus(value: float) -> torch.Tensor:
    x = torch.tensor(float(value))
    return torch.log(torch.expm1(x))


def log_two_cosh(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    return ax + F.softplus(-2.0 * ax)


@dataclass(frozen=True)
class NetworkConfig:
    hidden: int = 96
    pair_hidden: int = 32
    layers: int = 2
    hidden_density: int = 2
    rbm_hidden: int | None = None
    cusp_envelope: bool = True
    trainable_cusp: bool = False
    phase_hidden: int = 32
    pauli_nodes: bool = True
    node_epsilon: float = 1e-4
    envelope_softmin: float = 1.0
    orbital_reference: bool = False
    orbital_channels: int = 12
    orbital_jitter: float = 1e-4
    backflow: bool = True
    backflow_hidden: int = 64
    backflow_scale: float = 0.1


class NQSFeatureBlock(nn.Module):
    """Permutation-aware feature extractor for continuous electron configurations."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class AutoNQS(nn.Module):
    """Neural Quantum State wavefunction for all-electron VMC.

    The ansatz follows the NQS/RBM form from Carleo and Troyer while adapting the
    visible variables to continuous quantum-chemistry configurations. Electron-
    nuclear and electron-electron geometric features are pooled into a visible
    vector v(R), and the log wavefunction is

        a(v) + sum_j log(2 cosh(b_j + W_j v)) + cusp(R).

    This keeps the surrounding VMC machinery intact while allowing an
    antisymmetric orbital reference and neural backflow coordinates to be used
    underneath the NQS correlator.
    """

    def __init__(
        self,
        n_electrons: int,
        n_up: int,
        nuclei: torch.Tensor,
        charges: torch.Tensor,
        hidden: int = 96,
        layers: int = 2,
        hidden_density: int = 2,
        rbm_hidden: int | None = None,
        pair_hidden: int = 32,
        cusp_envelope: bool = True,
        trainable_cusp: bool = False,
        phase_hidden: int = 32,
        pauli_nodes: bool = True,
        node_epsilon: float = 1e-4,
        envelope_softmin: float = 1.0,
        orbital_reference: bool = False,
        orbital_channels: int = 12,
        orbital_jitter: float = 1e-4,
        backflow: bool = True,
        backflow_hidden: int = 64,
        backflow_scale: float = 0.1,
    ):
        super().__init__()
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.n_down = n_electrons - n_up
        self.n_nuclei = int(nuclei.shape[0])
        self.hidden_density = hidden_density
        self.rbm_hidden = int(rbm_hidden or max(hidden, hidden_density * n_electrons))
        self.cusp_envelope = cusp_envelope
        self.trainable_cusp = trainable_cusp
        self.pauli_nodes = pauli_nodes
        self.node_epsilon = node_epsilon
        self.envelope_softmin = envelope_softmin
        self.orbital_reference = orbital_reference
        self.orbital_channels = orbital_channels
        self.orbital_jitter = orbital_jitter
        self.backflow = backflow
        self.backflow_scale = backflow_scale
        self.register_buffer("nuclei", nuclei.detach().clone())
        self.register_buffer("charges", charges.detach().clone())
        axis = torch.tensor([1.0, 1.61803398875, 2.41421356237], dtype=nuclei.dtype, device=nuclei.device)
        self.register_buffer("node_axis", axis / torch.linalg.norm(axis))

        one_in = 4 * self.n_nuclei
        self.one_in = nn.Sequential(nn.Linear(one_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pair_in = nn.Sequential(nn.Linear(3, pair_hidden), nn.Tanh(), nn.Linear(pair_hidden, hidden), nn.Tanh())
        self.backflow_pair_in = nn.Sequential(nn.Linear(5, backflow_hidden), nn.Tanh(), nn.Linear(backflow_hidden, hidden), nn.Tanh())
        self.backflow_net = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, backflow_hidden),
            nn.Tanh(),
            nn.Linear(backflow_hidden, 3),
        )
        nn.init.zeros_(self.backflow_net[-1].weight)
        nn.init.zeros_(self.backflow_net[-1].bias)
        pooled = hidden * 4
        self.visible_in = nn.Sequential(nn.LayerNorm(pooled), nn.Linear(pooled, hidden), nn.Tanh())
        self.blocks = nn.ModuleList(NQSFeatureBlock(hidden) for _ in range(layers))

        self.visible_bias = nn.Linear(hidden, 1)
        self.hidden_linear = nn.Linear(hidden, self.rbm_hidden)
        self.phase_head = nn.Sequential(nn.Linear(hidden, phase_hidden), nn.Tanh(), nn.Linear(phase_hidden, 1))
        self.orbital_decay = nn.Parameter(torch.log(torch.expm1(charges.detach().clone().clamp_min(1e-4))))
        basis_dim = self.n_nuclei * orbital_channels
        self.up_orbital_coeff = nn.Parameter(torch.zeros(basis_dim, max(self.n_up, 1)))
        self.down_orbital_coeff = nn.Parameter(torch.zeros(basis_dim, max(self.n_down, 1)))
        with torch.no_grad():
            self.up_orbital_coeff[: self.n_nuclei, 0] = 1.0
            self.down_orbital_coeff[: self.n_nuclei, 0] = 1.0
            for i in range(1, min(basis_dim, max(self.n_up, 1))):
                self.up_orbital_coeff[i, i] = 1.0
            for i in range(1, min(basis_dim, max(self.n_down, 1))):
                self.down_orbital_coeff[i, i] = 1.0
        self.ee_cusp_same = nn.Parameter(inverse_softplus(0.25), requires_grad=trainable_cusp)
        self.ee_cusp_opposite = nn.Parameter(inverse_softplus(0.5), requires_grad=trainable_cusp)
        self.en_cusp_scale = nn.Parameter(inverse_softplus(1.0), requires_grad=trainable_cusp)

    @classmethod
    def from_config(
        cls,
        n_electrons: int,
        n_up: int,
        nuclei: torch.Tensor,
        charges: torch.Tensor,
        config: NetworkConfig,
    ) -> "AutoNQS":
        return cls(
            n_electrons,
            n_up,
            nuclei,
            charges,
            hidden=config.hidden,
            pair_hidden=config.pair_hidden,
            layers=config.layers,
            hidden_density=config.hidden_density,
            rbm_hidden=config.rbm_hidden,
            cusp_envelope=config.cusp_envelope,
            trainable_cusp=config.trainable_cusp,
            phase_hidden=config.phase_hidden,
            pauli_nodes=config.pauli_nodes,
            node_epsilon=config.node_epsilon,
            envelope_softmin=config.envelope_softmin,
            orbital_reference=config.orbital_reference,
            orbital_channels=config.orbital_channels,
            orbital_jitter=config.orbital_jitter,
            backflow=config.backflow,
            backflow_hidden=config.backflow_hidden,
            backflow_scale=config.backflow_scale,
        )

    def _distances(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_nuc = x[:, :, None, :] - self.nuclei[None, None, :, :]
        cell = getattr(self, "cell", None)
        if cell is not None:
            rel_nuc = minimum_image_displacement(rel_nuc, cell)
        r_en = torch.linalg.norm(rel_nuc, dim=-1).clamp_min(1e-6)
        rel_ee = x[:, :, None, :] - x[:, None, :, :]
        if cell is not None:
            rel_ee = minimum_image_displacement(rel_ee, cell)
        eye = torch.eye(self.n_electrons, device=x.device, dtype=x.dtype)
        r_ee = torch.linalg.norm(rel_ee + eye[None, :, :, None], dim=-1).clamp_min(1e-6)
        return rel_nuc, r_en, rel_ee, r_ee

    def electron_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_nuc, r_en, rel_ee, r_ee = self._distances(x)
        one = torch.cat([rel_nuc.flatten(2), r_en[..., None].flatten(2)], dim=-1)
        one_h = self.one_in(one)
        eye = torch.eye(self.n_electrons, device=x.device, dtype=x.dtype)
        same_spin_matrix = torch.zeros(self.n_electrons, self.n_electrons, device=x.device, dtype=x.dtype)
        if self.n_up:
            same_spin_matrix[: self.n_up, : self.n_up] = 1.0
        if self.n_down:
            same_spin_matrix[self.n_up :, self.n_up :] = 1.0
        pair_features = torch.cat(
            [
                rel_ee,
                (r_ee * (1.0 - eye)[None])[..., None],
                (same_spin_matrix * (1.0 - eye))[None, :, :, None].expand(x.shape[0], -1, -1, -1),
            ],
            dim=-1,
        )
        pair_h = self.backflow_pair_in(pair_features)
        pair_ctx = (pair_h * (1.0 - eye)[None, :, :, None]).sum(dim=2) / max(1, self.n_electrons - 1)
        return one_h, pair_ctx, r_en, r_ee

    def visible_features(self, x: torch.Tensor) -> torch.Tensor:
        one_h, pair_ctx, _, r_ee = self.electron_features(x)
        all_pool = one_h.mean(dim=1)
        up_pool = one_h[:, : self.n_up].mean(dim=1) if self.n_up else torch.zeros_like(all_pool)
        down_pool = one_h[:, self.n_up :].mean(dim=1) if self.n_down else torch.zeros_like(all_pool)

        idx = torch.triu_indices(self.n_electrons, self.n_electrons, offset=1, device=x.device)
        if idx.shape[1] == 0:
            pair_pool = torch.zeros_like(all_pool)
        else:
            pair_r = r_ee[:, idx[0], idx[1]]
            same_spin = (((idx[0] < self.n_up) & (idx[1] < self.n_up)) | ((idx[0] >= self.n_up) & (idx[1] >= self.n_up))).to(x.dtype)
            pair_features = torch.stack(
                [
                    pair_r,
                    1.0 / pair_r.clamp_min(1e-6),
                    same_spin[None].expand_as(pair_r),
                ],
                dim=-1,
            )
            pair_pool = self.pair_in(pair_features).mean(dim=1)

        v = self.visible_in(torch.cat([all_pool, up_pool, down_pool, pair_pool], dim=-1))
        for block in self.blocks:
            v = block(v)
        return v

    def backflow_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        if not self.backflow:
            return x
        one_h, pair_ctx, _, _ = self.electron_features(x)
        displacement = self.backflow_scale * torch.tanh(self.backflow_net(torch.cat([one_h, pair_ctx], dim=-1)))
        return x + displacement

    def cusp_log_factor(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cusp_envelope:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        _, r_en, _, r_ee = self._distances(x)
        if self.orbital_reference:
            electron_nuclear = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            en_scale = F.softplus(self.en_cusp_scale)
            orbital_decay = F.softplus(self.orbital_decay)[None, None, :]
            charged_distance = orbital_decay * r_en
            beta = torch.as_tensor(max(self.envelope_softmin, 1e-6), device=x.device, dtype=x.dtype)
            electron_nuclear = torch.logsumexp(-beta * en_scale * charged_distance, dim=-1).sum(dim=1) / beta

        idx = torch.triu_indices(self.n_electrons, self.n_electrons, offset=1, device=x.device)
        if idx.shape[1] == 0:
            return electron_nuclear
        pair_r = r_ee[:, idx[0], idx[1]]
        same_spin = ((idx[0] < self.n_up) & (idx[1] < self.n_up)) | ((idx[0] >= self.n_up) & (idx[1] >= self.n_up))
        cusp = torch.where(
            same_spin[None],
            F.softplus(self.ee_cusp_same).to(x.dtype),
            F.softplus(self.ee_cusp_opposite).to(x.dtype),
        )
        electron_electron = (cusp * pair_r / (1.0 + pair_r)).sum(dim=1)
        return electron_nuclear + electron_electron

    def pauli_log_factor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        logabs = torch.zeros_like(sign)
        if not self.pauli_nodes or self.orbital_reference:
            return sign, logabs
        for start, count in ((0, self.n_up), (self.n_up, self.n_down)):
            if count < 2:
                continue
            idx = torch.triu_indices(count, count, offset=1, device=x.device)
            sector = x[:, start : start + count]
            projected = sector @ self.node_axis.to(device=x.device, dtype=x.dtype)
            delta = projected[:, idx[0]] - projected[:, idx[1]]
            pair_sign = torch.sign(delta).masked_fill(delta == 0, 1.0).prod(dim=-1)
            sign = sign * pair_sign
            logabs = logabs + torch.log(delta.abs().clamp_min(self.node_epsilon)).sum(dim=-1)
        return sign, logabs

    def orbital_basis(self, x: torch.Tensor) -> torch.Tensor:
        rel_nuc, r_en, _, _ = self._distances(x)
        decay = F.softplus(self.orbital_decay)[None, None, :]
        env = torch.exp(-decay * r_en)
        rx, ry, rz = rel_nuc.unbind(dim=-1)
        r = r_en
        channels = [
            torch.ones_like(r_en),
            r,
            r.square(),
            rx,
            ry,
            rz,
            rx * r,
            ry * r,
            rz * r,
            rx * ry,
            rx * rz,
            ry * rz,
            rx.square() - ry.square(),
            2.0 * rz.square() - rx.square() - ry.square(),
        ][: self.orbital_channels]
        return torch.cat([(channel * env).flatten(2) for channel in channels], dim=-1)

    def orbital_log_factor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        logabs = torch.zeros_like(sign)
        if not self.orbital_reference:
            return sign, logabs
        basis = self.orbital_basis(self.backflow_coordinates(x))
        for start, count, coeff in ((0, self.n_up, self.up_orbital_coeff), (self.n_up, self.n_down, self.down_orbital_coeff)):
            if count == 0:
                continue
            sector_basis = basis[:, start : start + count]
            mat = torch.einsum("web,bo->weo", sector_basis, coeff[:, :count])
            if self.orbital_jitter:
                eye = torch.eye(count, device=x.device, dtype=x.dtype)
                mat = mat + self.orbital_jitter * eye[None]
            sector_sign, sector_logabs = torch.linalg.slogdet(mat)
            sign = sign * sector_sign.masked_fill(sector_sign == 0, 1.0)
            logabs = logabs + sector_logabs
        return sign, logabs

    def slog_psi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visible = self.visible_features(x)
        rbm_terms = log_two_cosh(self.hidden_linear(visible)).sum(dim=-1) / self.rbm_hidden**0.5
        logabs = self.visible_bias(visible).squeeze(-1) + rbm_terms + self.cusp_log_factor(x)
        phase = self.phase_head(visible).squeeze(-1)
        learned_sign = torch.where(phase >= 0.0, torch.ones_like(phase), -torch.ones_like(phase))
        pauli_sign, pauli_logabs = self.pauli_log_factor(x)
        orbital_sign, orbital_logabs = self.orbital_log_factor(x)
        return learned_sign * pauli_sign * orbital_sign, logabs + pauli_logabs + orbital_logabs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slog_psi(x)[1]
