from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplerDiagnostics:
    total_proposals: int = 0
    total_accepts: int = 0
    recent_acceptance: list[float] = field(default_factory=list)
    step_size: float = 0.08

    @property
    def acceptance_rate(self) -> float:
        if self.total_proposals == 0:
            return 0.0
        return self.total_accepts / self.total_proposals


@dataclass
class SamplerState:
    positions: torch.Tensor
    logabs: torch.Tensor
    acceptance: float = 0.0
    step_size: float = 0.08
    diagnostics: SamplerDiagnostics = field(default_factory=SamplerDiagnostics)


def initialize_walkers(n_walkers: int, n_electrons: int, nuclei: torch.Tensor, charges: torch.Tensor, spread: float = 0.5) -> torch.Tensor:
    probs = charges / charges.sum()
    idx = torch.multinomial(probs, n_walkers * n_electrons, replacement=True).view(n_walkers, n_electrons)
    centers = nuclei[idx]
    return centers + spread * torch.randn(n_walkers, n_electrons, 3, device=nuclei.device, dtype=nuclei.dtype)


def _record(state: SamplerState, accept: torch.Tensor, step_size: float) -> SamplerDiagnostics:
    diag = state.diagnostics
    diag.total_proposals += int(accept.numel())
    diag.total_accepts += int(accept.sum().item())
    diag.recent_acceptance.append(float(accept.float().mean().item()))
    diag.recent_acceptance = diag.recent_acceptance[-100:]
    diag.step_size = step_size
    return diag


@torch.no_grad()
def metropolis_step(model, state: SamplerState, step_size: float | None = None) -> SamplerState:
    step = state.step_size if step_size is None else step_size
    proposal = state.positions + step * torch.randn_like(state.positions)
    _, prop_logabs = model.slog_psi(proposal)
    log_ratio = 2.0 * (prop_logabs - state.logabs)
    accept = torch.log(torch.rand_like(log_ratio)) < log_ratio
    positions = torch.where(accept[:, None, None], proposal, state.positions)
    logabs = torch.where(accept, prop_logabs, state.logabs)
    diagnostics = _record(state, accept, step)
    return SamplerState(positions, logabs, float(accept.float().mean().item()), step, diagnostics)


@torch.no_grad()
def metropolis_sweep(model, state: SamplerState, step_size: float | None = None) -> SamplerState:
    step = state.step_size if step_size is None else step_size
    positions = state.positions
    logabs = state.logabs
    accepts = []
    for electron in range(positions.shape[1]):
        proposal = positions.clone()
        proposal[:, electron, :] = proposal[:, electron, :] + step * torch.randn_like(proposal[:, electron, :])
        _, prop_logabs = model.slog_psi(proposal)
        log_ratio = 2.0 * (prop_logabs - logabs)
        accept = torch.log(torch.rand_like(log_ratio)) < log_ratio
        positions = torch.where(accept[:, None, None], proposal, positions)
        logabs = torch.where(accept, prop_logabs, logabs)
        accepts.append(accept)
    accept_all = torch.stack(accepts, dim=1).reshape(-1)
    diagnostics = _record(state, accept_all, step)
    return SamplerState(positions, logabs, float(accept_all.float().mean().item()), step, diagnostics)


@torch.no_grad()
def adapt_step_size(state: SamplerState, target: float = 0.6, rate: float = 0.08, min_step: float = 1e-4, max_step: float = 2.0) -> SamplerState:
    recent = state.diagnostics.recent_acceptance[-10:]
    if not recent:
        return state
    acc = sum(recent) / len(recent)
    factor = torch.exp(torch.tensor(rate * (acc - target))).item()
    step = min(max(state.step_size * factor, min_step), max_step)
    state.step_size = step
    state.diagnostics.step_size = step
    return state


@torch.no_grad()
def burn_in(model, state: SamplerState, steps: int, target_acceptance: float = 0.6, adapt: bool = True) -> SamplerState:
    for _ in range(steps):
        state = metropolis_sweep(model, state)
        if adapt:
            state = adapt_step_size(state, target_acceptance)
    return state


@torch.no_grad()
def refresh_logabs(model, state: SamplerState) -> SamplerState:
    _, logabs = model.slog_psi(state.positions)
    return SamplerState(state.positions, logabs, state.acceptance, state.step_size, state.diagnostics)


@torch.no_grad()
def make_state(model, n_walkers: int, step_size: float = 0.08) -> SamplerState:
    positions = initialize_walkers(n_walkers, model.n_electrons, model.nuclei, model.charges)
    _, logabs = model.slog_psi(positions)
    return SamplerState(positions, logabs, step_size=step_size, diagnostics=SamplerDiagnostics(step_size=step_size))
