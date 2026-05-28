from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimizerConfig:
    name: str = "sr"
    lr: float = 1e-3
    damping: float = 5e-3
    momentum: float = 0.9
    stat_decay: float = 0.95
    update_freq: int = 1
    max_update_norm: float = 0.02


class DiagonalNaturalGradient(torch.optim.Optimizer):
    """Diagonal Fisher natural-gradient optimizer.

    This is a robust natural-gradient baseline for VMC. It preconditions each
    parameter by an exponential moving average of squared gradients.
    """

    def __init__(self, params, lr: float = 2e-4, damping: float = 1e-3, stat_decay: float = 0.95):
        defaults = dict(lr=lr, damping=damping, stat_decay=stat_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            decay = group["stat_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "fisher_diag" not in state:
                    state["fisher_diag"] = torch.zeros_like(p)
                fisher = state["fisher_diag"]
                fisher.mul_(decay).addcmul_(p.grad, p.grad, value=1.0 - decay)
                p.addcdiv_(p.grad, fisher.sqrt().add_(damping), value=-lr)
        return loss


class KFACOptimizer(torch.optim.Optimizer):
    """Lightweight K-FAC-style optimizer for Linear layers.

    The optimizer tracks Kronecker factors for `nn.Linear` modules via hooks and
    preconditions matrix gradients when factor dimensions are available. Other
    parameters fall back to diagonal natural-gradient scaling.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4, damping: float = 1e-2, stat_decay: float = 0.95):
        params = list(model.parameters())
        super().__init__(params, dict(lr=lr, damping=damping, stat_decay=stat_decay))
        self.model = model
        self.factors: dict[nn.Module, dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_pre_hook(self._save_input))
                self.handles.append(module.register_full_backward_hook(self._save_grad_output))

    def _ema(self, old: torch.Tensor | None, new: torch.Tensor, decay: float) -> torch.Tensor:
        return new.detach() if old is None else old.mul(decay).add(new.detach(), alpha=1.0 - decay)

    def _save_input(self, module: nn.Module, inputs):
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1])
        cov = x.t().matmul(x) / max(1, x.shape[0])
        decay = self.param_groups[0]["stat_decay"]
        self.factors[module]["a"] = self._ema(self.factors[module].get("a"), cov, decay)

    def _save_grad_output(self, module: nn.Module, grad_input, grad_output):
        g = grad_output[0].detach()
        g = g.reshape(-1, g.shape[-1])
        cov = g.t().matmul(g) / max(1, g.shape[0])
        decay = self.param_groups[0]["stat_decay"]
        self.factors[module]["g"] = self._ema(self.factors[module].get("g"), cov, decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        linear_params = set()
        for module in self.model.modules():
            if not isinstance(module, nn.Linear) or module.weight.grad is None:
                continue
            linear_params.add(module.weight)
            fac = self.factors.get(module, {})
            a = fac.get("a")
            g = fac.get("g")
            if a is None or g is None:
                module.weight.add_(module.weight.grad, alpha=-lr)
            else:
                eye_a = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
                eye_g = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
                a_inv = torch.linalg.pinv(a + damping * eye_a)
                g_inv = torch.linalg.pinv(g + damping * eye_g)
                precond = g_inv.matmul(module.weight.grad).matmul(a_inv)
                module.weight.add_(precond, alpha=-lr)
            if module.bias is not None and module.bias.grad is not None:
                linear_params.add(module.bias)
                module.bias.add_(module.bias.grad / (module.bias.grad.square().mean().sqrt() + damping), alpha=-lr)

        for p in self.param_groups[0]["params"]:
            if p in linear_params or p.grad is None:
                continue
            state = self.state[p]
            if "diag" not in state:
                state["diag"] = torch.zeros_like(p)
            state["diag"].mul_(group["stat_decay"]).addcmul_(p.grad, p.grad, value=1.0 - group["stat_decay"])
            p.addcdiv_(p.grad, state["diag"].sqrt().add_(damping), value=-lr)
        return loss


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    name = config.name.lower()
    if name in {"sr", "natural", "stochastic-reconfiguration"}:
        return torch.optim.SGD(model.parameters(), lr=0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    if name in {"diag-natural", "diagonal-natural"}:
        return DiagonalNaturalGradient(model.parameters(), lr=config.lr, damping=config.damping, stat_decay=config.stat_decay)
    if name == "kfac":
        return KFACOptimizer(model, lr=config.lr, damping=config.damping, stat_decay=config.stat_decay)
    raise ValueError(f"Unknown optimizer {config.name!r}")


def _trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def stochastic_reconfiguration_step(
    model: nn.Module,
    logpsi: torch.Tensor,
    local_energy: torch.Tensor,
    lr: float,
    damping: float,
    max_update_norm: float = 0.1,
    energy_clip_sigma: float = 5.0,
) -> torch.Tensor:
    """Apply a VMC stochastic-reconfiguration/natural-gradient update.

    The solve is done in walker space, avoiding a dense parameter covariance
    matrix. It is intended for moderate walker counts and is much closer to the
    optimizer used in production VMC than diagonal gradient scaling.
    """

    params = _trainable_params(model)
    rows = []
    for i in range(logpsi.shape[0]):
        grads = torch.autograd.grad(logpsi[i], params, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, grads)
                ]
            )
        )
    o_matrix = torch.stack(rows, dim=0)
    o_matrix = o_matrix - o_matrix.mean(dim=0, keepdim=True)
    energy = local_energy.detach()
    if energy_clip_sigma > 0 and energy.numel() > 1:
        center = energy.median()
        scale = (energy - center).abs().median().clamp_min(1e-6) * 1.4826
        energy = energy.clamp(center - energy_clip_sigma * scale, center + energy_clip_sigma * scale)
    centered_energy = (energy - energy.mean()).reshape(-1, 1)
    n = o_matrix.shape[0]
    gram = o_matrix.matmul(o_matrix.t())
    eye = torch.eye(n, device=gram.device, dtype=gram.dtype)
    rhs = 2.0 * centered_energy
    alpha = torch.linalg.solve(gram + (n * damping) * eye, rhs)
    update = o_matrix.t().matmul(alpha).squeeze(1)
    norm = update.norm().clamp_min(1e-12)
    if max_update_norm > 0 and norm > max_update_norm:
        update = update * (max_update_norm / norm)
    offset = 0
    with torch.no_grad():
        for p in params:
            size = p.numel()
            p.add_(update[offset : offset + size].view_as(p), alpha=-lr)
            offset += size
    return update.detach()
