from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import RunConfig
from .constants import BOHR_PER_ANGSTROM
from .analysis import analyze_history, compare_reference
from .distributed import average_tensor, init_distributed, wrap_distributed
from .excited import overlap_penalty, variance_penalty
from .forces import force_report
from .hamiltonian import local_energy_batched
from .logging_utils import MetricLogger
from .molecules import Molecule
from .network import AutoNQS, NetworkConfig
from .optim import OptimizerConfig, build_optimizer, stochastic_reconfiguration_step
from .pseudopotentials import get_pseudopotential
from .sampler import SamplerState, burn_in, make_state, metropolis_step, metropolis_sweep, refresh_logabs


@dataclass
class TrainConfig:
    steps: int = 400
    walkers: int = 128
    hidden: int = 96
    layers: int = 3
    hidden_density: int = 4
    rbm_hidden: int | None = None
    orbital_reference: bool = False
    backflow: bool = True
    lr: float = 1e-3
    mcmc_steps: int = 10
    burn_in: int = 100
    step_size: float = 0.08
    target_acceptance: float = 0.6
    adapt_mcmc: bool = True
    optimizer: str = "sr"
    damping: float = 5e-3
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 7
    deterministic: bool = False
    log_dir: str = ""
    log_every: int = 0
    checkpoint_path: str = ""
    checkpoint_every: int = 0
    energy_batch_size: int = 0
    data_parallel: bool = False
    reference_compare: bool = True
    estimate_forces: bool = False
    electron_wise_mcmc: bool = True
    restore_best: bool = True
    validation_sweeps: int = 20


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def energy_loss(logpsi: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    centered = energy.detach() - energy.detach().mean()
    return 2.0 * (centered * logpsi).mean()


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _state_from_checkpoint(payload: dict, device: torch.device, default: SamplerState) -> SamplerState:
    if "sampler_positions" not in payload:
        return default
    return SamplerState(
        payload["sampler_positions"].to(device),
        payload["sampler_logabs"].to(device),
        step_size=float(payload.get("sampler_step_size", default.step_size)),
        diagnostics=default.diagnostics,
    )


def train_from_run_config(config: RunConfig) -> dict:
    from .molecules import get_molecule

    train_cfg = TrainConfig(
        steps=config.steps,
        walkers=config.walkers,
        mcmc_steps=config.mcmc_steps,
        burn_in=config.burn_in,
        step_size=config.step_size,
        target_acceptance=config.target_acceptance,
        adapt_mcmc=config.adapt_mcmc,
        electron_wise_mcmc=config.electron_wise_mcmc,
        optimizer=config.optimizer.name,
        damping=config.optimizer.damping,
        lr=config.optimizer.lr,
        device=config.device,
        dtype=config.dtype,
        seed=config.seed,
        deterministic=config.deterministic,
        log_every=config.log_every,
        checkpoint_path=config.checkpoint_path,
        checkpoint_every=config.checkpoint_every,
        energy_batch_size=config.energy_batch_size,
        data_parallel=config.data_parallel,
        estimate_forces=config.estimate_forces,
        hidden=config.network.hidden,
        layers=config.network.layers,
        hidden_density=config.network.hidden_density,
        rbm_hidden=config.network.rbm_hidden,
        orbital_reference=config.network.orbital_reference,
        backflow=config.network.backflow,
    )
    return train(get_molecule(config.molecule), train_cfg, network_config=config.network, optimizer_config=config.optimizer, run_config=config)


def train(
    molecule: Molecule,
    config: TrainConfig,
    network_config: NetworkConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    run_config: RunConfig | None = None,
) -> dict:
    set_reproducibility(config.seed, config.deterministic)
    torch.set_default_dtype(_dtype(config.dtype))
    ddp_device = init_distributed(run_config.distributed) if run_config is not None else None
    device = ddp_device or resolve_device(config.device)
    nuclei, charges = molecule.tensors(device)
    n_up, _ = molecule.spin_counts
    net_cfg = network_config or NetworkConfig(
        hidden=config.hidden,
        layers=config.layers,
        hidden_density=config.hidden_density,
        rbm_hidden=config.rbm_hidden,
        orbital_reference=config.orbital_reference,
        backflow=config.backflow,
    )
    model = AutoNQS.from_config(molecule.electron_count, n_up, nuclei, charges, net_cfg).to(device)
    if run_config and run_config.periodic_cell is not None:
        model.register_buffer("cell", torch.tensor(run_config.periodic_cell, device=device, dtype=torch.get_default_dtype()) * BOHR_PER_ANGSTROM)
    else:
        cell = molecule.cell_tensor(device)
        if cell is not None:
            model.register_buffer("cell", cell)
    pseudo_names = run_config.pseudopotentials if run_config and run_config.pseudopotentials else molecule.pseudopotentials
    if pseudo_names:
        if len(pseudo_names) != len(molecule.symbols):
            raise ValueError("pseudopotentials must have one entry per nucleus")
        model.pseudopotentials = tuple(None if name is None else get_pseudopotential(name) for name in pseudo_names)
    else:
        model.pseudopotentials = None
    previous_models = []
    if run_config and run_config.excited.previous_checkpoints:
        for path in run_config.excited.previous_checkpoints:
            prev = AutoNQS.from_config(molecule.electron_count, n_up, nuclei, charges, net_cfg).to(device)
            prev.load_state_dict(load_checkpoint(path, map_location=device)["model"])
            prev.eval()
            previous_models.append(prev)
    if config.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if run_config and run_config.distributed.enabled:
        model = wrap_distributed(model, True)

    opt_cfg = optimizer_config or OptimizerConfig(name=config.optimizer, lr=config.lr, damping=config.damping)
    opt = build_optimizer(_base_model(model), opt_cfg)
    state = make_state(_base_model(model), config.walkers, step_size=config.step_size)
    history: list[dict] = []
    start_step = 1
    best_energy = float("inf")
    best_model_state = None
    best_sampler_state = None

    if config.checkpoint_path:
        try:
            payload = load_checkpoint(config.checkpoint_path, map_location=device)
            _base_model(model).load_state_dict(payload["model"])
            opt.load_state_dict(payload["optimizer"])
            state = _state_from_checkpoint(payload, device, state)
            history = list(payload.get("history", []))
            start_step = int(payload.get("step", 0)) + 1
            if history:
                best_energy = min(item["energy"] for item in history)
            if payload.get("rng_state") is not None:
                torch.set_rng_state(payload["rng_state"].cpu())
            if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
        except FileNotFoundError:
            pass

    if config.burn_in > 0 and start_step == 1:
        state = burn_in(_base_model(model), state, config.burn_in, config.target_acceptance, config.adapt_mcmc)

    logger = MetricLogger(config.log_dir)
    log_every = config.log_every or max(1, config.steps // 10)
    start = perf_counter()
    for step in range(start_step, config.steps + 1):
        for _ in range(config.mcmc_steps):
            if config.electron_wise_mcmc:
                state = metropolis_sweep(_base_model(model), state)
            else:
                state = metropolis_step(_base_model(model), state)
        if config.adapt_mcmc:
            from .sampler import adapt_step_size

            state = adapt_step_size(state, config.target_acceptance)

        opt.zero_grad(set_to_none=True)
        e_local = local_energy_batched(model, state.positions, config.energy_batch_size)
        e_mean_for_reduce = e_local.mean().detach().clone()
        e_mean_for_reduce = average_tensor(e_mean_for_reduce)
        logpsi = model(state.positions)
        loss = energy_loss(logpsi, e_local)
        if run_config:
            loss = loss + overlap_penalty(_base_model(model), previous_models, state.positions, run_config.excited.overlap_penalty)
            loss = loss + variance_penalty(e_local, run_config.excited.variance_penalty)
        opt_name = opt_cfg.name.lower()
        if opt_name in {"sr", "natural", "stochastic-reconfiguration"}:
            stochastic_reconfiguration_step(
                _base_model(model),
                logpsi,
                e_local,
                lr=opt_cfg.lr,
                damping=opt_cfg.damping,
                max_update_norm=opt_cfg.max_update_norm,
            )
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_base_model(model).parameters(), 1.0)
            opt.step()
        state = refresh_logabs(_base_model(model), state)

        if step == 1 or step % log_every == 0 or step == config.steps:
            metrics = {
                "step": step,
                "energy": float(e_mean_for_reduce.detach().cpu()),
                "std": float(e_local.std(unbiased=False).detach().cpu()),
                "acceptance": state.acceptance,
                "global_acceptance": state.diagnostics.acceptance_rate,
                "step_size": state.step_size,
                "loss": float(loss.detach().cpu()),
            }
            history.append(metrics)
            logger.write(metrics)
            if config.restore_best and metrics["energy"] < best_energy:
                best_energy = metrics["energy"]
                best_model_state = {k: v.detach().cpu().clone() for k, v in _base_model(model).state_dict().items()}
                best_sampler_state = deepcopy(state)

        if config.checkpoint_path and config.checkpoint_every and step % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint_path, model=model, optimizer=opt, config=run_config or config, state=state, history=history, step=step)

    elapsed = perf_counter() - start
    if config.restore_best and best_model_state is not None:
        _base_model(model).load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if best_sampler_state is not None:
            state = refresh_logabs(_base_model(model), best_sampler_state)
    validation_energy = None
    validation_std = None
    if config.validation_sweeps > 0:
        for _ in range(config.validation_sweeps):
            if config.electron_wise_mcmc:
                state = metropolis_sweep(_base_model(model), state)
            else:
                state = metropolis_step(_base_model(model), state)
        validation_local = local_energy_batched(model, state.positions, config.energy_batch_size)
        validation_energy = float(validation_local.mean().detach().cpu())
        validation_std = float(validation_local.std(unbiased=False).detach().cpu())
    if config.checkpoint_path:
        save_checkpoint(config.checkpoint_path, model=model, optimizer=opt, config=run_config or config, state=state, history=history, step=config.steps)

    analysis = analyze_history(history, molecule.name, run_config.analysis_block_size if run_config else 5)
    forces = force_report(_base_model(model), state.positions) if config.estimate_forces or (run_config and run_config.estimate_forces) else None
    reported_final = validation_energy if validation_energy is not None else (best_energy if config.restore_best and best_model_state is not None else history[-1]["energy"])
    return {
        "molecule": molecule.name,
        "device": str(device),
        "data_parallel": isinstance(model, torch.nn.DataParallel),
        "seed": config.seed,
        "deterministic": config.deterministic,
        "electrons": molecule.electron_count,
        "spin_counts": molecule.spin_counts,
        "parameters": sum(p.numel() for p in _base_model(model).parameters()),
        "seconds": elapsed,
        "steps_per_second": max(0, config.steps - start_step + 1) / elapsed if elapsed else float("inf"),
        "history": history,
        "final_energy": reported_final,
        "best_energy": min(item["energy"] for item in history),
        "validation_energy": validation_energy,
        "validation_std": validation_std,
        "sampler": {
            "acceptance_rate": state.diagnostics.acceptance_rate,
            "step_size": state.step_size,
            "total_proposals": state.diagnostics.total_proposals,
        },
        "reference": compare_reference(molecule.name, analysis["energy_blocks"]["mean"]) if config.reference_compare else None,
        "analysis": analysis,
        "forces": forces,
        "model": model,
    }
