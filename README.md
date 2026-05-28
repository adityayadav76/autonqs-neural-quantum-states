# AutoNQS - Neural Quantum States
A Research Grade Framework For Quantum Chemistry using Pytorch

![](https://automatski.com/wp-content/uploads/2025/05/Automatski-New-Logo.svg)

## About

This repository contains a research grade implementation of Quantum Chemistry using Fermi Neural Networks created by [Automatski](https://automatski.com). 

AutoNQS is a PyTorch NQS/RBM-style variational Monte Carlo framework for molecular ground-state calculations on CUDA GPUs. It now includes a closer continuous-coordinate neural quantum state ansatz, cusp-aware log factors, adaptive MCMC, natural-gradient and K-FAC-style optimizers, checkpointing, JSON configs, metric logs, benchmark systems, excited-state penalties, periodic cells, pseudopotentials, force
estimation, distributed wrappers, and result analysis.

To see a demonstration of code in this repo please see the [Demo Video]()

### Intellectual Property

All rights are reserved by Automatski for Automatski-authored components of this codebase. Rights to third-party or upstream components remain with their respective original authors and licensors.

## Installation

```sh
pip install -r requirements.txt
```

## Quick Start

```cmd
python -m pytest -q
python -m autonqs.cli h2 --steps 80 --walkers 96 --optimizer sr --device cuda
python -m autonqs.cli --config configs\h2_production_smoke.json
python examples\benchmark_suite.py --list
python examples\analyze_run.py runs\h2\metrics.jsonl --molecule h2
```

## What Is Included

- `autonqs.network.AutoNQS`: RBM-style neural quantum state with continuous
  one-electron and electron-electron visible features, hidden-unit
  `log(2 cosh)` factors, a learned phase/sign head, and Kato-style cusp/Jastrow
  terms.
- `autonqs.hamiltonian`: Coulomb Hamiltonian plus second-order autodiff kinetic
  energy, with batched local-energy evaluation for memory control.
- `autonqs.sampler`: GPU Metropolis walkers with burn-in, adaptive step size,
  and acceptance diagnostics.
- `autonqs.optim`: Adam, diagonal natural-gradient, and lightweight K-FAC-style
  preconditioning for `nn.Linear` layers.
- `autonqs.config`: JSON run configs with reproducible model, sampler,
  optimizer, checkpoint, logging, and device settings.
- `autonqs.checkpoint`: checkpoint/resume support for model, optimizer, sampler,
  history, and RNG state.
- `autonqs.benchmarks`: small validated benchmark systems: H, He, LiH, Be, H2,
  N2, and H2O.
- `autonqs.excited`: overlap and variance penalties for excited-state VMC
  experiments against checkpointed lower states.
- `autonqs.periodic`: periodic-cell helpers, position wrapping, reciprocal
  vectors, and minimum-image displacements.
- `autonqs.pseudopotentials`: simple local effective-core pseudopotential terms
  and effective nuclear charges.
- `autonqs.forces`: Hellmann-Feynman local-potential force estimates.
- `autonqs.analysis`: block statistics, reference-energy comparison, and metric
  log analysis.
- `autonqs.distributed`: `torch.distributed` initialization and DDP wrapping.
- `examples/top20_ground_state.py`: runnable examples for the 20 requested
  molecules and active-space representatives for large transition-metal systems.
- `examples/benchmark_suite.py`: benchmark runner with reference energies.

## Common Commands

Environment and verification:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -m pytest -q
```

Package and catalogs:

```cmd
python -m autonqs.cli --help
python examples\top20_ground_state.py --list
python examples\benchmark_suite.py --list
```

Training runs:

```cmd
python -m autonqs.cli h2 --steps 80 --walkers 96 --optimizer sr --device cuda
python examples\top20_ground_state.py --molecule h2o --steps 50 --device cuda
python examples\top20_ground_state.py --molecule all --steps 10 --walkers 32 --hidden 32 --hidden-density 1 --device cuda
python examples\benchmark_suite.py --benchmark h2 --steps 100 --optimizer sr --device cuda
python examples\benchmark_suite.py --benchmark all --steps 10 --walkers 32 --device cuda
```

Configs, checkpointing, logging, and analysis:

```cmd
python -m autonqs.cli --config configs\h2_production_smoke.json
python -m autonqs.cli h2 --checkpoint-path runs\h2\checkpoint.pt --checkpoint-every 50 --log-dir runs\h2 --device cuda
python -m autonqs.cli h2 --checkpoint-path runs\h2\checkpoint.pt --checkpoint-every 50 --log-dir runs\h2 --device cuda
python examples\analyze_run.py runs\h2\metrics.jsonl --molecule h2 --block-size 5
```

Physics extensions:

```cmd
python -m autonqs.cli h2 --estimate-forces --device cuda
python -m autonqs.cli --config configs\periodic_h_smoke.json
python -m autonqs.cli --config configs\h2_excited_smoke.json
```

Feature example scripts:

```cmd
python examples\force_estimation.py --steps 2 --walkers 8 --device cuda
python examples\periodic_system.py --steps 2 --walkers 8 --device cuda
python examples\pseudopotential_demo.py --steps 2 --walkers 8 --device cuda
python examples\checkpoint_resume.py --device cuda
python examples\logging_analysis.py --steps 4 --device cuda
python examples\reference_comparison.py --benchmark h2 --steps 2 --walkers 8 --device cuda
python examples\excited_states.py --steps 2 --walkers 8 --device cuda
python examples\distributed_training.py --print-command
python examples\regression_suite.py --references
python examples\convergence_benchmarks.py --case h2 --steps 25 --walkers 32 --device cuda
python examples\profiler_baseline.py --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile.json
```

Distributed smoke config:

```cmd
torchrun --standalone --nproc_per_node=1 -m autonqs.cli --config configs\ddp_h2_smoke.json
```

See [docs/commands.md](docs/commands.md) for the complete command reference.

# Benchmark Suite/Quick Verification Results

```cmd
python examples\benchmark_suite.py --benchmark all --steps 100 --walkers 64 --hidden 64 --hidden-density 4 --optimizer sr --lr 0.001 --energy-metric final --device cuda  --orbital-reference
```

| System | Calculated Energy (Ha) | Reference Energy (Ha) | Error (Ha) | Abs. Error (Ha) |
|---|---:|---:|---:|---:|
| H | -0.492925 | -0.500000 | 0.007075 | 0.007075 |
| He | -2.755503 | -2.903724 | 0.148222 | 0.148222 |
| LiH | -8.271539 | -7.882000 | -0.389539 | 0.389539 |
| Be | -2.252051 | -14.667360 | 12.415309 | 12.415309 |
| H2 | -0.892025 | -1.174475 | 0.282450 | 0.282450 |
| N2 | 17.158772 | -109.542000 | 126.700772 | 126.700772 |
| H2O | -51.818283 | -76.438000 | 24.619717 | 24.619717 |

## The Ultimate Goal

Simulating a molecule at "chemical accuracy" (generally defined as an error margin of < 1kcal/mole or ~0.04 eV) with first-principle quantum mechanics requires immense computational power. On traditional supercomputers, this high-accuracy "gold standard" (such as Coupled-Cluster theory) is usually capped at small molecules with 10 to 50 atoms.

Everyday we are inventing new ways using both Quantum Computing and Classical Computing with the goal of simulating molecules at or better than "chemical accuracy"

## References

[Neural quantum-state states for ab-initio electronic structure](paper.pdf)

[Neural network backflow for ab-initio quantum chemistry](paper2.pdf)
