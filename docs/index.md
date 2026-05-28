# AutoNQS Documentation

AutoNQS is a PyTorch framework for NQS/RBM-style variational Monte Carlo
ground-state energy experiments. It runs on CUDA GPUs and includes production
features such as adaptive MCMC, natural-gradient optimization, checkpointing,
configs, logging, benchmark systems, excited states, periodic cells,
pseudopotentials, force estimates, distributed training hooks, and result
analysis, regression suites, convergence benchmarks, and profiler baselines.

## Documentation Map

- [Installation](installation.md): environment setup and GPU checks.
- [Architecture](architecture.md): model, sampler, Hamiltonian, and trainer.
- [Algorithm Audit](algorithm_audit.md): algorithmic problems found and fixed.
- [Examples](examples.md): command-line usage and the 20 molecule catalog.
- [Commands](commands.md): complete command reference for running everything.
- [Testing](testing.md): verification commands and expected smoke-test behavior.

## Core Workflow

1. Pick a molecule from the built-in catalog.
2. Build an `AutoNQS` wavefunction with continuous visible features and RBM hidden factors.
3. Burn in adaptive GPU Metropolis walkers near nuclei.
4. Estimate local energies using batched PyTorch autodiff.
5. Optimize with Adam, diagonal natural gradient, or K-FAC-style preconditioning.
6. Save checkpoints and logs for reproducibility.
7. Compare block-averaged results against references and optional force reports.

The framework reports stochastic checkpoint energies, acceptance rates, runtime,
steps per second, final energy, and best observed checkpoint energy.
