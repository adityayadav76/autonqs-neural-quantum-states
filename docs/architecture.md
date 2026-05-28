# Architecture

AutoNQS is organized around a production-style VMC loop: a NQS/RBM-style
wavefunction, GPU MCMC sampling, local-energy evaluation, natural-gradient
optimization, checkpointing, benchmark reporting, force estimation, periodic
boundary helpers, pseudopotentials, excited-state penalties, and analysis.

## Molecules and Benchmarks

`autonqs.molecules` defines the 20 requested example systems. Small systems use
all electrons, while large transition-metal systems use active-electron
representatives for workstation GPU feasibility.

`autonqs.benchmarks` defines validated small benchmarks:

- H
- He
- LiH
- Be
- H2
- N2
- H2O

Each benchmark includes a reference energy, source note, and suggested tolerance.

## Network

`autonqs.network.AutoNQS` follows the NQS pattern:

- continuous electron-nucleus visible features
- continuous electron-electron pair visible features
- spin-conditioned pooling across alpha, beta, and all electrons
- residual visible-feature blocks
- RBM hidden factors using `log(2 cosh(b + Wv))`
- a learned phase/sign head for the `slog_psi` interface
- Kato-style electron-nuclear and electron-electron cusp/Jastrow log factors

`slog_psi` returns sign and `log|psi|`; `forward` returns `log|psi|`.

## Hamiltonian

`autonqs.hamiltonian` computes:

- electron-nucleus attraction
- electron-electron repulsion
- nucleus-nucleus repulsion
- kinetic energy from second derivatives of `log|psi|`

`local_energy_batched` splits walker batches to reduce peak memory use during
second-derivative autodiff.

Periodic calculations use minimum-image electron-electron, electron-nucleus,
and nucleus-nucleus displacements when a cell is attached to the model. The
current implementation is a practical supercell/minimum-image baseline, not a
full Ewald or twist-averaged periodic-electronic-structure engine.

Pseudopotentials are represented by simple local effective-core terms and
effective charges. This gives the framework a working ECP path, while
production chemistry still needs validated element-specific nonlocal
pseudopotential projectors.

## Excited States

`autonqs.excited` supports excited-state experiments through:

- checkpointed previous states
- normalized overlap penalties
- optional variance penalties

This implements the standard variational idea of penalizing overlap with lower
states. A converged production excited-state workflow still needs careful state
tracking and symmetry labels.

## Forces

`autonqs.forces` reports Hellmann-Feynman local-potential nuclear forces. These
are useful diagnostics and geometry-optimization starting points. Full VMC force
accuracy also requires Pulay terms from wavefunction geometry dependence.

## Sampler

`autonqs.sampler` provides:

- walker initialization around nuclei
- random-walk Metropolis updates
- burn-in
- adaptive step-size tuning toward a target acceptance rate
- cumulative and recent acceptance diagnostics
- log-probability refresh after model updates

## Optimizers

`autonqs.optim` provides:

- Adam
- diagonal natural-gradient optimizer
- lightweight K-FAC-style optimizer for `nn.Linear` layers

The K-FAC implementation tracks activation and gradient-output covariance
factors with PyTorch hooks and preconditions matrix gradients. Non-linear-layer
parameters use a diagonal natural-gradient fallback.

## Training Runtime

`autonqs.training.train` supports:

- reproducible seeding
- CUDA device validation
- optional `torch.nn.DataParallel`
- optional `torch.nn.parallel.DistributedDataParallel`
- MCMC burn-in and adaptation
- local-energy memory batching
- gradient clipping
- metric logging to JSONL/CSV
- checkpoint save/resume with RNG state
- block-statistics analysis and reference-energy comparison

JSON configs are represented by `autonqs.config.RunConfig`.
