# Examples

## List the 20 Molecules

```powershell
python examples\top20_ground_state.py --list
```

## Train H2 With K-FAC-Style Preconditioning

```powershell
python -m autonqs.cli h2 --steps 80 --walkers 96 --hidden 64 --layers 2 --hidden-density 4 --optimizer sr --device cuda
```

## Train From JSON Config

```powershell
python -m autonqs.cli --config configs\h2_production_smoke.json
```

The sample config enables adaptive MCMC, K-FAC-style optimization, checkpointing,
energy batching, and explicit network settings.

## Checkpoint and Logging

```powershell
python -m autonqs.cli h2 --steps 100 --checkpoint-path runs\h2\checkpoint.pt --checkpoint-every 50 --log-dir runs\h2 --device cuda
```

Metrics are written to:

- `runs\h2\metrics.jsonl`
- `runs\h2\metrics.csv`

The checkpoint includes model weights, optimizer state, sampler state, history,
and RNG state.

Running the same command with the same checkpoint path resumes from the saved
step when the checkpoint exists.

## Benchmark Suite

```powershell
python examples\benchmark_suite.py --list
python examples\benchmark_suite.py --benchmark h2 --steps 100 --optimizer sr --device cuda
```

Available benchmarks are H, He, LiH, Be, H2, N2, and H2O.

## Excited-State Experiment

Train a ground state first and save it:

```powershell
python -m autonqs.cli h2 --checkpoint-path runs\h2_ground\checkpoint.pt --checkpoint-every 50 --device cuda
```

Then run an excited-state config with an overlap penalty against that checkpoint:

```powershell
python -m autonqs.cli --config configs\h2_excited_smoke.json
```

## Periodic Supercell Smoke

```powershell
python -m autonqs.cli --config configs\periodic_h_smoke.json
```

`periodic_cell` values are given in Angstrom in JSON configs and converted to
Bohr internally.

## Force Estimation

```powershell
python -m autonqs.cli h2 --steps 20 --estimate-forces --device cuda
python examples\force_estimation.py --steps 2 --walkers 8 --device cuda
```

The force report uses the Hellmann-Feynman local-potential estimator.

## Pseudopotentials

```powershell
python examples\pseudopotential_demo.py --molecule h2o --steps 2 --walkers 8 --device cuda
```

This runs a local effective-core pseudopotential example with per-nucleus
pseudopotential labels.

## Result Analysis

```powershell
python examples\analyze_run.py runs\h2\metrics.jsonl --molecule h2 --block-size 5
python examples\logging_analysis.py --steps 4 --device cuda
python examples\reference_comparison.py --benchmark h2 --steps 2 --walkers 8 --device cuda
```

This prints block statistics and reference-energy comparison when the molecule
has a built-in benchmark.

## Regression And Convergence

```powershell
python examples\regression_suite.py --references
python examples\regression_suite.py --case h --case h2 --case he --steps 5 --walkers 16 --device cuda
python examples\convergence_benchmarks.py --case h2 --steps 25 --walkers 32 --hidden 32 --hidden-density 2 --optimizer sr --max-error 5 --device cuda --output runs\benchmarks\h2_convergence.json
```

The regression suite checks known atom/molecule metadata and deterministic
smoke thresholds. The convergence runner is intended for longer, budgeted
accuracy jobs with explicit pass/fail thresholds.

## Profiler Baseline

```powershell
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile.json
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile_new.json --baseline runs\profiler\h2_profile.json --max-slowdown 1.25
```

Profiler baselines record wall time, training time, steps per second, parameter
count, device metadata, and CUDA peak memory when available.

## Memory Controls

Use local-energy batching when second-order autodiff is too memory hungry:

```powershell
python -m autonqs.cli h2o --walkers 128 --energy-batch-size 32 --device cuda
```

For machines with multiple CUDA devices:

```powershell
python -m autonqs.cli h2 --data-parallel --device cuda
```

DataParallel is optional; single-GPU execution remains the default.

For process-based distributed runs, launch with `torchrun` and set
`distributed.enabled` in the JSON config. autonqs initializes `torch.distributed`
from environment variables and wraps the model with DDP.

```powershell
python examples\distributed_training.py --print-command
torchrun --standalone --nproc_per_node=1 examples\distributed_training.py
```

## Checkpoint Resume

```powershell
python examples\checkpoint_resume.py --device cuda
```

This creates a checkpoint in `runs\examples\checkpoint_resume\...`, resumes
from it, and reports the final resumed step.
