# Testing

Run the test suite from the repository root:

```cmd
python -m pytest -q
```

Current tests cover:

- all 20 requested molecule specs
- electron and spin counts
- same-spin exchange antisymmetry
- finite local energies
- Metropolis sampler behavior
- adaptive sampler diagnostics
- benchmark catalog and reference energies
- minimum-image periodic helpers
- local pseudopotential terms
- force estimator shape and finiteness
- block analysis and reference comparison
- reference table validation for known atoms/molecules
- deterministic reproducibility controls
- convergence benchmark pass/fail surfaces
- profiler baseline comparison
- JSON config parsing
- natural-gradient optimizer construction
- short CUDA training when CUDA is available

## GPU Verification

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If CUDA is unavailable, CUDA-specific tests are skipped. If `--device cuda` is
requested while CUDA is unavailable, autonqs raises a runtime error.

## Smoke Commands Used During Development

```cmd
python -m autonqs.cli h2 --steps 3 --walkers 8 --hidden 16 --layers 1 --hidden-density 1 --optimizer sr --burn-in 2 --mcmc-steps 2 --energy-batch-size 4 --device cuda
python -m autonqs.cli h2 --steps 2 --walkers 6 --hidden 12 --layers 1 --hidden-density 1 --optimizer sr --burn-in 1 --mcmc-steps 1 --device cuda
python examples\benchmark_suite.py --list
python -m autonqs.cli h2 --steps 2 --walkers 6 --hidden 12 --layers 1 --hidden-density 1 --burn-in 1 --mcmc-steps 1 --estimate-forces --device cuda
python -m autonqs.cli --config configs\periodic_h_smoke.json
python examples\regression_suite.py --references
python examples\convergence_benchmarks.py --case h2 --steps 1 --walkers 4 --hidden 8 --hidden-density 1 --max-error 20 --device cpu
python examples\profiler_baseline.py --steps 1 --walkers 4 --device cpu --output runs\profiler\test_profile.json
```

## Regression Tests

```cmd
python examples\regression_suite.py --references
python examples\regression_suite.py --case h --case h2 --case he --steps 5 --walkers 16 --device cuda
```

## Convergence Benchmarks

```cmd
python examples\convergence_benchmarks.py --case h2 --steps 25 --walkers 32 --hidden 32 --hidden-density 2 --optimizer sr --max-error 5 --device cuda --output runs\benchmarks\h2_convergence.json
```

## Profiler Baselines

```cmd
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile.json
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile_new.json --baseline runs\profiler\h2_profile.json --max-slowdown 1.25
```

PyTorch may emit a full-backward-hook warning during K-FAC runs for modules
whose inputs do not require gradients. The optimizer still records factors and
the smoke run completes.

## Accuracy Expectations

Short smoke tests verify execution, not convergence. For real experiments:

- increase `--steps`
- increase `--walkers`
- increase `--hidden-density`
- use `--optimizer sr` or `--optimizer sr`
- use `--dtype float64` for stricter local-energy numerics
- monitor acceptance, variance, and checkpointed best energy
- compare against `examples\benchmark_suite.py` reference energies
