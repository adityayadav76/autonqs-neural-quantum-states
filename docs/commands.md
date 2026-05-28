# Command Reference

Run these from the repository root:

```cmd
cd D:\Coding-Agents\workspace\autonqs
```

## Environment

Check the PyTorch and CUDA environment:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Install Python dependencies with pip if needed:

```cmd
pip install -r requirements.txt
pip install -e .
```

## Tests

Run the full test suite:

```cmd
python -m pytest -q
```

Print the reference table used by regression checks:

```cmd
python examples\regression_suite.py --references
```

## CLI Help

Show all CLI options:

```cmd
python -m autonqs.cli --help
```

## Molecule Catalog

List all 20 requested molecule examples:

```cmd
python examples\top20_ground_state.py --list
```

Run one catalog molecule:

```cmd
python examples\top20_ground_state.py --molecule h2o --steps 50 --walkers 64 --device cuda
```

Run all 20 catalog molecules as a short smoke suite:

```cmd
python examples\top20_ground_state.py --molecule all --steps 10 --walkers 32 --hidden 32 --hidden-density 1 --device cuda
```

## Direct Training

Train H2 with the default CLI path:

```cmd
python -m autonqs.cli h2 --steps 80 --walkers 96 --hidden 64 --layers 2 --hidden-density 4 --device cuda
```

Train with diagonal natural gradient:

```cmd
python -m autonqs.cli h2 --steps 80 --walkers 96 --optimizer sr --device cuda
```

Train with K-FAC-style preconditioning:

```cmd
python -m autonqs.cli h2 --steps 80 --walkers 96 --optimizer sr --device cuda
```

Use local-energy memory batching:

```cmd
python -m autonqs.cli h2o --steps 50 --walkers 128 --energy-batch-size 32 --device cuda
```

Use float64 local-energy numerics:

```cmd
python -m autonqs.cli h2 --steps 50 --walkers 64 --dtype float64 --device cuda
```

## Config Runs

Production-style H2 smoke config:

```cmd
python -m autonqs.cli --config configs\h2_production_smoke.json
```

Excited-state smoke config:

```cmd
python -m autonqs.cli --config configs\h2_excited_smoke.json
```

Periodic minimum-image supercell smoke config:

```cmd
python -m autonqs.cli --config configs\periodic_h_smoke.json
```

Single-process DDP smoke config:

```cmd
torchrun --standalone --nproc_per_node=1 -m autonqs.cli --config configs\ddp_h2_smoke.json
```

Multi-GPU DDP when multiple CUDA devices are available:

```cmd
torchrun --standalone --nproc_per_node=2 -m autonqs.cli --config configs\ddp_h2_smoke.json
```

## Checkpoint And Resume

Create checkpoints and logs:

```cmd
python -m autonqs.cli h2 --steps 100 --checkpoint-path runs\h2\checkpoint.pt --checkpoint-every 50 --log-dir runs\h2 --device cuda
```

Resume from the same checkpoint path:

```cmd
python -m autonqs.cli h2 --steps 150 --checkpoint-path runs\h2\checkpoint.pt --checkpoint-every 50 --log-dir runs\h2 --device cuda
```

## Logging And Analysis

Analyze a JSONL metrics log:

```cmd
python examples\analyze_run.py runs\h2\metrics.jsonl --molecule h2 --block-size 5
```

Metric outputs are written to:

```text
runs\h2\metrics.jsonl
runs\h2\metrics.csv
```

## Benchmark Suite

List reference-energy benchmarks:

```cmd
python examples\benchmark_suite.py --list
```

Run H2 benchmark:

```cmd
python examples\benchmark_suite.py --benchmark h2 --steps 100 --walkers 96 --optimizer sr --device cuda
```

Run all benchmarks as a short smoke suite:

```cmd
python examples\benchmark_suite.py --benchmark all --steps 10 --walkers 32 --hidden 32 --hidden-density 1 --device cuda
```

## Regression And Convergence

Run deterministic smoke regressions against known atoms/molecules:

```cmd
python examples\regression_suite.py --case h --case h2 --case he --steps 5 --walkers 16 --device cuda
```

Run a configurable convergence benchmark with a pass/fail threshold:

```cmd
python examples\convergence_benchmarks.py --case h2 --steps 25 --walkers 32 --hidden 32 --hidden-density 2 --optimizer sr --max-error 5 --device cuda --output runs\benchmarks\h2_convergence.json
```

Use CPU for quick deterministic smoke checks:

```cmd
python examples\convergence_benchmarks.py --case h2 --steps 1 --walkers 4 --hidden 8 --hidden-density 1 --max-error 20 --device cpu
```

## Profiler Baselines

Create a profiler baseline:

```cmd
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile.json
```

Compare a new run against an existing baseline:

```cmd
python examples\profiler_baseline.py --molecule h2 --steps 5 --walkers 16 --device cuda --output runs\profiler\h2_profile_new.json --baseline runs\profiler\h2_profile.json --max-slowdown 1.25
```

## Physics Extensions

Estimate nuclear forces:

```cmd
python -m autonqs.cli h2 --steps 20 --walkers 64 --estimate-forces --device cuda
python examples\force_estimation.py --steps 2 --walkers 8 --device cuda
```

Run periodic supercell support:

```cmd
python -m autonqs.cli --config configs\periodic_h_smoke.json
python examples\periodic_system.py --steps 2 --walkers 8 --device cuda
```

Run excited-state overlap-penalty support:

```cmd
python -m autonqs.cli --config configs\h2_excited_smoke.json
python examples\excited_states.py --steps 2 --walkers 8 --device cuda
```

Run local pseudopotential support:

```cmd
python examples\pseudopotential_demo.py --molecule h2o --steps 2 --walkers 8 --device cuda
```

## Example Scripts

Run the H2 accuracy smoke:

```cmd
python examples\h2_accuracy_smoke.py
```

Run benchmark listing:

```cmd
python examples\benchmark_suite.py --list
```

Run top-20 molecule listing:

```cmd
python examples\top20_ground_state.py --list
```

Run checkpoint/resume example:

```cmd
python examples\checkpoint_resume.py --device cuda
```

Run logging plus analysis example:

```cmd
python examples\logging_analysis.py --steps 4 --device cuda
```

Run reference comparison example:

```cmd
python examples\reference_comparison.py --benchmark h2 --steps 2 --walkers 8 --device cuda
```

Print distributed example launch command:

```cmd
python examples\distributed_training.py --print-command
```
