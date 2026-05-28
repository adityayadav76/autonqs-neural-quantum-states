from __future__ import annotations

import argparse
import json

from .config import load_config
from .molecules import get_molecule, list_molecules
from .training import TrainConfig, train, train_from_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AutoNQS ground-state VMC models.")
    parser.add_argument("molecule", nargs="?", default="h2", choices=list_molecules())
    parser.add_argument("--config", help="JSON run config. CLI molecule/options are ignored when set.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-density", type=int, default=4, help="RBM hidden units per electron when --rbm-hidden is unset.")
    parser.add_argument("--rbm-hidden", type=int, default=0, help="Explicit number of RBM hidden units.")
    parser.add_argument("--orbital-reference", action="store_true", help="Multiply the NQS correlator by an antisymmetric orbital reference.")
    parser.add_argument("--no-backflow", action="store_true", help="Disable neural backflow coordinates for orbital references.")
    parser.add_argument("--optimizer", default="sr", choices=["adam", "natural", "sr", "diag-natural", "diagonal-natural", "kfac"])
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--target-acceptance", type=float, default=0.6)
    parser.add_argument("--no-adapt-mcmc", action="store_true")
    parser.add_argument("--walker-move-mcmc", action="store_true", help="Use whole-walker proposals instead of electron-wise sweeps.")
    parser.add_argument("--energy-batch-size", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--estimate-forces", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.config:
        result = train_from_run_config(load_config(args.config))
    else:
        cfg = TrainConfig(
            steps=args.steps,
            walkers=args.walkers,
            hidden=args.hidden,
            layers=args.layers,
            hidden_density=args.hidden_density,
            rbm_hidden=args.rbm_hidden or None,
            orbital_reference=args.orbital_reference,
            backflow=not args.no_backflow,
            optimizer=args.optimizer,
            burn_in=args.burn_in,
            mcmc_steps=args.mcmc_steps,
            step_size=args.step_size,
            target_acceptance=args.target_acceptance,
            adapt_mcmc=not args.no_adapt_mcmc,
            electron_wise_mcmc=not args.walker_move_mcmc,
            energy_batch_size=args.energy_batch_size,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            log_dir=args.log_dir,
            data_parallel=args.data_parallel,
            estimate_forces=args.estimate_forces,
            device=args.device,
            dtype=args.dtype,
            deterministic=args.deterministic,
        )
        result = train(get_molecule(args.molecule), cfg)
    result.pop("model")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
