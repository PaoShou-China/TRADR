#!/usr/bin/env python3
"""
DR_DPPO Minimal Demo — Entropic Risk Reweighting for Domain Randomization.

Two training modes:
  baseline   — Standard uniform DR training (PPO baseline)
  iterative  — Entropic Risk Reweighting DR (our method, dual β + Sobol evaluation)

Usage:
  python run.py                        # Default: iterative mode
  python run.py baseline               # Baseline training
  python run.py iterative --rounds 10  # Custom round count
"""

import argparse
from hover_train import baseline_train, iterative_train


def main():
    parser = argparse.ArgumentParser(
        description="DR_DPPO — Entropic Risk Reweighting DR minimal demo"
    )
    parser.add_argument(
        "mode", nargs="?", default="iterative",
        choices=["baseline", "iterative"],
        help="Training mode: baseline (Uniform DR) | iterative (Entropic Risk Reweighting, default)",
    )
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)

    # Baseline parameters
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="Number of PPO iterations for baseline training")

    # Iterative training parameters
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of iterative training rounds")
    parser.add_argument("--iters_per_round", type=int, default=50,
                        help="PPO iterations per round")
    parser.add_argument("--eval_envs", type=int, default=10000,
                        help="Evaluation parameter grid size")
    parser.add_argument("--episodes_per_eval", type=int, default=15,
                        help="Episodes per evaluation parameter point")
    parser.add_argument("--target_kl", type=float, default=2.0,
                        help="Target KL divergence value")

    args = parser.parse_args()

    if args.mode == "baseline":
        baseline_train(
            exp_name="demo_baseline",
            algo="uniform",
            num_envs=args.num_envs,
            max_iterations=args.max_iters,
            seed=args.seed,
        )
    else:
        iterative_train(
            exp_name="demo_iterative",
            algo="uniform",
            num_envs=args.num_envs,
            iterations_per_round=args.iters_per_round,
            total_rounds=args.rounds,
            num_eval_envs=args.eval_envs,
            episodes_per_eval=args.episodes_per_eval,
            eval_sample_method="sobol",
            seed=args.seed,
            target_kl=args.target_kl,
        )


if __name__ == "__main__":
    main()
