#!/usr/bin/env python3
"""TRADR command-line entry point.

Examples:
  python run.py
  python run.py baseline
  python run.py iterative --rounds 10
"""
from __future__ import annotations

from tradr_config import parse_config


def main(argv: list[str] | None = None) -> None:
    config = parse_config(argv)

    # Import training only after parsing so --help and invalid input do not
    # initialize the simulator stack or require GPU-specific dependencies.
    from hover_train import baseline_train, iterative_train

    if config.mode == "baseline":
        baseline_train(
            exp_name="demo_baseline",
            algo="uniform",
            num_envs=config.num_envs,
            max_iterations=config.max_iters,
            seed=config.seed,
        )
        return

    iterative_train(
        exp_name="demo_iterative",
        algo="uniform",
        num_envs=config.num_envs,
        iterations_per_round=config.iters_per_round,
        total_rounds=config.rounds,
        num_eval_envs=config.eval_envs,
        episodes_per_eval=config.episodes_per_eval,
        eval_sample_method="sobol",
        seed=config.seed,
        target_kl=config.target_kl,
    )


if __name__ == "__main__":
    main()
