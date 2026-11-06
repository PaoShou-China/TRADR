"""Command-line configuration for the TRADR demonstration.

Keeps user-facing defaults and argument validation out of the training code.
The defaults mirror the values previously defined by run.py.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    mode: str = "iterative"
    num_envs: int = 2048
    seed: int = 1
    max_iters: int = 1000
    rounds: int = 20
    iters_per_round: int = 50
    eval_envs: int = 10000
    episodes_per_eval: int = 15
    target_kl: float = 2.0

    def validate(self) -> "RunConfig":
        for name in ("num_envs", "max_iters", "rounds", "iters_per_round",
                     "eval_envs", "episodes_per_eval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
        if self.target_kl <= 0:
            raise ValueError("target-kl must be positive")
        return self


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRADR — entropic risk reweighting for domain randomization"
    )
    parser.add_argument(
        "mode", nargs="?", default="iterative", choices=("baseline", "iterative"),
        help="Training mode: uniform baseline or iterative entropic reweighting",
    )
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="Parallel training environments")
    parser.add_argument("--seed", type=int, default=1, help="Training seed")
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="PPO iterations for baseline training")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Iterative training rounds")
    parser.add_argument("--iters_per_round", type=int, default=50,
                        help="PPO iterations per round")
    parser.add_argument("--eval_envs", type=int, default=10000,
                        help="Evaluation parameter grid size")
    parser.add_argument("--episodes_per_eval", type=int, default=15,
                        help="Episodes per evaluation parameter point")
    parser.add_argument("--target_kl", type=float, default=2.0,
                        help="Target KL divergence for beta update")
    return parser


def parse_config(argv: list[str] | None = None) -> RunConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RunConfig(**vars(args))
    try:
        return config.validate()
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse parser.error() always exits")
