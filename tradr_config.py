"""Command-line configuration for the TRADR demonstration.

Keeps user-facing defaults and argument validation out of the training code.
The defaults mirror the values previously defined by run.py.
"""
from __future__ import annotations

import argparse
import math
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
        integer_fields = ("num_envs", "seed", "max_iters", "rounds",
                          "iters_per_round", "eval_envs", "episodes_per_eval")
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name.replace('_', '-')} must be an integer")
            if name == "seed":
                if not 0 <= value <= 2**32 - 1:
                    raise ValueError("seed must be between 0 and 4294967295")
            elif value <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
        if not isinstance(self.target_kl, (int, float)) or not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("target-kl must be a finite positive number")
        return self


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRADR — entropic risk reweighting for domain randomization"
    )
    defaults = RunConfig()
    parser.add_argument(
        "mode", nargs="?", default=defaults.mode, choices=("baseline", "iterative"),
        help="Training mode: uniform baseline or iterative entropic reweighting",
    )
    parser.add_argument("--num_envs", type=int, default=defaults.num_envs,
                        help="Parallel training environments")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="Training seed (0..4294967295)")
    parser.add_argument("--max_iters", type=int, default=defaults.max_iters,
                        help="PPO iterations for baseline training")
    parser.add_argument("--rounds", type=int, default=defaults.rounds,
                        help="Iterative training rounds")
    parser.add_argument("--iters_per_round", type=int, default=defaults.iters_per_round,
                        help="PPO iterations per round")
    parser.add_argument("--eval_envs", type=int, default=defaults.eval_envs,
                        help="Evaluation parameter grid size")
    parser.add_argument("--episodes_per_eval", type=int, default=defaults.episodes_per_eval,
                        help="Episodes per evaluation parameter point")
    parser.add_argument("--target_kl", type=float, default=defaults.target_kl,
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
