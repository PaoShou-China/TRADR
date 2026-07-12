# Tail-Risk-Aware Domain Randomization

Minimal demo of iterative **Entropic Risk Reweighting** for Domain Randomization (DR) in reinforcement learning. A quadrotor hover task simulated with the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) GPU physics engine, trained with PPO.

## Core Idea

Standard DR samples dynamics parameters uniformly — treating easy and hard cases equally.  
We use **Entropic Risk Reweighting**: after each training round, evaluate the policy on a parameter grid, then compute a sampling distribution

```
q(θ) ∝ exp(−β · R(θ))
```

that up-weights low-reward (hard) parameter regions. A **dual variable β** is updated via dual ascent to keep KL(q || uniform) near a target value, preventing distribution collapse.


## Dependencies

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) (GPU physics engine)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO implementation)
- PyTorch, NumPy, SciPy

## Usage

```bash
# Default: Entropic Risk Reweighting (dual β + Sobol evaluation grid)
python run.py

# Baseline: Standard uniform DR training
python run.py baseline

# Custom parameters
python run.py --rounds 10 --iters_per_round 50 --num_envs 4096
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `iterative` | `baseline` (Uniform DR) or `iterative` (Entropic Risk Reweighting) |
| `--num_envs` | 2048 | Number of parallel training environments |
| `--seed` | 1 | Random seed |
| `--rounds` | 20 | Iterative training rounds |
| `--iters_per_round` | 50 | PPO iterations per round |
| `--eval_envs` | 10000 | Evaluation parameter grid size |
| `--episodes_per_eval` | 15 | Episodes per evaluation parameter point |
| `--target_kl` | 2.0 | Target KL divergence for dual β update |

## DR Parameters
Multiple dynamics parameters are randomized over predefined ranges, including mass, inertia, center-of-mass offset, thrust coefficient, and turbulence wind disturbance.

## Output

Checkpoints and evaluation logs are saved under `logs/`:

```
logs/demo_iterative_uniform/
├── model_50.pt         # Checkpoint after round 1
├── model_100.pt        # Checkpoint after round 2
├── ...
├── eval_history.json   # Per-round evaluation statistics
└── eval_history.pkl
```
