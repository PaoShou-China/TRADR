import copy
import json
import os
import pickle
import shutil
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from hover_env import HoverEnv


# ============================================================
# PPO Configuration
# ============================================================


def build_algo_cfg():
    """Build PPO algorithm configuration."""
    return {
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.008,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.0003,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 8,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
        "class_name": "PPO",
    }


def get_train_cfg(exp_name):
    algo_cfg = build_algo_cfg()
    return {
        "algorithm": algo_cfg,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 100,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


# ============================================================
# Base DR Configuration
# ============================================================

BASE_DR = {
    "randomize_mass": True, "mass_range": [0.7, 1.3],
    "randomize_inertia_axis": True, "inertia_axis_range": [0.7, 1.3],
    "randomize_com": True, "com_shift_range": [-0.008, 0.008],
    "randomize_kf": True, "kf_range": [0.7, 1.3],
    "randomize_turbulence": True,
}

# ============================================================
# Entropic Risk Reweighting: DR Distribution
# ============================================================
# q(θ) ∝ exp(−β · R(θ))
# This is an exponential tilt / entropic risk formulation:
#   q*(θ) = argmin_q E_q[R(θ)] + (1/β) KL(q || p)
# β controls sensitivity to low-reward regions.


def compute_dr_weights(rewards, beta):
    r = np.asarray(rewards, dtype=np.float64)

    # ---- Reward normalization: ensure scale invariance ----
    r = (r - r.mean()) / (r.std() + 1e-8)

    # ---- Truncate at ±3σ to curb outliers dominating the weights ----
    r = np.clip(r, -3.0, 3.0)

    logits = -beta * r
    logits -= logits.max()   # numerical stability

    w = np.exp(logits)
    w /= w.sum()

    return w


def compute_kl(q, p):
    return float(np.sum(q * (np.log(q + 1e-12) - np.log(p + 1e-12))))


def update_beta(beta, kl, target_kl, lr=0.1):
    """Negative feedback: KL > target → decrease β (distribution too peaked);
    KL < target → increase β."""
    beta = beta * np.exp(lr * (target_kl - kl))
    return float(np.clip(beta, 1e-3, 10.0))


# ============================================================
# Evaluation Helpers
# ============================================================

def evaluate_on_param_grid(env, policy, episodes_per_env=1):
    num_envs = env.num_envs
    env_reward_sums = [[] for _ in range(num_envs)]
    cumulative = torch.zeros(num_envs, device=gs.device, dtype=gs.tc_float)
    done_count = torch.zeros(num_envs, device=gs.device)

    obs_dict = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs_dict)
            obs_dict, rews, dones, _ = env.step(actions)
            cumulative += rews

            done_envs = dones.nonzero(as_tuple=False).reshape(-1)
            for idx in done_envs:
                idx_i = idx.item()
                env_reward_sums[idx_i].append(cumulative[idx].item())
                cumulative[idx] = 0.0
                done_count[idx] += 1

            if (done_count >= episodes_per_env).all():
                break

    mean_rewards = np.array([np.mean(rs) if rs else -1e9 for rs in env_reward_sums])
    dr_params_raw = env.get_dr_params()
    dr_params_np = {k: v.cpu().numpy() for k, v in dr_params_raw.items()}
    return mean_rewards, dr_params_np


def evaluate_with_trajectory(env, policy, episodes_per_env=1, record_envs=8, max_steps=500):
    """Evaluate and record state/action trajectories for visualization."""
    num_envs = env.num_envs
    record_envs = min(record_envs, num_envs)

    env_reward_sums = [[] for _ in range(num_envs)]
    cumulative = torch.zeros(num_envs, device=gs.device, dtype=gs.tc_float)
    done_count = torch.zeros(num_envs, device=gs.device)

    traj = [{
        "base_pos": [], "base_euler": [], "commands": [],
        "actions": [], "base_lin_vel": [], "base_ang_vel": [],
        "rewards": [], "dones": [],
    } for _ in range(record_envs)]
    traj_active = [True] * record_envs

    obs_dict = env.reset()
    step = 0

    with torch.no_grad():
        while True:
            actions = policy(obs_dict)
            obs_dict, rews, dones, _ = env.step(actions)
            cumulative += rews
            step += 1

            for j in range(record_envs):
                if not traj_active[j]:
                    continue
                traj[j]["base_pos"].append(env.base_pos[j].cpu().clone())
                traj[j]["base_euler"].append(env.base_euler[j].cpu().clone())
                traj[j]["commands"].append(env.commands[j].cpu().clone())
                traj[j]["actions"].append(env.actions[j].cpu().clone())
                traj[j]["base_lin_vel"].append(env.base_lin_vel[j].cpu().clone())
                traj[j]["base_ang_vel"].append(env.base_ang_vel[j].cpu().clone())
                traj[j]["rewards"].append(rews[j].cpu().clone())
                traj[j]["dones"].append(dones[j].cpu().clone())
                if dones[j]:
                    traj_active[j] = False

            done_envs = dones.nonzero(as_tuple=False).reshape(-1)
            for idx in done_envs:
                idx_i = idx.item()
                env_reward_sums[idx_i].append(cumulative[idx].item())
                cumulative[idx] = 0.0
                done_count[idx] += 1

            if (done_count >= episodes_per_env).all():
                break

            if step >= max_steps:
                break

    trajectories = []
    for j in range(record_envs):
        if len(traj[j]["base_pos"]) > 0:
            t = {k: torch.stack(v).cpu().numpy() for k, v in traj[j].items()}
            trajectories.append(t)
        else:
            trajectories.append(None)

    mean_rewards = np.array([np.mean(rs) if rs else -1e9 for rs in env_reward_sums])
    dr_params_raw = env.get_dr_params()
    dr_params_np = {k: v.cpu().numpy() for k, v in dr_params_raw.items()}
    return mean_rewards, dr_params_np, trajectories


def sample_from_discrete_q(dr_params_np, dr_weights, n_sample):
    """Sample parameters from the discrete distribution q(θ) ∝ exp(−β R).

    Pure discrete sampling — NO continuous perturbation, NO KDE, NO noise.
    Each sampled point is guaranteed to be a valid grid point.
    """
    n_pool = len(dr_weights)

    # Sample indices from categorical distribution q
    idx = np.random.choice(n_pool, size=n_sample, p=dr_weights)

    def _get(key, default_val):
        return dr_params_np[key] if key in dr_params_np else np.full(n_pool, default_val)

    mass_arr = _get("mass", 1.0)
    ixx_arr = _get("ixx", 1.0)
    iyy_arr = _get("iyy", 1.0)
    izz_arr = _get("izz", 1.0)
    com_x_arr = _get("com_x", 0.0)
    com_y_arr = _get("com_y", 0.0)
    com_z_arr = _get("com_z", 0.0)
    kf_arr = _get("kf", 1.0)

    param_list = []
    for i in idx:
        param_list.append({
            "mass_scale": float(mass_arr[i]),
            "ixx_scale": float(ixx_arr[i]),
            "iyy_scale": float(iyy_arr[i]),
            "izz_scale": float(izz_arr[i]),
            "com_shift_x": float(com_x_arr[i]),
            "com_shift_y": float(com_y_arr[i]),
            "com_shift_z": float(com_z_arr[i]),
            "kf_scale": float(kf_arr[i]),
        })

    return param_list


# ============================================================
# Sobol-based DR Parameter Grid
# ============================================================
# Use Sobol low-discrepancy sequences instead of uniform random
# to generate evaluation parameter grids. This provides more
# uniform spatial coverage and avoids "holes" in the parameter space.


def _generate_sobol_dr_params(n_samples, dr_ranges=None):
    """Generate DR parameter grid using Sobol low-discrepancy sequences.

    Sobol sequences guarantee more uniform spatial coverage than uniform
    random at a given sample size, reducing evaluation noise from
    uneven random sampling.

    Args:
        n_samples: Number of parameter points
        dr_ranges: DR dimension ranges, dict of (low, high)

    Returns:
        list of dicts, each containing DR parameters for one environment
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        # Fallback to uniform if scipy not available
        print("  [WARNING] scipy not available, falling back to uniform DR sampling")
        return None

    if dr_ranges is None:
        dr_ranges = {
            "mass": (0.7, 1.3),
            "ixx": (0.7, 1.3),
            "iyy": (0.7, 1.3),
            "izz": (0.7, 1.3),
            "com_x": (-0.008, 0.008),
            "com_y": (-0.008, 0.008),
            "com_z": (-0.008, 0.008),
            "kf": (0.7, 1.3),
        }

    dim = len(dr_ranges)
    sampler = qmc.Sobol(d=dim, scramble=True)
    n_sobol = 1
    while n_sobol < n_samples:
        n_sobol *= 2
    sobol_points = sampler.random(n_sobol)[:n_samples]  # (n_samples, dim)

    keys = list(dr_ranges.keys())
    params_list = []
    for i in range(n_samples):
        param = {}
        for j, key in enumerate(keys):
            low, high = dr_ranges[key]
            param[key] = float(low + sobol_points[i, j] * (high - low))
        params_list.append(param)

    print(f"  [Sobol] Generated {n_samples} parameter points (dim={dim}, coverage=quasi-random)")
    return params_list


def _inject_sobol_into_env(env, sobol_params):
    """Inject Sobol-generated DR parameters into the environment."""
    mapped_params = []
    key_map = {
        "mass": "mass_scale",
        "ixx": "ixx_scale",
        "iyy": "iyy_scale",
        "izz": "izz_scale",
        "com_x": "com_shift_x",
        "com_y": "com_shift_y",
        "com_z": "com_shift_z",
        "kf": "kf_scale",
    }
    for p in sobol_params:
        mapped = {key_map[k]: v for k, v in p.items()}
        mapped_params.append(mapped)
    n = env.inject_hard_params(mapped_params)
    return n


def evaluate_and_update_dr_distribution(
    policy, round_num, num_eval_envs,
    episodes_per_eval, eval_sample_method,
    beta, target_kl, beta_lr, update_beta_flag=True,
):
    """Evaluate policy and update the entropic risk DR distribution
    q(θ) ∝ exp(−β · R(θ)), with dual variable β update.

    Args:
        update_beta_flag: Whether to update β via dual ascent.
                          False keeps β constant (ablation).
        target_kl: Target KL divergence for dual ascent.

    Returns:
        dr_weights:    (N,) normalized sampling probabilities
        dr_params_np:  dict of evaluation DR parameters
        eval_rewards:  (N,) nominal reward estimates
        beta:          updated temperature parameter
        kl:            KL divergence KL(q_old || p) before β update
    """
    print(f"\n{'='*60}")
    print(f"  [Round {round_num}] ENTROPIC RISK REWEIGHTING — EVALUATION")
    print(f"  Eval envs: {num_eval_envs}  |  Sample: {eval_sample_method}")
    print(f"  Episodes per param: {episodes_per_eval} (random init state)")
    print(f"  β = {beta:.4f}  |  KL target = {target_kl}")
    print(f"{'='*60}")

    # ---- Deterministic eval: fixed seed for consistent init state / targets across rounds ----
    EVAL_SEED = 42
    torch.manual_seed(EVAL_SEED)

    dr_cfg = {**BASE_DR, "sample_method": eval_sample_method}
    eval_env = HoverEnv(num_envs=num_eval_envs, show_viewer=False, dr_cfg=dr_cfg,
                        randomize_init_state=True)

    if eval_sample_method == "sobol":
        sobol_params = _generate_sobol_dr_params(num_eval_envs)
        if sobol_params is not None:
            n_sobol = _inject_sobol_into_env(eval_env, sobol_params)
            print(f"  [Eval grid] Sobol: {n_sobol} params (low-discrepancy sequence)")
        else:
            print(f"  [Eval grid] Uniform: fallback (scipy not available)")
    else:
        print(f"  [Eval grid] Uniform: {num_eval_envs} params (HoverEnv native sampling)")

    # ---- Evaluate nominal rewards only (no variance penalty, no DRO probes) ----
    print(f"  Running {episodes_per_eval} episodes/param on {num_eval_envs} param combos ...")
    eval_rewards, dr_params_np = evaluate_on_param_grid(eval_env, policy, episodes_per_eval)

    # ---- Step 1: compute q_old with current β ----
    q_old = compute_dr_weights(eval_rewards, beta)

    # ---- Step 2: KL(q_old || p) ----
    p = np.ones_like(q_old) / len(q_old)
    kl = compute_kl(q_old, p)

    # ---- Step 3: dual ascent update β (if enabled) ----
    if update_beta_flag:
        beta = update_beta(beta, kl, target_kl, beta_lr)
    else:
        print(f"  [Ablation] β update DISABLED — keeping β = {beta:.4f}")

    # ---- Step 4: compute q_new (final distribution after β update) ----
    dr_weights = compute_dr_weights(eval_rewards, beta)

    # ---- ESS on final q (implicit entropy regularization) ----
    ess = 1.0 / (dr_weights ** 2).sum()
    if update_beta_flag and ess < 0.05 * len(dr_weights):
        beta *= 0.7
        dr_weights = compute_dr_weights(eval_rewards, beta)
        ess = 1.0 / (dr_weights ** 2).sum()
        print(f"  [Warning] ESS too low, β → {beta:.4f} (implicit entropy regularization)")

    print(f"\n  DR distribution stats (entropic):")
    print(f"    β       = {beta:.4f}")
    print(f"    KL(q_old||p) = {kl:.4f}")
    print(f"    E_q[R] = {(dr_weights * eval_rewards).sum():.4f}  (weighted mean)")
    print(f"    E_u[R] = {eval_rewards.mean():.4f}  (uniform mean)")
    print(f"    ESS    = {ess:.0f} / {num_eval_envs}  (effective sample size)")
    print(f"    w_max  = {dr_weights.max():.4f}  w_min  = {dr_weights.min():.6f}")
    print(f"    w_std  = {dr_weights.std():.4f}")

    print(f"\n  Overall evaluation stats:")
    print(f"    min={eval_rewards.min():.4f}  max={eval_rewards.max():.4f}  "
          f"mean={eval_rewards.mean():.4f}  std={eval_rewards.std():.4f}")

    del eval_env
    return dr_weights, dr_params_np, eval_rewards, beta, kl


# ============================================================
# Iterative Training (Entropic Risk Reweighting)
# ============================================================

def iterative_train(
    exp_name="drone-hovering",
    algo="uniform",
    num_envs=2048,
    num_eval_envs=10000,
    iterations_per_round=50,
    total_rounds=20,
    episodes_per_eval=15,
    eval_sample_method="sobol",
    seed=1,
    update_beta_flag=True,
    target_kl=2.0,
):
    """Entropic Risk Reweighting DR iterative training with dual β.

    q(θ) ∝ exp(−β · R(θ))

    Each round:
        Step 1: Evaluate policy → rewards
        Step 2: Compute entropic risk weights q(θ) = exp(−βR),
                then update β via dual ascent: β ← β * exp(lr * (KL − ε))
        Step 3: Sample θ ~ q(θ), inject ALL training environments
        Step 4: PPO update

    Args:
        update_beta_flag: Whether to update β via dual ascent.
                          False keeps β constant (ablation).
        target_kl: Target KL divergence (default=2.0).
    """
    log_dir = f"logs/{exp_name}_{algo}"
    train_cfg = get_train_cfg(exp_name + f"_{algo}")

    dr_cfg = {**BASE_DR, "sample_method": algo}
    dr_weights = None
    dr_params_np = None

    # ---- Dual variable state ----
    beta = 1.0
    # target_kl passed as argument (default=2.0)
    beta_lr = 0.2
    beta_history = [beta]
    kl_history = []
    eval_history = []  # record per-round evaluation reward statistics

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=seed, performance_mode=True)

    for round_idx in range(total_rounds):
        rnd = round_idx + 1
        print(f"\n{'#'*60}")
        print(f"  ITERATIVE TRAINING: Round {rnd}/{total_rounds}")
        print(f"  algo={algo}  |  iters/round={iterations_per_round}")
        print(f"  Entropic risk reweighting (dual β, target_kl={target_kl})")
        print(f"{'#'*60}")

        is_first = (round_idx == 0)

        if is_first:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/cfgs.pkl", "wb") as f:
                pickle.dump([train_cfg, dr_cfg], f)

        # ---- Create training env ----
        env = HoverEnv(num_envs=num_envs, show_viewer=False, dr_cfg=dr_cfg)

        # ---- Step 3: Sample θ ~ q(θ) from previous round's DR distribution ----
        if not is_first and dr_weights is not None and dr_params_np is not None:
            # Sample all training environments from q(θ)
            q_params = sample_from_discrete_q(dr_params_np, dr_weights, num_envs)
            print(f"  [Step 3] {num_envs} params from q(θ) (β={beta:.4f})")
            env.inject_hard_params(q_params)
        else:
            print(f"  [Step 3] Round 1: using uniform DR (no q(θ) yet)")

        # ---- Step 4: PPO update ----
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
        if is_first:
            runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)
        else:
            prev_ckpt = f"{log_dir}/model_{round_idx * iterations_per_round}.pt"
            if os.path.exists(prev_ckpt):
                runner.load(prev_ckpt, map_location="cpu")
                runner.learn(num_learning_iterations=iterations_per_round)
            else:
                print(f"  WARNING: {prev_ckpt} not found, training from scratch")
                runner.learn(num_learning_iterations=iterations_per_round, init_at_random_ep_len=True)

        ckpt_path = f"{log_dir}/model_{rnd * iterations_per_round}.pt"
        runner.save(ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

        # ---- Extract policy, then free training env/runner to avoid GPU OOM during eval ----
        policy = runner.get_inference_policy(device=gs.device)
        del env
        del runner

        # ---- Steps 1 & 2: Evaluate updated policy, compute entropic risk DR distribution + update β ----
        dr_weights, dr_params_np, eval_rewards, beta, kl = evaluate_and_update_dr_distribution(
            policy=policy,
            round_num=rnd,
            num_eval_envs=num_eval_envs,
            episodes_per_eval=episodes_per_eval,
            eval_sample_method=eval_sample_method,
            beta=beta, target_kl=target_kl, beta_lr=beta_lr,
            update_beta_flag=update_beta_flag,
        )
        beta_history.append(beta)
        kl_history.append(kl)

        # ---- Record evaluation statistics for this round (summary only, not full array) ----
        eval_history.append({
            "round": rnd,
            "mean": float(eval_rewards.mean()),
            "std": float(eval_rewards.std()),
            "min": float(eval_rewards.min()),
            "max": float(eval_rewards.max()),
            "median": float(np.median(eval_rewards)),
            "beta": beta,
            "kl": kl,
        })

    # ---- Save evaluation history to file ----
    save_path_pkl = f"{log_dir}/eval_history.pkl"
    with open(save_path_pkl, "wb") as f:
        pickle.dump(eval_history, f)
    # Also save JSON version
    save_path_json = f"{log_dir}/eval_history.json"
    with open(save_path_json, "w") as f:
        json.dump(eval_history, f, indent=2)
    print(f"\n  Evaluation history saved → {save_path_pkl} + {save_path_json}  ({len(eval_history)} rounds)")

    print(f"\n{'='*60}")
    print(f"  ITERATIVE TRAINING COMPLETE: {total_rounds} rounds finished")
    print(f"  Final model: {log_dir}/model_{total_rounds * iterations_per_round}.pt")
    print(f"\n  β trajectory: {[round(b, 4) for b in beta_history]}")
    print(f"  KL trajectory: {[round(k, 4) for k in kl_history]}")
    print(f"{'='*60}")


# ============================================================
# Baseline Training
# ============================================================

def baseline_train(exp_name="drone-hovering", algo="uniform", num_envs=2048,
                   max_iterations=1000, seed=1, no_dr=False,
                   eval_interval=50, num_eval_envs=10000, episodes_per_eval=1):
    log_dir = f"logs/{exp_name}_{algo}_baseline"
    train_cfg = get_train_cfg(exp_name + f"_{algo}_baseline")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if no_dr:
        dr_cfg = None  # No DR: use nominal parameters only
    else:
        dr_cfg = {**BASE_DR, "sample_method": algo}
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([train_cfg, dr_cfg], f)

    if not gs._initialized:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=seed, performance_mode=True)

    dr_label = "NO DR (Nominal)" if no_dr else f"{algo.upper()} DR"
    print(f"\n{'#'*60}")
    print(f"  BASELINE TRAINING: {dr_label}  |  {max_iterations} iters")
    print(f"  Periodic eval every {eval_interval} iters on {num_eval_envs} DR combos")
    print(f"{'#'*60}")

    # ---- Eval DR config: always use uniform DR to evaluate generalization ----
    eval_dr_cfg = {**BASE_DR, "sample_method": "uniform"}
    EVAL_SEED = 42

    env = HoverEnv(num_envs=num_envs, show_viewer=False, dr_cfg=dr_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    eval_history = []
    num_evals = (max_iterations + eval_interval - 1) // eval_interval

    for eval_idx in range(num_evals):
        start_iter = eval_idx * eval_interval
        n_iter = min(eval_interval, max_iterations - start_iter)

        if eval_idx == 0:
            runner.learn(num_learning_iterations=n_iter, init_at_random_ep_len=True)
        else:
            prev_ckpt = f"{log_dir}/model_{start_iter}.pt"
            if os.path.exists(prev_ckpt):
                runner.load(prev_ckpt)
                runner.learn(num_learning_iterations=n_iter)
            else:
                print(f"  WARNING: {prev_ckpt} not found, training from scratch")
                runner.learn(num_learning_iterations=n_iter, init_at_random_ep_len=True)

        curr_iter = start_iter + n_iter
        runner.save(f"{log_dir}/model_{curr_iter}.pt")

        # ---- Evaluate current policy ----
        policy = runner.get_inference_policy(device=gs.device)

        torch.manual_seed(EVAL_SEED)
        eval_env = HoverEnv(num_envs=num_eval_envs, show_viewer=False,
                            dr_cfg=eval_dr_cfg, randomize_init_state=True)
        eval_rewards, _ = evaluate_on_param_grid(eval_env, policy, episodes_per_eval)
        del eval_env

        eval_history.append({
            "iter": curr_iter,
            "mean": float(eval_rewards.mean()),
            "std": float(eval_rewards.std()),
            "min": float(eval_rewards.min()),
            "max": float(eval_rewards.max()),
            "median": float(np.median(eval_rewards)),
        })

        print(f"\n  [Eval @ iter {curr_iter}]  mean={eval_rewards.mean():.4f}  "
              f"std={eval_rewards.std():.4f}  min={eval_rewards.min():.4f}  "
              f"max={eval_rewards.max():.4f}")

        del policy

    # ---- Save evaluation history ----
    save_path_pkl = f"{log_dir}/eval_history.pkl"
    with open(save_path_pkl, "wb") as f:
        pickle.dump(eval_history, f)
    # Also save JSON version
    save_path_json = f"{log_dir}/eval_history.json"
    with open(save_path_json, "w") as f:
        json.dump(eval_history, f, indent=2)
    print(f"\n  Evaluation history saved → {save_path_pkl} + {save_path_json}  ({len(eval_history)} evals)")

    print(f"\n  BASELINE COMPLETE → {log_dir}/model_{max_iterations}.pt")


# ============================================================
# Policy Loading
# ============================================================

def load_policy(log_dir, ckpt=None):
    """Load a trained policy from a log directory."""
    cfg_path = f"{log_dir}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        return None

    if ckpt is None:
        model_files = sorted(
            [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
            key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
        )
        if not model_files:
            return None
        model_path = os.path.join(log_dir, model_files[-1])
        used_ckpt = int(model_files[-1].replace("model_", "").replace(".pt", ""))
    else:
        model_path = f"{log_dir}/model_{ckpt}.pt"
        used_ckpt = ckpt
        if not os.path.exists(model_path):
            model_files = sorted(
                [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")],
                key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
            )
            if not model_files:
                return None
            model_path = os.path.join(log_dir, model_files[-1])
            used_ckpt = int(model_files[-1].replace("model_", "").replace(".pt", ""))

    with open(cfg_path, "rb") as f:
        loaded = pickle.load(f)
        train_cfg = loaded[0] if isinstance(loaded, list) else loaded

    alg = train_cfg.get("algorithm", {})
    if "class_name" not in alg:
        train_cfg.setdefault("algorithm", {})["class_name"] = "PPO"

    return train_cfg, model_path, used_ckpt
