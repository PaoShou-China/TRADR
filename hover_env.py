import torch
import math
import copy
from tensordict import TensorDict

import genesis as gs
from genesis.engine.force_fields import Turbulence
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import qd_to_torch


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HoverEnv:
    """Quadrotor hover environment with domain randomization.

    Supports GPU-parallel simulation with Genesis physics engine.
    DR parameters: mass, inertia (per-axis), center-of-mass shift,
    thrust coefficient, and turbulence wind.
    """
    def __init__(self, num_envs, visualize_camera=False, show_viewer=False, dr_cfg=None,
                 randomize_init_state=False):
        self.num_envs = num_envs
        self.dt = 0.01
        self.dr_cfg = dr_cfg
        self.rendered_env_num = min(10, num_envs)
        self.device = gs.device
        self._randomize_init_state = randomize_init_state

        self.env_cfg = {
            "visualize_target": True if show_viewer else False,
            "visualize_camera": visualize_camera,
            "max_visualize_FPS": 60,
            "num_actions": 4,
            "termination_if_roll_greater_than": 180,
            "termination_if_pitch_greater_than": 180,
            "termination_if_close_to_ground": 0.1,
            "termination_if_x_greater_than": 3.0,
            "termination_if_y_greater_than": 3.0,
            "termination_if_z_greater_than": 2.0,
            "base_init_pos": [0.0, 0.0, 1.0],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "episode_length_s": 15.0,
            "at_target_threshold": 0.1,
            "resampling_time_s": 3.0,
            "simulate_action_latency": True,
            "clip_actions": 1.0,
        }
        self.num_actions = self.env_cfg["num_actions"]
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)

        self.obs_scales = {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / math.pi,
        }
        self.reward_cfg = {
            "yaw_lambda": -10.0,
            "reward_scales": {
                "target": 10.0,
                "smooth": -1e-4,
                "yaw": 0.01,
                "angular": -2e-4,
                "crash": -10.0,
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "pos_x_range": [-1.0, 1.0],
            "pos_y_range": [-1.0, 1.0],
            "pos_z_range": [1.0, 1.0],
        }
        self.num_commands = self.command_cfg["num_commands"]
        self.reward_scales = copy.deepcopy(self.reward_cfg["reward_scales"])

        self._build_scene(show_viewer)
        self._init_buffers()

        if self.dr_cfg is not None:
            self._pre_sample_dr_params()
            self._randomize_dynamics(torch.arange(self.num_envs, device=gs.device))

        self._init_rewards()
        self.cfg = self.env_cfg
        self.reset()

    # ============================================================
    # Scene & Buffers
    # ============================================================

    def _build_scene(self, show_viewer):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5)),
                ),
            )
        else:
            self.target = None

        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # ---- Turbulence force field (wind disturbance) ----
        if self.dr_cfg is not None and self.dr_cfg.get("randomize_turbulence", False):
            turb_strength = self.dr_cfg.get("turbulence_strength", 0.08)
            self.scene.add_force_field(Turbulence(strength=turb_strength, frequency=1.0))

        self.scene.build(n_envs=self.num_envs)

        if self.dr_cfg is not None:
            self._base_link_idx = self.drone.get_link("base_link").idx_local
            self._nominal_mass = self.drone.get_link("base_link").get_mass()
            self._prev_inertia_scales = torch.ones((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
            self._base_link_global = self._base_link_idx + self.drone._link_start

        self._kf_scales = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _init_buffers(self):
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = {}

    def _init_rewards(self):
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    # ============================================================
    # Command Sampling
    # ============================================================

    def _resample_commands(self, envs_idx):
        def sample_range(key):
            r = self.command_cfg[key]
            return gs_rand_float(r[0], r[1], (len(envs_idx),), gs.device)

        self.commands[envs_idx, 0] = sample_range("pos_x_range")
        self.commands[envs_idx, 1] = sample_range("pos_y_range")
        self.commands[envs_idx, 2] = sample_range("pos_z_range")

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    # ============================================================
    # Step
    # ============================================================

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        base_rpm = (1 + self.actions * 0.8) * 14468.429183500699
        self.drone.set_propellers_rpm(base_rpm * torch.sqrt(self._kf_scales).unsqueeze(-1))
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._update_observation()
        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _update_observation(self):
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

    # ============================================================
    # Domain Randomization
    # ============================================================

    def _pre_sample_dr_params(self):
        n = self.num_envs

        def sample_1d(lower, upper):
            return gs_rand_float(lower, upper, (n,), gs.device)

        def sample_if(key, target_attr, squeeze=False):
            if not self.dr_cfg.get(key, False):
                return
            r = self.dr_cfg[key.replace("randomize_", "") + "_range"]
            val = sample_1d(r[0], r[1])
            setattr(self, target_attr, val.unsqueeze(-1) if squeeze else val)

        sample_if("randomize_mass", "_dr_mass_scales", squeeze=True)

        # ---- Inertia: per-axis (ixx, iyy) or legacy scalar ----
        if self.dr_cfg.get("randomize_inertia_axis", False):
            r = self.dr_cfg.get("inertia_axis_range", [0.7, 1.3])
            ixx = sample_1d(r[0], r[1])   # (n,)
            iyy = sample_1d(r[0], r[1])   # (n,)
            izz = torch.ones(n, device=gs.device)
            self._dr_inertia_scales = torch.stack([ixx, iyy, izz], dim=-1)  # (n, 3)
        elif self.dr_cfg.get("randomize_inertia", False):
            r = self.dr_cfg["inertia_range"]
            s = sample_1d(r[0], r[1])  # (n,)
            self._dr_inertia_scales = s.unsqueeze(-1).repeat(1, 3)  # (n, 3), contiguous

        sample_if("randomize_kf", "_dr_kf_scales")

        if self.dr_cfg.get("randomize_com", False):
            r = self.dr_cfg["com_shift_range"]
            com_x = sample_1d(r[0], r[1])
            com_y = sample_1d(r[0], r[1])
            com_z = sample_1d(r[0], r[1])
            self._dr_com_shifts = torch.stack([com_x, com_y, com_z], dim=-1).unsqueeze(-2)

    def _randomize_dynamics(self, envs_idx):
        if self.dr_cfg is None or len(envs_idx) == 0:
            return

        if hasattr(self, "_dr_inertia_scales"):
            new_scale = self._dr_inertia_scales[envs_idx]  # (M, 3)
            # First use Ixx for isotropic scaling (synchronize mass / invweight)
            self.drone._solver.set_links_inertia(
                1.0 / self._prev_inertia_scales[envs_idx, :1].clamp(min=1e-6),
                links_idx=[self._base_link_global],
                envs_idx=envs_idx,
            )
            self.drone._solver.set_links_inertia(
                new_scale[:, :1],
                links_idx=[self._base_link_global],
                envs_idx=envs_idx,
            )
            # Correct Iyy / Izz as independent scaling (direct manipulation of inertia tensor diagonal)
            ixx = new_scale[:, 0].clamp(min=1e-6)
            corr_iyy = new_scale[:, 1] / ixx
            corr_izz = new_scale[:, 2] / ixx
            inertial_i_data = qd_to_torch(self.drone._solver.links_info.inertial_i, transpose=True, copy=False)
            inertial_i_data[envs_idx, self._base_link_global, 1, 1] *= corr_iyy
            inertial_i_data[envs_idx, self._base_link_global, 2, 2] *= corr_izz

            self._prev_inertia_scales[envs_idx] = new_scale

        if hasattr(self, "_dr_mass_scales"):
            self.drone.set_links_inertial_mass(
                self._nominal_mass * self._dr_mass_scales[envs_idx],
                links_idx_local=[self._base_link_idx],
                envs_idx=envs_idx,
            )

        if hasattr(self, "_dr_com_shifts"):
            self.drone.set_COM_shift(
                self._dr_com_shifts[envs_idx],
                links_idx_local=[self._base_link_idx],
                envs_idx=envs_idx,
            )

        if hasattr(self, "_dr_kf_scales"):
            self._kf_scales[envs_idx] = self._dr_kf_scales[envs_idx]

    # ============================================================
    # DR Parameter Export & Hard Case Injection
    # ============================================================

    def get_dr_params(self):
        params = {}
        if hasattr(self, "_dr_mass_scales"):
            params["mass"] = self._dr_mass_scales.squeeze(-1)
        if hasattr(self, "_dr_inertia_scales"):
            s = self._dr_inertia_scales  # (n, 3)
            params["inertia"] = s[:, 0]     # Ixx as representative (backward compat)
            params["ixx"] = s[:, 0]
            params["iyy"] = s[:, 1]
            params["izz"] = s[:, 2]
        if hasattr(self, "_dr_com_shifts"):
            shifts = self._dr_com_shifts.squeeze(-2)
            params["com_x"] = shifts[:, 0]
            params["com_y"] = shifts[:, 1]
            params["com_z"] = shifts[:, 2]
        if hasattr(self, "_dr_kf_scales"):
            params["kf"] = self._dr_kf_scales
        return params

    def inject_hard_params(self, hard_params_list):
        n_hard = min(len(hard_params_list), self.num_envs)
        if n_hard == 0:
            return n_hard

        hard_envs = torch.arange(n_hard, device=gs.device)

        for i, hp in enumerate(hard_params_list[:n_hard]):
            if hasattr(self, "_dr_mass_scales"):
                self._dr_mass_scales[i] = hp["mass_scale"]
            if hasattr(self, "_dr_inertia_scales"):
                if "ixx_scale" in hp and "iyy_scale" in hp:
                    izz = hp.get("izz_scale", 1.0)
                    self._dr_inertia_scales[i] = torch.tensor(
                        [hp["ixx_scale"], hp["iyy_scale"], izz],
                        device=gs.device, dtype=gs.tc_float,
                    )
                else:
                    s = hp["inertia_scale"]
                    self._dr_inertia_scales[i] = torch.tensor([s, s, s], device=gs.device, dtype=gs.tc_float)
            if hasattr(self, "_dr_com_shifts"):
                self._dr_com_shifts[i, 0, 0] = hp["com_shift_x"]
                self._dr_com_shifts[i, 0, 1] = hp["com_shift_y"]
                self._dr_com_shifts[i, 0, 2] = hp["com_shift_z"]
            if hasattr(self, "_dr_kf_scales"):
                self._dr_kf_scales[i] = hp["kf_scale"]

        if hasattr(self, "_dr_inertia_scales"):
            self.drone._solver.set_links_inertia(
                1.0 / self._prev_inertia_scales[hard_envs, :1].clamp(min=1e-6),
                links_idx=[self._base_link_global],
                envs_idx=hard_envs,
            )
            self.drone._solver.set_links_inertia(
                self._dr_inertia_scales[hard_envs][:, :1],
                links_idx=[self._base_link_global],
                envs_idx=hard_envs,
            )
            # Correct Iyy / Izz as independent scaling
            new_scale = self._dr_inertia_scales[hard_envs]
            ixx = new_scale[:, 0].clamp(min=1e-6)
            corr_iyy = new_scale[:, 1] / ixx
            corr_izz = new_scale[:, 2] / ixx
            inertial_i_data = qd_to_torch(self.drone._solver.links_info.inertial_i, transpose=True, copy=False)
            inertial_i_data[hard_envs, self._base_link_global, 1, 1] *= corr_iyy
            inertial_i_data[hard_envs, self._base_link_global, 2, 2] *= corr_izz

            self._prev_inertia_scales[hard_envs] = self._dr_inertia_scales[hard_envs]

        if hasattr(self, "_dr_mass_scales"):
            self.drone.set_links_inertial_mass(
                self._nominal_mass * self._dr_mass_scales[hard_envs],
                links_idx_local=[self._base_link_idx],
                envs_idx=hard_envs,
            )

        if hasattr(self, "_dr_com_shifts"):
            self.drone.set_COM_shift(
                self._dr_com_shifts[hard_envs],
                links_idx_local=[self._base_link_idx],
                envs_idx=hard_envs,
            )

        if hasattr(self, "_dr_kf_scales"):
            self._kf_scales[hard_envs] = self._dr_kf_scales[hard_envs]

        return n_hard

    # ============================================================
    # Reset
    # ============================================================

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        if self._randomize_init_state:
            n = len(envs_idx)
            pos_noise = gs_rand_float(-0.05, 0.05, (n, 3), gs.device)
            self.base_pos[envs_idx] = self.base_pos[envs_idx] + pos_noise

            yaw = gs_rand_float(-math.pi / 6, math.pi / 6, (n,), gs.device)
            half_yaw = yaw * 0.5
            yaw_quat = torch.stack([
                torch.cos(half_yaw),
                torch.zeros(n, device=gs.device),
                torch.zeros(n, device=gs.device),
                torch.sin(half_yaw),
            ], dim=-1)
            self.base_quat[envs_idx] = transform_quat_by_quat(
                self.base_quat[envs_idx], yaw_quat
            )

        self.last_base_pos[envs_idx] = self.base_pos[envs_idx]
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ============================================================
    # Rewards
    # ============================================================

    def _reward_target(self):
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw)
        yaw = torch.where(yaw < -180, yaw + 360, yaw)
        yaw = yaw / 180 * math.pi
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / math.pi, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
