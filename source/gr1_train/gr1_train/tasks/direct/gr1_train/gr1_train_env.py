# gr1_pickplace_env.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.

# SPDX-License-Identifier: BSD-3-Clause
import math
from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip

from .gr1_train_env_cfg import Gr1TrainEnvCfg

import numpy as np
import matplotlib.pyplot as plt


class Gr1TrainEnv(DirectRLEnv):
    cfg: Gr1TrainEnvCfg

    def __init__(self, cfg: Gr1TrainEnvCfg, render_mode: str | None = None, **kwargs):
        self.rewards = {
            "total_reward": [],
            # "rew_lift": [],
            # "rew_dist_right": [],
            "rew_left_dist": [],
            "rew_left_bonus": [],
            "rew_left_vel": [],
            "rew_falling_penalty": [],
            "rew_stopping_bonus": [],
            # "rew_obj_velocity": []
            "rew_time": []
        }

        self.dones = {
            "dropped": 0,
            "timed_out": 0,
            "successful": 0,
            "total": 0,
            "dropped_running": [],
            "timed_out_running": [],
            "successful_running": [],
            "total_running": [],
        }

        super().__init__(cfg, render_mode, **kwargs)
        target_names = [
            # left-arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint"
        ]

        # If number of joints lower than action space, pad with last joint
        N = self.cfg.action_space
        if len(target_names) < N:
            while len(target_names) < N:
                target_names.append(target_names[-1])
        target_names = target_names[:N]

        # get indices of target joints
        joint_ids = []
        for nm in target_names:
            idxs, _ = self.robot.find_joints(nm)
            if len(idxs) == 0:
                raise RuntimeError(f"Joint '{nm}' not found on robot articulation.")
            joint_ids.append(idxs[0])
        self._controlled_joint_ids = torch.tensor(joint_ids, dtype=torch.long)

        # cache convenient handles
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
    
    def close(self):
        total_reward = np.array(self.rewards["total_reward"])
        # rew_lift = np.array(self.rewards["rew_lift"])
        # rew_dist_right = np.array(self.rewards["rew_dist_right"])
        rew_left_dist = np.array(self.rewards["rew_left_dist"])
        rew_left_bonus = np.array(self.rewards["rew_left_bonus"])
        rew_time = np.array(self.rewards["rew_time"])
        rew_left_vel = np.array(self.rewards["rew_left_vel"])
        rew_falling_penalty = np.array(self.rewards["rew_falling_penalty"])
        rew_stopping_bonus = np.array(self.rewards["rew_stopping_bonus"])
        # rew_obj_vel = np.array(self.rewards["rew_obj_velocity"])

        x_coords = np.arange(len(self.rewards["total_reward"]))

        # plt.plot(x_coords,rew_lift)
        # plt.plot(x_coords,rew_dist_right)
        plt.plot(x_coords,rew_falling_penalty, label="Fall Pen")
        plt.plot(x_coords,rew_left_dist, label="Left Distance")
        plt.plot(x_coords,rew_left_bonus, label="Left Bonus")
        plt.plot(x_coords,rew_left_vel, label="Left Vel")
        plt.plot(x_coords,rew_stopping_bonus, label="Stopping Bonus")
        # plt.plot(x_coords,rew_obj_vel, label="Obj Vel")
        plt.plot(x_coords,rew_time,label="Time")

        plt.plot(x_coords,total_reward, label="Total")

        plt.title("Steps vs Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend(loc="upper left")
        plt.savefig("rewards.png")
        plt.clf()

        rewards = [(total_reward, "Total Reward"), (rew_left_dist, "Left Distance Reward"), (rew_left_bonus, "Left Bonus Reward"), (rew_time, "Time Reward"), (rew_left_vel, "Left Velocity Reward"), (rew_falling_penalty, "Falling Penalty"), (rew_stopping_bonus, "Stopping Bonus Reward")]
        for reward_tuple in rewards:
            reward_values, label = reward_tuple
            plt.plot(x_coords,reward_values)
            plt.xlabel("Step")
            plt.ylabel(label)
            plt.title("Steps vs " + label)
            plt.savefig(label.lower().replace(" ", "_") + ".png")
            plt.clf()

        plt.clf()

        dones_data = np.array([self.dones["dropped"], self.dones["timed_out"], self.dones["successful"]])
        dones_labels = [f"Dropped: {int(self.dones['dropped'])}", f"Timed Out: {int(self.dones['timed_out'])}", f"Successful: {int(self.dones['successful'])}"]
        plt.pie(dones_data, labels=dones_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.savefig("dones.png")
        plt.clf()

        x_coords = np.arange(len(self.dones["dropped_running"]))
        plt.plot(x_coords,self.dones["dropped_running"], label="Dropped")
        plt.plot(x_coords,self.dones["timed_out_running"], label="Timed out")
        plt.plot(x_coords,self.dones["successful_running"], label="Successful")
        plt.plot(x_coords,self.dones["total_running"], label="Total")
        plt.xlabel("Step")
        plt.ylabel("Number of Terminated")
        plt.legend(loc="upper left")
        plt.savefig("dones_running.png")
        plt.clf()

        super().close()          

    def _setup_scene(self):
        #  articulation and block
        self.robot = Articulation(cfg=self.cfg.scene.robot)
        self.block = RigidObject(cfg=self.cfg.scene.block)
        
        # spawn ground
        spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())

        # clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # CPU sim collision filter
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add object and robot to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["block"] = self.block

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        normalised_actions = torch.tanh(self.actions)
        scaled_actions = normalised_actions * self.cfg.max_action_value
        noise_actions = scaled_actions + torch.randn_like(self.actions) * 0.05 # For exploration
        self.robot.set_joint_position_target(noise_actions, joint_ids=self._controlled_joint_ids.tolist())

    def _get_observations(self) -> dict:
        # gather joint pos/vel for controlled joints
        joint_pos = self.joint_pos[:, self._controlled_joint_ids].view(self.num_envs, -1)
        joint_pos += torch.randn_like(joint_pos) * 0.05 
        joint_vel = self.joint_vel[:, self._controlled_joint_ids].view(self.num_envs, -1)
        joint_vel += torch.randn_like(joint_vel) * 0.05

        # object world position and quaternion
        obj_pos = self.block.data.root_pos_w  # (num_envs, 3)
        obj_quat = self.block.data.root_quat_w  # (num_envs, 4)
        obj_pos += torch.randn_like(obj_pos) * 0.05
        obj_quat += torch.randn_like(obj_quat) * 0.05

        obs = torch.cat((joint_pos, joint_vel, obj_pos, obj_quat), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Joint velocity (between a range)
        # Distance of the hands the bowl (left hand more than right)
        # Height of the bowl (big reward)
        # Velocity of the bowl (betwen a range)

        obj_pos = self.block.data.root_pos_w # refers to the root position in the world frame
        obj_z = obj_pos[:, 2]
        # obj_velocity = self.block.data.root_vel_w
        # 1.0166

        # Hand joints
        # left: name="left_hand_roll_link", index=29
        # right: name="right_hand_roll_link", index=30
        # Left middle finger: name="L_middle_intermediate_link", index=44
        left_hand_pos = None

        link_data = self.robot.data.body_link_pos_w
        vel_data = self.robot.data.body_link_lin_vel_w
        left_hand_pos = link_data[:, 44]
        left_vel = vel_data[:, 29]
        # euclidean distance to object
        left_dist = torch.linalg.norm(left_hand_pos - obj_pos, dim=-1) # distance between hand and bowl
        left_vel_norm = torch.linalg.norm(left_vel, dim=-1)
        # print(f"Left dist norm: {left_dist}, Left velocity norm: {left_vel_norm}")
        # print(f"Left velocity {left_vel_norm}")
        rew_stopping_bonus = self.cfg.reward_scale_stopping_bonus * (left_dist < 0.1).to(dtype=torch.float32) * (left_vel_norm < 0.1).to(dtype=torch.float32)

        # right_dist = torch.linalg.norm(right_hand_pos - obj_pos, dim=-1) # distance between hand and bowl
        rew_left_dist = self.cfg.reward_scale_distance_left * left_dist * (left_dist < 0.1).to(dtype=torch.float32)
        rew_left_bonus = self.cfg.reward_scale_left_bonus * (left_dist < 0.1).to(dtype=torch.float32)

        rew_left_vel = self.cfg.reward_scale_left_vel * left_vel_norm

        rew_falling_penalty = self.cfg.reward_scale_falling_penalty * (obj_z < 0.8).to(dtype=torch.float32)
        
        # rew_obj_vel = self.cfg.reward_scale_obj_vel * torch.linalg.norm(obj_velocity)

        # rew_dist_right = self.cfg.reward_scale_distance_right * right_dist

        # rew_lift = self.cfg.reward_scale_lift * (obj_z > 1.1).to(dtype=torch.float32) * (obj_z < 1.5).to(dtype=torch.float32)
        # print(f"Episode length buff {self.episode_length_buf}")
        rew_time = self.cfg.reward_scale_time * self.episode_length_buf
        # rew_object_velocity = self.cfg.reward_scale_velocity * (obj_velocity > 0.1 and obj_velocity < 0.5).to(dtype=torch.float32)

        total_reward = rew_left_dist + rew_left_bonus + rew_time + rew_falling_penalty + rew_stopping_bonus + rew_left_vel

        self.rewards["total_reward"].append(torch.mean(total_reward).item())
        # self.rewards["rew_lift"].append(torch.mean(rew_lift).item())
        # self.rewards["rew_dist_right"].append(torch.mean(rew_dist_right).item())
        self.rewards["rew_left_dist"].append(torch.mean(rew_left_dist).item())
        self.rewards["rew_left_bonus"].append(torch.mean(rew_left_bonus).item())
        self.rewards["rew_left_vel"].append(torch.mean(rew_left_vel).item())
        self.rewards["rew_falling_penalty"].append(torch.mean(rew_falling_penalty).item())
        self.rewards["rew_stopping_bonus"].append(torch.mean(rew_stopping_bonus).item())

        # self.rewards["rew_obj_velocity"].append(torch.mean(rew_obj_vel).item())
        self.rewards["rew_time"].append(torch.mean(rew_time).item())
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # object dropped too low or timeout
        obj_pos = self.block.data.root_pos_w
        dropped = (obj_pos[:, 2] < 0.5).to(dtype=torch.float32)  # block is below 0.5m
        self.dones["dropped"] += torch.sum(dropped).item()
        self.dones["dropped_running"].append(self.dones["dropped"])
        # print("Dropped: " + self.dones["dropped"])

        # print(f"Max episode: {self.max_episode_length}, Episode length: {self.episode_length_buf}")
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(dtype=torch.float32)
        self.dones["timed_out"] += torch.sum(time_out).item()
        self.dones["timed_out_running"].append(self.dones["timed_out"])
        # print("timed_out: " + self.dones['timed_out])

        link_data = self.robot.data.body_link_pos_w
        vel_data = self.robot.data.body_link_lin_vel_w
        left_hand_pos = link_data[:, 44]
        left_vel = vel_data[:, 29]
        # euclidean distance to object
        left_dist = torch.linalg.norm(left_hand_pos - obj_pos, dim=-1) # distance between hand and bowl
        left_vel_norm = torch.linalg.norm(left_vel, dim=-1)
        close_hand = (left_dist < 0.2).to(dtype=torch.float32) * (left_vel_norm < 0.2).to(dtype=torch.float32)
        self.dones["successful"] += torch.sum(close_hand).item()
        self.dones["successful_running"].append(self.dones["successful"])
        # print("successful: " + self.dones["timed_out"])
        self.dones["total"] += torch.sum(dropped).item() + torch.sum(time_out).item() + torch.sum(close_hand).item()
        self.dones["total_running"].append(self.dones["total"])

        done = torch.logical_or(close_hand, dropped).to(dtype=torch.float32)

        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot joints to default
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # small uniform noise to joints
        noise_scale = 0.02
        joint_pos[:, self._controlled_joint_ids] += sample_uniform(
            -noise_scale, noise_scale, joint_pos[:, self._controlled_joint_ids].shape, joint_pos.device
        )

        default_root = self.robot.data.default_root_state[env_ids].clone()

        # Bring object near the table start location with small noise
        # object initial pose (world)
        obj_init_pos = torch.tensor([-0.3, 0.45, 1.0], device=joint_pos.device).unsqueeze(0).repeat(len(env_ids), 1)

        # Adds the env positions (just for x and y), add offsets based on each env origin
        obj_init_pos[:, :2] += self.scene.env_origins[env_ids, :2]
        obj_init_pos[:, 0] += sample_uniform(-0.2, 0.2, (len(env_ids), 1), joint_pos.device).squeeze(-1)
        obj_init_pos[:, 1] += sample_uniform(-0.1, 0.1, (len(env_ids), 1), joint_pos.device).squeeze(-1)

        # write robot joint states
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # reset object root pose & velocity in sim
        # root_quat: keep identity
        obj_root_state = torch.zeros((len(env_ids), 7), device=joint_pos.device)
        obj_root_state[:, :3] = obj_init_pos
        obj_root_state[:, 3] = 1.0  # quat w
        self.block.write_root_pose_to_sim(obj_root_state, env_ids)

        obj_root_vel = torch.zeros((len(env_ids), 6), device=joint_pos.device)
        self.block.write_root_velocity_to_sim(obj_root_vel, env_ids)

        # write robot pose/joint states into sim
        default_root_state = default_root.clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
