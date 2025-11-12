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
from isaaclab.sensors import ContactSensor

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
            "rew_success": [],
            # "rew_left_vel": [],
            "rew_falling_penalty": [],
            # "rew_stopping_bonus": [],
            # "rew_obj_velocity": []
            "rew_time": [],
            "rew_pinky": [],
            "rew_palm_facing": [],
            "rew_object_orientation": []
        }

        self.dones = {
            "dropped": [],
            "timed_out": [],
            "successful": [],
            # "total": [],
            "dropped_x_time": [],
            "timed_out_x_time": [],
            "successful_x_time": [],
            # "total_x_time": [],
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
            "left_wrist_pitch_joint",
            # hip
            # "waist_yaw_joint",
            # "waist_pitch_joint",
            # "waist_roll_joint",
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
        rew_success = np.array(self.rewards["rew_success"])
        rew_time = np.array(self.rewards["rew_time"])
        # rew_left_vel = np.array(self.rewards["rew_left_vel"])
        rew_falling_penalty = np.array(self.rewards["rew_falling_penalty"])
        # rew_stopping_bonus = np.array(self.rewards["rew_stopping_bonus"])
        # rew_obj_vel = np.array(self.rewards["rew_obj_velocity"])
        rew_palm_facing = np.array(self.rewards["rew_palm_facing"])
        rew_pinky = np.array(self.rewards["rew_pinky"])
        rew_object_orientation = np.array(self.rewards["rew_object_orientation"])


        x_coords = np.arange(len(self.rewards["total_reward"]))

        # plt.plot(x_coords,rew_lift)
        # plt.plot(x_coords,rew_dist_right)
        plt.plot(x_coords,rew_falling_penalty, label="Fall Pen")
        plt.plot(x_coords,rew_left_dist, label="Left Distance")
        plt.plot(x_coords,rew_success, label="Left Bonus")
        plt.plot(x_coords,rew_time,label="Time")
        plt.plot(x_coords,rew_palm_facing,label="Palm Facing")
        plt.plot(x_coords,rew_pinky,label="Pinky")
        plt.plot(x_coords,rew_object_orientation,label="Obj Ori")

        plt.plot(x_coords,total_reward, label="Total")

        plt.title("Steps vs Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend(loc="upper left")
        plt.savefig("rewards.png")
        plt.clf()

        rewards = [(total_reward, "Total Reward"), (rew_left_dist, "Left Distance Reward"), (rew_success, "Success Reward"), (rew_time, "Time Reward"), (rew_falling_penalty, "Falling Penalty"), (rew_palm_facing, "Palm Facing"), (rew_pinky, "Pinky"), (rew_object_orientation, "Obj Ori")]
        for reward_tuple in rewards:
            reward_values, label = reward_tuple
            plt.plot(x_coords,reward_values)
            plt.xlabel("Step")
            plt.ylabel(label)
            plt.title("Steps vs " + label)
            plt.savefig(label.lower().replace(" ", "_") + ".png")
            plt.clf()

        plt.clf()

        # Creating charts for dones
        dropped_grouped = []
        timed_out_grouped = []
        successful_grouped = []
        dropped_x_time_grouped = []
        timed_out_x_time_grouped = []
        successful_x_time_grouped = []
        dones_len = len(self.dones["dropped"])
        for i in range(0, dones_len, 30):
            i_end = min(i + 30, dones_len)
            dropped_grouped.append(sum(self.dones["dropped"][i:i_end]))
            timed_out_grouped.append(sum(self.dones["timed_out"][i:i_end]))
            successful_grouped.append(sum(self.dones["successful"][i:i_end]))
            dropped_x_time_grouped.append(sum(self.dones["dropped_x_time"][i:i_end]))
            timed_out_x_time_grouped.append(sum(self.dones["timed_out_x_time"][i:i_end]))
            successful_x_time_grouped.append(sum(self.dones["successful_x_time"][i:i_end]))

        if sum(dropped_grouped[-2:]) == 0:
            dones_data = np.array([sum(timed_out_grouped[-2:]), sum(successful_grouped[-2:])])
            dones_labels = ["Timed Out", "Successful"]
        else :
            dones_data = np.array([sum(dropped_grouped[-2:]), sum(timed_out_grouped[-2:]), sum(successful_grouped[-2:])])
            dones_labels = ["Dropped", "Timed Out", "Successful"]
        plt.pie(dones_data, labels=dones_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Final Dones Breakdown")
        plt.savefig("dones_final.png")
        plt.clf()

        if sum(dropped_x_time_grouped[-2:]) == 0:
            dones_data = np.array([sum(timed_out_x_time_grouped[-2:]), sum(successful_x_time_grouped[-2:])])
            dones_labels = ["Timed Out", "Successful"]
        else :
            dones_data = np.array([sum(dropped_x_time_grouped[-2:]), sum(timed_out_x_time_grouped[-2:]), sum(successful_x_time_grouped[-2:])])
            dones_labels = ["Dropped", "Timed Out", "Successful"]
        plt.pie(dones_data, labels=dones_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Final Dones x Time Breakdown")
        plt.savefig("dones_time_final.png")

        plt.clf()
        grouped_len = len(dropped_grouped)
        x_coords = np.linspace(0, dones_len, grouped_len)


        dropped = np.array(dropped_grouped)
        timed_out = np.array(timed_out_grouped)
        successful = np.array(successful_grouped)
        total = dropped + timed_out + successful
        total[total == 0] = 1

        dropped = dropped / total
        timed_out = timed_out / total
        successful = successful / total
        plt.plot(x_coords,dropped,label="Dropped")
        plt.plot(x_coords,timed_out,label="Timed Out")
        plt.plot(x_coords,successful,label="Successful")
        plt.title("Dones vs Steps")
        plt.xlabel("Step")
        plt.ylabel("Dones")
        plt.legend(loc="upper left")
        plt.savefig("dones.png")

        plt.clf()

        dropped = np.array(dropped_x_time_grouped)
        timed_out = np.array(timed_out_x_time_grouped)
        successful = np.array(successful_x_time_grouped)
        total = dropped + timed_out + successful
        total[total == 0] = 1
        dropped = dropped / total
        timed_out = timed_out / total
        successful = successful / total

        plt.plot(x_coords,dropped,label="Dropped")
        plt.plot(x_coords,timed_out,label="Timed Out")
        plt.plot(x_coords,successful,label="Successful")
        plt.title("Dones x Time vs Steps")
        plt.xlabel("Step")
        plt.ylabel("Dones x Time")
        plt.legend(loc="upper left")
        plt.savefig("dones_x_time.png")


        super().close()          

    def _setup_scene(self):
        #  articulation and block
        self.robot = Articulation(cfg=self.cfg.scene.robot)
        self.block = RigidObject(cfg=self.cfg.scene.block)
        self.force_hand = ContactSensor(self.cfg.scene.contact_forces_left_hand)
        self.force_finger = ContactSensor(self.cfg.scene.contact_forces_left_hand)

        
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
        self.scene.sensors["contact_forces_left_hand"] = self.force_hand
        self.scene.sensors["contact_forces_left_finger"] = self.force_finger


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        normalised_actions = torch.tanh(self.actions)
        # calculate distance
        obj_pos = self.block.data.root_pos_w
        link_data = self.robot.data.body_link_pos_w
        # left_hand_pos = link_data[:, 29]
        # left_dist = torch.linalg.norm(left_hand_pos - obj_pos, dim=-1)
        # normalise distance to be between 0 and 1
        left_finger_dist = torch.linalg.norm(link_data[:, 44] - obj_pos, dim=-1)
        normalised_distance = torch.tanh(left_finger_dist / 0.5) * 2
        # print(f"Normalised distance: max={torch.max(normalised_distance).item()} min={torch.min(normalised_distance).item()}")
        # When it gets closer, max action value decreases
        # "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint" but wrist not slowed
        scaled_actions = normalised_actions * (normalised_distance.unsqueeze(1))

        noise_actions = scaled_actions + torch.randn_like(self.actions) * 0.05
        # For exploration and to simulate motor errors 
        # noise_actions = torch.cat((scaled_actions, normalised_actions[:, 4:]), dim=-1) + torch.randn_like(self.actions) * 0.05 
        self.robot.set_joint_position_target(noise_actions, joint_ids=self._controlled_joint_ids.tolist())

    def _get_observations(self) -> dict:
        # gather joint pos/vel for controlled joints
        joint_pos = self.joint_pos[:, self._controlled_joint_ids].view(self.num_envs, -1)
        joint_pos += torch.randn_like(joint_pos) * 0.05 
        joint_vel = self.joint_vel[:, self._controlled_joint_ids].view(self.num_envs, -1)
        joint_vel += torch.randn_like(joint_vel) * 0.05

        # Add the hand world frame pose to observation
        hand_pose_w = self.robot.data.body_link_pose_w[:, 29]
        hand_pose_w[:, :2] -= self.scene.env_origins[:, :2]

        # object world position and quaternion
        obj_pose = self.block.data.root_pose_w  # (num_envs, 3)
        obj_pose += torch.randn_like(obj_pose) * 0.05

        obs = torch.cat((joint_pos, joint_vel, obj_pose, hand_pose_w), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Distance of the hand to the bowl
        obj_pos = self.block.data.root_pos_w # refers to the root position in the world frame
        obj_z = obj_pos[:, 2] # 1.0166 when it is on the table
        obj_quat = self.block.data.root_quat_w
        x = obj_quat[:, 1]
        y = obj_quat[:, 2]
        z = obj_quat[:, 3]
        w = obj_quat[:, 0]
        t2 = 2.0 * (w * y - z * x)
        # Clamp t2 to handle potential floating-point inaccuracies
        t2 = torch.clamp(t2, min=-1.0, max=1.0)
        pitch = torch.abs(torch.arcsin(t2))
        
        rew_object_orientation = self.cfg.reward_object_orientation * (pitch >= 0.5).to(dtype=torch.float32)

        # Hand joints
        # left: name="left_hand_roll_link", index=29
        # right: name="right_hand_roll_link", index=30
        # left yaw: name="left_hand_yaw_link", index=27
        # Left middle finger: name="L_middle_intermediate_link", index=44
        # L Proximal Middle link: name="L_middle_proximal_link", index=34
        # left_wrist_yaw_joint#26 left_wrist_pitch_joint#30 left_wrist_roll_joint#28
        link_data = self.robot.data.body_link_pos_w
        
        left_wrist_pos = link_data[:, 29]
        left_middle_pos = link_data[:, 34]
        left_ring_pos = link_data[:, 36]
        # euclidean distance to object
        palm_vector_a = left_ring_pos - left_wrist_pos
        palm_vector_b = left_middle_pos - left_wrist_pos
        palm_normal = torch.nn.functional.normalize(torch.cross(palm_vector_a, palm_vector_b), dim=-1)
        palm_pos = (left_middle_pos * 0.7 + left_wrist_pos * 0.3)

        obj_hand = obj_pos - palm_pos
        obj_hand = torch.nn.functional.normalize(obj_hand, dim=1)
        dot = torch.sum(palm_normal * obj_hand, dim=1)        

        rew_palm_facing = self.cfg.reward_palm_facing_object * torch.tanh((dot - 0.5) / 0.3)

        left_dist = torch.linalg.norm(obj_pos - palm_pos, dim=-1) # distance between hand and bowl
        rew_left_dist = self.cfg.reward_scale_distance_left * left_dist * (left_dist > 0.1).to(dtype=torch.float32)

        # Palm facing block
        rew_success = self.cfg.reward_scale_success * (left_dist <= 0.15).to(dtype=torch.float32) * (dot > 0.5).to(dtype=torch.float32)

        rew_pinky = self.cfg.reward_scale_contact_left_pinky * (torch.linalg.norm(self.force_hand.data.net_forces_w, dim=-1).squeeze(-1) + torch.linalg.norm(self.force_finger.data.net_forces_w, dim=-1).squeeze(-1))

        rew_falling_penalty = self.cfg.reward_scale_falling_penalty * (obj_z < 0.8).to(dtype=torch.float32)
        rew_time = self.cfg.reward_scale_time * self.episode_length_buf

        total_reward = rew_left_dist + rew_success + rew_time + rew_falling_penalty + rew_palm_facing + rew_object_orientation

        self.rewards["total_reward"].append(torch.mean(total_reward).item())
        self.rewards["rew_left_dist"].append(torch.mean(rew_left_dist).item())
        self.rewards["rew_success"].append(torch.mean(rew_success).item())
        self.rewards["rew_falling_penalty"].append(torch.mean(rew_falling_penalty).item())
        self.rewards["rew_time"].append(torch.mean(rew_time).item())
        self.rewards["rew_palm_facing"].append(torch.mean(rew_palm_facing).item())
        self.rewards["rew_pinky"].append(torch.mean(rew_pinky).item())
        self.rewards["rew_object_orientation"].append(torch.mean(rew_object_orientation).item())
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # object dropped too low or timeout
        obj_pos = self.block.data.root_pos_w
        dropped = (obj_pos[:, 2] < 0.8).to(dtype=torch.float32)  # block is below 0.5m
        self.dones["dropped"].append(torch.sum(dropped).item())
        self.dones["dropped_x_time"].append(torch.sum(dropped * self.episode_length_buf).item())
        # print("Dropped: " + self.dones["dropped"])

        # print(f"Max episode: {self.max_episode_length}, Episode length: {self.episode_length_buf}")
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(dtype=torch.float32)
        self.dones["timed_out"].append(torch.sum(time_out).item())
        self.dones["timed_out_x_time"].append(torch.sum(time_out * self.episode_length_buf).item())
        # print("timed_out: " + self.dones['timed_out])

        link_data = self.robot.data.body_link_pos_w
        left_wrist_pos = link_data[:, 29]
        left_middle_pos = link_data[:, 34]
        left_ring_pos = link_data[:, 36]
        palm_pos = (left_middle_pos * 0.7 + left_wrist_pos * 0.3)
        # euclidean distance to object
        left_dist = torch.linalg.norm(palm_pos - obj_pos, dim=-1) # distance between hand and bowl

        # Is the palm facing
        palm_vector_a = left_ring_pos - left_wrist_pos
        palm_vector_b = left_middle_pos - left_wrist_pos
        palm_normal = torch.nn.functional.normalize(torch.cross(palm_vector_a, palm_vector_b), dim=-1)
        obj_hand = obj_pos - palm_pos
        obj_hand = torch.nn.functional.normalize(obj_hand, dim=1)
        dot = torch.sum(palm_normal * obj_hand, dim=1)

        success = (left_dist <= 0.15).to(dtype=torch.float32) * (dot > 0.5).to(dtype=torch.float32)
        self.dones["successful"].append(torch.sum(success).item())
        self.dones["successful_x_time"].append(torch.sum(success * self.episode_length_buf).item())

        # self.dones["total"].append(self.dones["dropped"][-1] + self.dones["timed_out"][-1] + self.dones["successful"][-1])
        # self.dones["total_x_time"].append(self.dones["dropped_x_time"][-1] + self.dones["timed_out_x_time"][-1] + self.dones["successful_x_time"][-1])
        fails = torch.logical_or(time_out, dropped).to(dtype=torch.float32)

        return success, fails

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
        obj_init_pos = torch.tensor([-0.45, 0.40, 1.0], device=joint_pos.device).unsqueeze(0).repeat(len(env_ids), 1)

        # Adds the env positions (just for x and y), add offsets based on each env origin
        obj_init_pos[:, :2] += self.scene.env_origins[env_ids, :2]
        obj_init_pos[:, 0] += sample_uniform(-0.3, 0.3, (len(env_ids), 1), joint_pos.device).squeeze(-1)
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
