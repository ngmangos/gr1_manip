# gr1_pickplace_env_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.

# SPDX-License-Identifier: BSD-3-Clause
import tempfile
import torch

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG
from isaaclab.sensors import ContactSensorCfg

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    packing_table = AssetBaseCfg( # Height: 0.9941
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 6cm by 6cm cube
    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9996], rot=[1, 0, 0, 0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 1.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
    )

    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.3, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": -1.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # "waist_yaw_joint": 0.0,
                # "waist_pitch_joint": 0.0,
                # "waist_roll_joint": 0.0,
                # -- keep all others at zero
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    

    contact_forces_left_hand = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_hand_pitch_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )

    contact_forces_left_finger = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/L_pinky_intermediate_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True
    )


    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class Gr1TrainEnvCfg(DirectRLEnvCfg):
    decimation: int = 6
    episode_length_s: float = 5.0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2)
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)

    action_space: int = 7
    # observation: joint positions (7), joint velocities (7) (left and right)
    # object position (3) orientation (4)
    # Left hand roll link pose (7)
    # waist joints: pos (3) vel (3)
    observation_space: int = 14 + 7 + 7
    state_space: int = 0

    max_action = 0.5
    # reward scaling parameters
    # reward_scale_lift: float = 1.0
    # reward_scale_distance_right: float = -10.0
    reward_scale_distance_left: float = -4.0
    reward_scale_success: float = 150.0
    reward_palm_facing_object: float = 10.0

    # reward_scale_stopping_bonus: float = 150.0
    # reward_scale_left_vel: float = -0.01
    # reward_scale_obj_vel: float = -0.7
    reward_scale_falling_penalty: float = -1000.0
    reward_scale_contact_left_pinky: float = -1.0
    reward_object_orientation: float = -10.0

    reward_scale_time: float = -0.3
    # reward_scale_velocity: float = 0.5

    def __post_init__(self):
        idle_action = torch.zeros(self.action_space)
