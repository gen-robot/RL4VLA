from copy import deepcopy
from typing import Dict

import numpy as np
import sapien
import torch
# import robocasa  #can be replaced by other
import os

from mani_skill.utils.scene_builder.robocasa.scene_builder import FIXTURES,FIXTURES_INTERIOR

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR

from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import RoboCasaKitchenEnv

ROBOCASA_OBJAVERSE_DIR = ROBOCASA_ASSET_DIR / "objects/objaverse"

@register_env(
    "RoboCasaCustomKitchen-v1", max_episode_steps=100, asset_download_ids=["RoboCasa"]
)
class RoboCasaCustomKitchenEnv(RoboCasaKitchenEnv):
    SUPPORTED_ROBOTS = ["fetch", "none", "mobile_aloha"]
    SUPPORTED_REWARD_MODES = ["none"]
    EXCLUDE_LAYOUTS = []

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        # hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        scene_split=None,  # unsued, for backwards compatibility
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
        fixtures_only=False,
        **kwargs,
    ):        
        all_kwargs = locals()
        all_kwargs.pop("self")
        all_kwargs.pop("args")
        all_kwargs.pop("kwargs")
        all_kwargs.pop("__class__")
        all_kwargs.update(kwargs)
        super().__init__(*args, **all_kwargs)

    def get_task_description(self):
        return "put the apple in the bowl"

    def _get_obj_cfgs(self):
        cfgs = []
        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "apple/apple_0/model.xml"
        )
        # The name should in the scene. Else will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_apple_0",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_main_main_group',
                        sample_region_kwargs=dict(
                            ref='counter_main_main_group',
                        ),
                        size=(0.35, 0.2),
                        pos=("ref", -1.0),
                        rotation=[0,0] 
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "bowl/bowl_1/model.xml"
        )
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_bowl",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.5, 0.0),
                        rotation=[0,0] 
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "apple/apple_10/model.xml"
        )
        # The name should in the scene. Else will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_apple_10",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.8, 0.05),
                        rotation=[0,0] 
                        ),
                ) 
            )
        



        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "banana/banana_1/model.xml"
        )
        # The name should in the scene. Else will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_banana_1",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(1.0, -0.03),
                        rotation=[0,0] 
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "carrot/carrot_1/model.xml"
        )
        # The name should in the scene. Else will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_carrot_1",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='paper_towel_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.6, -0.034),
                        rotation=[-0.1,0.1] 
                        ),
                ) 
            )
        

        obj_model_path = os.path.join(
            ROBOCASA_OBJAVERSE_DIR, "milk/milk_1/model.xml"
        )
        # The name should in the scene. Else will return error.
        cfgs.append(dict(
                info = {"mjcf_path": obj_model_path,},
                type = None,
                name = "obj_milk_1",
                obj_groups = None,
                placement = dict(
                        # ref get_fixtures(): fixture can be a class or an id string.
                        fixture='counter_1_right_group',
                        sample_region_kwargs=dict(
                            ref='counter_1_right_group',
                            loc="left_right",
                        ),
                        size=(0.35, 0.2),
                        pos=(0.0, 0.0),
                        offset=(0.9, 0.04),
                        rotation=[-0.1,0.1] 
                        ),
                ) 
            )

        return cfgs