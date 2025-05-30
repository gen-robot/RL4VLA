from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import BaseController, ControllerConfig


class PDJointPosController(BaseController):
    config: "PDJointPosControllerConfig"
    _start_qpos = None
    _target_qpos = None

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices.long()].cpu().numpy()
        )
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            drive_mode = self.config.drive_mode
            if not isinstance(drive_mode, str):
                drive_mode = drive_mode[i]
            joint.set_drive_properties(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=drive_mode
            )
            joint.set_friction(friction[i])

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        if self._start_qpos is None:
            self._start_qpos = self.qpos.clone()
        else:

            self._start_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()
        if self._target_qpos is None:
            self._target_qpos = self.qpos.clone()
        else:
            self._target_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()

    def set_drive_targets(self, targets):
        self.articulation.set_joint_drive_targets(
            targets, self.joints, self.active_joint_indices
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        self._step += 1

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_qpos": self._target_qpos}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            self._target_qpos = state["target_qpos"]


@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDJointPosController


class PDJointPosMimicController(PDJointPosController):
    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]


class PDJointPosMimicControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosMimicController

class PDJointPosMimicAsymmetricController(PDJointPosController):
    def _get_joint_limits(self):
        """
        Get joint limits while ensuring the main joint (positive range) is used for control,
        and the mimic joint (negative range) is derived accordingly.
        """
        joint_limits = super()._get_joint_limits()
        main_limit = joint_limits[0:-1]
        mimic_limit = joint_limits[1:]
        diff = main_limit - -mimic_limit[..., ::-1]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit, but got {}".format(joint_limits)

        if main_limit[0, -1] > 0: 
            return main_limit  # Return the limits for the main joint
        else:
            return mimic_limit

    def set_drive_targets(self, targets):
        """
        Set drive targets for both the main and mimic joints.
        """
        # Compute the mimic joint target based on the main joint target
        # mimic_targets = -targets

        # Set the drive targets for both joints
        self.articulation.set_joint_drive_targets(
            targets, self.joints, self.active_joint_indices
        )

    def set_action(self, action: Array):
        """
        Override the action to handle asymmetric mimic joints.
        """
        # if np.mean(np.abs(action)) > 0:
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        mimic_action = -action
        action = torch.cat([action, mimic_action], dim=-1)

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Only set the main joint's target position
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape  # Only considering the main joint
            ).clone()

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

class PDJointPosMimicAsymmetricControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosMimicAsymmetricController
