# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import quaternion as qt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    LookUp,
    MoveForward,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
)
from tbp.monty.frameworks.models.motor_policies import TouchSurfacePolicy
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("agent_id_0")
FINGER = SensorID("finger")
DESIRED_OBJECT_DISTANCE = 0.025
REACH_STEP = 0.02
MAX_REACH = 0.1
SEARCH_DEGREES = 30.0


def observations(**finger: Any) -> Observations:
    return Observations({AGENT_ID: AgentObservations({FINGER: finger})})


NOT_TOUCHING = observations(touching=False, touch_gap=float("inf"))


class TouchSurfacePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(42))
        self.policy = TouchSurfacePolicy(
            touch_sensor_id=FINGER,
            reach_step=REACH_STEP,
            max_reach=MAX_REACH,
            search_degrees=SEARCH_DEGREES,
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=AGENT_ID,
            desired_object_distance=DESIRED_OBJECT_DISTANCE,
        )
        self.state = MotorSystemState(
            {
                AGENT_ID: AgentState(
                    sensors={
                        FINGER: SensorState(position=np.zeros(3), rotation=qt.one)
                    },
                    position=np.array([0.0, 1.5, 0.1]),
                    rotation=qt.one,
                )
            }
        )

    def touch_object(self, obs: Observations) -> Any:
        return self.policy._touch_object(self.ctx, obs, FINGER, self.state)

    def test_reaches_forward_until_it_touches(self) -> None:
        action = self.touch_object(NOT_TOUCHING)

        assert isinstance(action, MoveForward)
        np.testing.assert_allclose(action.distance, REACH_STEP)
        assert self.policy.attempting_to_find_object

    def test_touching_holds_the_working_distance(self) -> None:
        self.touch_object(NOT_TOUCHING)
        action = self.touch_object(observations(touching=True, touch_gap=0.031))

        assert isinstance(action, MoveForward)
        np.testing.assert_allclose(action.distance, 0.031 - DESIRED_OBJECT_DISTANCE)
        assert not self.policy.attempting_to_find_object

    def test_reaching_too_far_comes_back_and_swings(self) -> None:
        reaches = round(MAX_REACH / REACH_STEP)
        for _ in range(reaches):
            assert isinstance(self.touch_object(NOT_TOUCHING), MoveForward)

        action = self.touch_object(NOT_TOUCHING)

        assert isinstance(action, OrientHorizontal)
        np.testing.assert_allclose(action.rotation_degrees, SEARCH_DEGREES)
        # It comes back by everything it reached, less the arc's own advance.
        assert action.forward_distance < 0.0
        # ... and starts reaching again.
        assert isinstance(self.touch_object(NOT_TOUCHING), MoveForward)

    def test_a_full_turn_of_swings_then_searches_vertically(self) -> None:
        reaches = round(MAX_REACH / REACH_STEP)
        swings = []
        for _ in range(round(360 / SEARCH_DEGREES)):
            for _ in range(reaches):
                self.touch_object(NOT_TOUCHING)
            swings.append(self.touch_object(NOT_TOUCHING))

        assert all(isinstance(swing, OrientHorizontal) for swing in swings[:-1])
        assert isinstance(swings[-1], OrientVertical)

    def test_slipping_off_goes_back_to_the_last_touch(self) -> None:
        position = np.array([0.1, 1.5, 0.0])
        rotation = qt.from_rotation_vector([0.0, np.pi / 2, 0.0])
        self.policy._last_touch_pose = (position, rotation)

        action = self.touch_object(NOT_TOUCHING)

        assert isinstance(action, SetAgentPose)
        np.testing.assert_allclose(action.location, position)
        assert isinstance(action.rotation_quat, tuple)
        np.testing.assert_allclose(action.rotation_quat, qt.as_float_array(rotation))
        assert self.policy.slips == 1
        # Only once: after that it reaches from there.
        assert isinstance(self.touch_object(NOT_TOUCHING), MoveForward)

    def test_jump_is_undone_when_it_touches_nothing_or_lands_inside(self) -> None:
        assert self.policy._should_undo_jump(NOT_TOUCHING)
        assert self.policy._should_undo_jump(
            observations(touching=True, touch_gap=-0.002)
        )
        assert not self.policy._should_undo_jump(
            observations(touching=True, touch_gap=0.025)
        )

    def test_reset_forgets_the_last_touch(self) -> None:
        self.policy._last_touch_pose = (np.zeros(3), qt.one)
        self.touch_object(NOT_TOUCHING)

        self.policy.reset()

        assert self.policy.slips == 0
        assert isinstance(self.touch_object(NOT_TOUCHING), MoveForward)
