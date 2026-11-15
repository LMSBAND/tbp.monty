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
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.sensor_modules import TouchSM
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.geometry import SurfacePoseVectors

FINGER = "finger"
AGENT_POSITION = np.array([0.5, 1.5, 0.25])


def touching(**extra: Any) -> dict[str, Any]:
    """A fingertip observation in contact with a surface that faces +x.

    Returns:
        The observation, with any extra fields.
    """
    return {
        "touching": True,
        "press": 30.0,
        "touch_depth": 0.015,
        "grip": 0.7,
        "touch_pos": np.array([0.1, 0.0, 0.0]),
        "touch_normal": np.array([2.0, 0.0, 0.0]),
        "touch_gap": 0.025,
        **extra,
    }


def curved(k1: float, k2: float) -> dict[str, Any]:
    return touching(
        touch_curvatures=np.array([k1, k2]),
        touch_curvature_dirs=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        touch_curvature_valid=True,
    )


class TouchSMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = RuntimeContext(rng=np.random.RandomState(42))

    def touch_sm(self, **kwargs: Any) -> TouchSM:
        sm = TouchSM(sensor_module_id=FINGER, **kwargs)
        sm.update_state(
            AgentState(
                sensors={
                    SensorID(FINGER): SensorState(position=np.zeros(3), rotation=qt.one)
                },
                position=AGENT_POSITION,
                rotation=qt.one,
            )
        )
        return sm

    def test_percept_is_the_contact(self) -> None:
        percept = self.touch_sm().step(self.ctx, touching())

        np.testing.assert_allclose(percept.location, [0.1, 0.0, 0.0])
        pose_vectors = percept.morphological_features["pose_vectors"]
        assert isinstance(pose_vectors, SurfacePoseVectors)
        np.testing.assert_allclose(pose_vectors.surface_normal, [1.0, 0.0, 0.0])
        # The two tangents only complete the frame.
        np.testing.assert_allclose(
            pose_vectors.value @ pose_vectors.value.T, np.eye(3), atol=1e-12
        )
        assert not percept.morphological_features["pose_fully_defined"]
        assert percept.get_on_object()
        assert percept.process_features_in_lm
        assert percept.sender_id == FINGER

    def test_grip_and_stiffness(self) -> None:
        percept = self.touch_sm().step(self.ctx, touching())

        features = percept.non_morphological_features
        np.testing.assert_allclose(features["grip"], 0.7)
        np.testing.assert_allclose(features["stiffness"], 2.0)  # 30 N over 15 mm

    def test_percept_carries_what_a_surface_policy_steers_by(self) -> None:
        percept = self.touch_sm().step(self.ctx, touching())

        features = percept.non_morphological_features
        assert features["object_coverage"] == 1.0
        np.testing.assert_allclose(features["min_depth"], 0.025)
        np.testing.assert_allclose(features["mean_depth"], 0.025)

    def test_not_touching_is_off_object_at_the_sensor(self) -> None:
        percept = self.touch_sm().step(self.ctx, {"touching": False})

        np.testing.assert_allclose(percept.location, AGENT_POSITION)
        assert not percept.get_on_object()
        assert not percept.process_features_in_lm
        assert percept.non_morphological_features["object_coverage"] == 0.0

    def test_curvature_directions_complete_the_pose(self) -> None:
        sm = self.touch_sm(
            features=["grip", "principal_curvatures", "principal_curvatures_log"]
        )
        percept = sm.step(self.ctx, curved(-0.4, -18.3))

        pose_vectors = percept.morphological_features["pose_vectors"]
        np.testing.assert_allclose(pose_vectors.principal_curvature_1, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(pose_vectors.principal_curvature_2, [0.0, 0.0, 1.0])
        assert percept.morphological_features["pose_fully_defined"]
        features = percept.non_morphological_features
        np.testing.assert_allclose(features["principal_curvatures"], [-0.4, -18.3])
        assert "principal_curvatures_log" in features

    def test_equal_curvatures_leave_the_pose_undefined(self) -> None:
        sm = self.touch_sm(features=["principal_curvatures"])
        percept = sm.step(self.ctx, curved(-10.0, -10.5))

        assert not percept.morphological_features["pose_fully_defined"]

    def test_touch_without_its_curvature_does_not_reach_the_lm(self) -> None:
        sm = self.touch_sm(features=["grip", "principal_curvatures"])
        percept = sm.step(self.ctx, touching(touch_curvature_valid=False))

        # Still on the object, so a policy can steer by it.
        assert percept.get_on_object()
        assert not percept.process_features_in_lm

    def test_texture_features(self) -> None:
        t = np.arange(256) / 1000.0
        vibration = 2.0 * np.sin(2 * np.pi * 125.0 * t)
        sm = self.touch_sm(features=["roughness", "vibration_centroid"])
        percept = sm.step(self.ctx, touching(vibration=vibration))

        features = percept.non_morphological_features
        np.testing.assert_allclose(
            features["roughness"], np.log10(np.sqrt(2.0)), atol=0.01
        )
        np.testing.assert_allclose(features["vibration_centroid"], 125.0, atol=2.0)

    def test_motor_only_step_is_not_passed_on(self) -> None:
        percept = self.touch_sm().step(self.ctx, touching(), motor_only_step=True)

        assert not percept.pass_message

    def test_feature_change_filter_reads_the_normal(self) -> None:
        sm = self.touch_sm(delta_thresholds={"distance": 0.01, "grip": 0.05})
        first = sm.step(self.ctx, touching())
        same = sm.step(self.ctx, touching())
        slicker = sm.step(self.ctx, touching(grip=0.3))

        assert first.process_features_in_lm
        assert not same.process_features_in_lm
        assert slicker.process_features_in_lm

    def test_state_dict_has_every_percept(self) -> None:
        sm = self.touch_sm()
        sm.step(self.ctx, touching())
        sm.step(self.ctx, {"touching": False})

        assert len(sm.state_dict()["processed_observations"]) == 2
        sm.reset()
        assert sm.state_dict()["processed_observations"] == []
