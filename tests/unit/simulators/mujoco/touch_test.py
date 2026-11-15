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
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.mujoco.agents import SurfaceAgent, TouchConfig
from tbp.monty.simulators.mujoco.simulator import MuJoCoSimulator

AGENT_ID = AgentID("agent_id_0")
FINGER = SensorID("finger")

CUSTOM_OBJECT_DATA_PATH = (
    Path(__file__).parents[2] / "resources" / "test_custom_objects"
)

HALF_SIZE = 0.1
"""Half the box's edge, and the sphere's radius."""
WORKING_DISTANCE = 0.025
"""How far off the surface the agent is parked. Less than the fingertip's radius."""
FINGERTIP_RADIUS = 0.04

FACING_MINUS_X = tuple(
    np.roll(Rotation.from_euler("y", 90, degrees=True).as_quat(), 1).tolist()
)
"""The agent looks down its -z axis. This turns that to the world's -x. (w, x, y, z)"""


def touch_simulator(
    materials: dict[str, dict[str, Any]] | None = None,
    palpate: float = 0.0,
    **kwargs: Any,
) -> MuJoCoSimulator:
    """A simulator with one surface agent that carries a fingertip and no camera.

    Returns:
        The simulator.
    """
    agent = partial(
        SurfaceAgent,
        agent_id=AGENT_ID,
        sensor_configs={},
        touch=TouchConfig(sensor_id=FINGER, radius=FINGERTIP_RADIUS, palpate=palpate),
    )
    return MuJoCoSimulator(agents=[agent], materials=materials, **kwargs)


def touch_from_plus_x(sim: MuJoCoSimulator) -> dict[str, Any]:
    """Parks the agent off the +x side of an object at the origin, facing it.

    Returns:
        The fingertip's observation.
    """
    embodiment = sim._agents[AGENT_ID]._embodiment
    embodiment.position = (HALF_SIZE + WORKING_DISTANCE, 0.0, 0.0)
    embodiment.rotation = FACING_MINUS_X
    observations, _ = sim.step([])
    return observations[AGENT_ID][FINGER]


class TouchTest(unittest.TestCase):
    def test_nothing_to_touch(self) -> None:
        with touch_simulator() as sim:
            finger = touch_from_plus_x(sim)

            assert not finger["touching"]
            assert finger["press"] == 0.0

    def test_contact_is_where_the_surface_is(self) -> None:
        with touch_simulator() as sim:
            sim.add_object("box", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
            finger = touch_from_plus_x(sim)

            assert finger["touching"]
            assert finger["press"] > 0.0
            np.testing.assert_allclose(
                finger["touch_normal"], [1.0, 0.0, 0.0], atol=1e-6
            )
            np.testing.assert_allclose(finger["touch_gap"], WORKING_DISTANCE, atol=1e-6)
            np.testing.assert_allclose(
                finger["touch_depth"], FINGERTIP_RADIUS - WORKING_DISTANCE, atol=1e-6
            )
            # Somewhere between the box's face and the fingertip's pole inside it.
            assert (
                HALF_SIZE - finger["touch_depth"] <= finger["touch_pos"][0] <= HALF_SIZE
            )
            np.testing.assert_allclose(finger["touch_pos"][1:], [0.0, 0.0], atol=1e-6)

    def test_grip_reads_the_friction_coefficient(self) -> None:
        for friction in (0.3, 0.6, 0.9):
            materials = {"box": {"friction": [friction, 0.005, 0.0001]}}
            with self.subTest(friction=friction), touch_simulator(materials) as sim:
                sim.add_object("box", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
                finger = touch_from_plus_x(sim)

                np.testing.assert_allclose(finger["grip"], friction, rtol=0.02)

    def test_a_stiffer_material_presses_back_harder(self) -> None:
        press = {}
        for name, time_constant in (("stiff", 0.005), ("soft", 0.05)):
            materials = {"box": {"solref": [time_constant, 1.0]}}
            with touch_simulator(materials) as sim:
                sim.add_object("box", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
                press[name] = touch_from_plus_x(sim)["press"]

        assert press["stiff"] > press["soft"] > 0.0

    def test_palpation_reads_curvature(self) -> None:
        with touch_simulator(palpate=0.01) as sim:
            sim.add_object("sphere", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
            ball = touch_from_plus_x(sim)

            assert ball["touch_curvature_valid"]
            # Convex is negative, as in CameraSM.
            np.testing.assert_allclose(
                ball["touch_curvatures"], [-1 / HALF_SIZE, -1 / HALF_SIZE], rtol=0.1
            )

        with touch_simulator(palpate=0.01) as sim:
            sim.add_object("box", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
            face = touch_from_plus_x(sim)

            assert face["touch_curvature_valid"]
            np.testing.assert_allclose(face["touch_curvatures"], [0.0, 0.0], atol=0.5)

    def test_no_palpation_no_curvature(self) -> None:
        with touch_simulator() as sim:
            sim.add_object("box", scale=(HALF_SIZE, HALF_SIZE, HALF_SIZE))
            finger = touch_from_plus_x(sim)

            assert "touch_curvatures" not in finger

    def test_material_is_applied_to_the_geom(self) -> None:
        materials = {
            "box": {"friction": [0.7, 0.005, 0.0001], "solref": [0.02, 1.0]},
        }
        with MuJoCoSimulator(materials=materials) as sim:
            sim.add_object("box")
            sim.add_object("sphere")

            np.testing.assert_allclose(sim.model.geom("box_0").friction[0], 0.7)
            np.testing.assert_allclose(sim.model.geom("box_0").solref, [0.02, 1.0])
            # No entry: MuJoCo's default.
            np.testing.assert_allclose(sim.model.geom("sphere_1").friction[0], 1.0)

    def test_convex_pieces_are_added_and_removed_with_the_object(self) -> None:
        pytest.importorskip("coacd")
        with (
            TemporaryDirectory() as cache,
            patch.dict("os.environ", {"XDG_CACHE_HOME": cache}),
            MuJoCoSimulator(
                data_path=CUSTOM_OBJECT_DATA_PATH, convex_decomposition=True
            ) as sim,
        ):
            sim.add_object("valid_object")

            pieces = [
                sim.model.geom(i).name
                for i in range(sim.model.ngeom)
                if sim.model.geom(i).name.startswith("valid_object_0_c")
            ]
            assert len(pieces) >= 1
            # The rendered mesh no longer collides; its pieces do.
            assert sim.model.geom("valid_object_0").contype[0] == 0
            assert all(sim.model.geom(name).contype[0] == 1 for name in pieces)

            sim.remove_all_objects()

            assert sim.model.ngeom == 0
            assert sim.model.nmesh == 0
