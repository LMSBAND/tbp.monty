# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt


class SensorObservation(TypedDict, total=False):
    """Observations from a sensor."""

    rgba: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.float64]  # TODO: Verify specific type
    semantic: npt.NDArray[np.int_]  # TODO: Verify specific type
    semantic_3d: npt.NDArray[np.float64]
    sensor_frame_data: npt.NDArray[np.float64]
    cam_to_world: npt.NDArray[np.float64]  # TODO: Verify specific type
    pixel_loc: npt.NDArray[np.float64]  # TODO: Verify specific type
    raw: npt.NDArray[np.uint8]
    # Touch, from the physics engine's contact solver (see the MuJoCo `TouchConfig`):
    touching: bool  # the fingertip is in contact with something
    press: float  # normal force at the contact, N
    touch_depth: float  # how far the fingertip is pressed in, m
    grip: float  # tangential / normal force while dragged: the friction coefficient
    touch_pos: npt.NDArray[np.float64]  # the contact point, world coordinates
    touch_normal: npt.NDArray[np.float64]  # surface normal there, out of the object
    touch_gap: float  # from the fingertip's centre out to the surface, m
    touch_curvatures: npt.NDArray[np.float64]  # principal curvatures by palpation, 1/m
    touch_curvature_dirs: npt.NDArray[np.float64]  # their directions, world coordinates
    touch_curvature_valid: bool  # every palpating press landed on the surface
    vibration: npt.NDArray[np.float64]  # acceleration of a stroke over the texture
