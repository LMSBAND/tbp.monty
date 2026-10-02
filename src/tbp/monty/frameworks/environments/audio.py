# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""An environment whose objects are sounds.

Each object is a recording (`<sounds_dir>/<name>.wav`). Every step delivers
the next `step_seconds` of it as a raw waveform observation; the cochlea and
the SAI live in the sensor module (AudioSM), not here -- the ear is a sensor,
not a fact about the world. The recording loops, rests included, and restarts
from its top at every new episode, so each episode hears the same take from
the same needle-drop.

The listener does not move: displacement in an auditory object's frame comes
from the sound evolving over time, so actions are accepted and ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import quaternion as qt
from scipy.io import wavfile

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.path import monty_data_path

logger = logging.getLogger(__name__)

__all__ = ["AudioEnvironment", "load_wav_mono"]


def load_wav_mono(path: str | Path, sample_rate: float = 44100.0) -> np.ndarray:
    """Load a WAV as mono float64, peak-normalized to 0.5.

    Level is not identity: every take is brought to the same peak.

    Returns:
        The samples.

    Raises:
        ValueError: If the file's sample rate differs from `sample_rate`.
    """
    sr, x = wavfile.read(str(path))
    x = (
        x.astype(np.float64) / np.iinfo(x.dtype).max
        if x.dtype.kind == "i"
        else (x.astype(np.float64))
    )
    if x.ndim == 2:
        x = x.mean(axis=1)
    if int(sr) != int(sample_rate):
        raise ValueError(f"{path}: {sr} Hz; resample to {int(sample_rate)} Hz")
    peak = np.abs(x).max()
    return x / peak * 0.5 if peak > 0 else x


class AudioEnvironment(SimulatedObjectEnvironment):
    """Objects are recordings; observations are consecutive chunks of them."""

    def __init__(
        self,
        sounds_dir: str | Path,
        sample_rate: float = 44100.0,
        step_seconds: float = 0.25,
        agent_id: str = "agent_id_0",
        sensor_id: str = "audio",
        files: dict[str, str] | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            sounds_dir: Directory of `<object name>.wav` files. A relative path
                is resolved against `$MONTY_DATA/audio_riffs` (the downloaded dataset).
            sample_rate: Must match the sensor module (44100 for AudioSM).
            step_seconds: Seconds of audio delivered per step.
            agent_id: The listening agent.
            sensor_id: Sensor (and sensor module) id the waveform is sent to.
            files: Optional object name -> file stem overrides. An object
                plays `<files[name]>.wav` instead of `<name>.wav`, so a test
                can present one recording under another object's label (e.g.
                the same riff in a different key) and grade against it.
        """
        d = Path(sounds_dir).expanduser()
        self._dir = d if d.is_absolute() else monty_data_path(None, "audio_riffs") / d
        self._sr = float(sample_rate)
        self._n = round(step_seconds * self._sr)
        self._agent = AgentID(agent_id)
        self._sensor = SensorID(sensor_id)
        self._samples = np.zeros(self._n)
        self._k = 0
        self._next_id = 0
        self._files = dict(files or {})

    def _observe(self) -> tuple[Observations, ProprioceptiveState]:
        n = len(self._samples)
        a = (self._k * self._n) % max(n, 1)
        chunk = np.take(self._samples, range(a, a + self._n), mode="wrap")
        obs = Observations(
            {
                self._agent: AgentObservations(
                    {
                        self._sensor: SensorObservation({"waveform": chunk}),
                    }
                )
            }
        )
        here = (0.0, 0.0, 0.0)
        state = ProprioceptiveState(
            {
                self._agent: AgentState(
                    sensors={self._sensor: SensorState(position=here, rotation=qt.one)},
                    position=here,
                    rotation=qt.one,
                )
            }
        )
        return obs, state

    def step(
        self,
        actions: Sequence[Action],  # noqa: ARG002
    ) -> tuple[Observations, ProprioceptiveState]:
        self._k += 1
        return self._observe()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._k = 0
        return self._observe()

    def close(self) -> None:
        pass

    def add_object(
        self,
        name: str,
        position=(0.0, 0.0, 0.0),  # noqa: ARG002
        rotation=(1.0, 0.0, 0.0, 0.0),  # noqa: ARG002
        scale=(1.0, 1.0, 1.0),  # noqa: ARG002
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,  # noqa: ARG002
    ) -> ObjectInfo:
        stem = self._files.get(name, name)
        self._samples = load_wav_mono(self._dir / f"{stem}.wav", self._sr)
        self._k = 0
        object_id = ObjectID(self._next_id)
        self._next_id += 1
        return ObjectInfo(object_id=object_id, semantic_id=semantic_id)

    def remove_all_objects(self) -> None:
        self._samples = np.zeros(self._n)
        self._k = 0
