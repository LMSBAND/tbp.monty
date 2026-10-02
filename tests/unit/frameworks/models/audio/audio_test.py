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
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.audio import AudioEnvironment
from tbp.monty.frameworks.models.audio import carfac as C
from tbp.monty.frameworks.models.audio import sai as R
from tbp.monty.frameworks.models.audio.ear import Ear, _run_segment
from tbp.monty.frameworks.models.audio.sensor_module import LAG_SCALE, AudioSM
from tbp.monty.frameworks.sensors import SensorID

try:
    from tbp.monty.frameworks.models.audio import fastear
except ImportError:  # numba is the optional `audio` extra
    fastear = None  # type: ignore[assignment]

SR = 44100


def tone(f0: float, seconds: float, harmonics: int = 5) -> np.ndarray:
    """A harmonic tone (a plucked-string stand-in) peaking at 0.3.

    Returns:
        The samples.
    """
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, harmonics + 1))
    return 0.3 * x / np.abs(x).max()


def fresh_cfp():
    cfp = C.design_carfac(fs=SR)
    C.carfac_init(cfp)
    return cfp


def hear(sm: AudioSM, wave: np.ndarray, step_seconds: float = 0.25):
    n = int(step_seconds * SR)
    return [
        sm.step(None, {"waveform": wave[i : i + n]})
        for i in range(0, len(wave) - n + 1, n)
    ]


class CochleaTest(unittest.TestCase):
    @unittest.skipIf(fastear is None, "numba not installed")
    def test_compiled_cochlea_matches_reference(self):
        rng = np.random.default_rng(0)
        wave = tone(196.0, 0.3) + 0.01 * rng.standard_normal(int(0.3 * SR))
        ref = C.run_segment(fresh_cfp(), wave[:, None])[0]
        fast = fastear.run_segment(fresh_cfp(), wave[:, None])[0]
        rel = np.max(np.abs(ref - fast)) / np.max(np.abs(ref))
        self.assertLess(rel, 1e-6)

    def test_streaming_ear_matches_block_sai(self):
        # same cochlea as the Ear (compiled if numba is present), so only the
        # SAI schedule differs: block-at-once vs hop-by-hop streaming
        wave = tone(220.0, 0.3)
        nap = _run_segment(fresh_cfp(), wave[:, None])[0][:, :, 0].T
        block = [f.copy() for f in R.sai_frames(nap)]
        ear = Ear(SR)
        n = int(0.05 * SR)
        streamed = []
        for i in range(0, len(wave), n):
            streamed += ear.hear(wave[i : i + n])
        k = min(len(block), len(streamed))
        self.assertGreater(k, 10)
        np.testing.assert_allclose(streamed[:k], block[:k], rtol=1e-9, atol=1e-12)


class AudioSMTest(unittest.TestCase):
    def test_hears_pitch(self):
        for f0 in (110.0, 220.0, 330.0):
            sm = AudioSM()
            percepts = [p for p in hear(sm, tone(f0, 1.5)) if p.pass_message]
            self.assertTrue(percepts, f"no percept for {f0} Hz")
            pitch = percepts[-1].non_morphological_features["pitch_hz"]
            self.assertAlmostEqual(pitch / f0, 1.0, delta=0.02)

    def test_octave_is_one_lag_scale_along_x(self):
        xs = []
        for f0 in (220.0, 440.0):
            sm = AudioSM()
            p = [p for p in hear(sm, tone(f0, 1.5)) if p.pass_message][-1]
            xs.append(p.location[0])
        # an octave up halves the period: one LAG_SCALE closer to the origin
        self.assertAlmostEqual(xs[0] - xs[1], LAG_SCALE, delta=0.1 * LAG_SCALE)

    def test_silence_passes_nothing(self):
        sm = AudioSM()
        for p in hear(sm, np.zeros(SR)):
            self.assertFalse(p.pass_message)
            self.assertIsNone(p.location)

    def test_min_rms_gates_quiet_sounds(self):
        sm = AudioSM(min_rms=0.5)
        self.assertFalse(any(p.pass_message for p in hear(sm, tone(220.0, 1.0))))

    def test_motor_only_step_does_not_pass(self):
        sm = AudioSM()
        hear(sm, tone(220.0, 1.0))
        p = sm.step(None, {"waveform": tone(220.0, 0.25)}, motor_only_step=True)
        self.assertFalse(p.pass_message)

    def test_reset_clears_processed_observations(self):
        sm = AudioSM()
        hear(sm, tone(220.0, 0.5))
        self.assertTrue(sm.state_dict()["processed_observations"])
        sm.reset()
        self.assertEqual(sm.state_dict()["processed_observations"], [])


class AudioEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for name, value in (("a", 1000), ("b", -2000)):
            ramp = (np.arange(SR) % 100 + value).astype(np.int16)
            wavfile.write(self.dir / f"{name}.wav", SR, ramp)

    def tearDown(self):
        self._tmp.cleanup()

    def waveform(self, obs):
        return obs[AgentID("agent_id_0")][SensorID("audio")]["waveform"]

    def test_steps_deliver_consecutive_chunks_and_loop(self):
        env = AudioEnvironment(self.dir, step_seconds=0.25)
        env.add_object("a")
        obs, state = env.reset()
        first = self.waveform(obs)
        self.assertEqual(len(first), int(0.25 * SR))
        chunks = [first] + [self.waveform(env.step([])[0]) for _ in range(4)]
        np.testing.assert_array_equal(chunks[4], chunks[0])  # 1 s file loops
        self.assertIn(SensorID("audio"), state[AgentID("agent_id_0")].sensors)

    def test_new_object_restarts_from_the_top(self):
        env = AudioEnvironment(self.dir)
        env.add_object("a")
        first = self.waveform(env.reset()[0])
        env.step([])
        env.remove_all_objects()
        env.add_object("a")
        np.testing.assert_array_equal(self.waveform(env.reset()[0]), first)

    def test_files_presents_one_recording_under_another_name(self):
        plain = AudioEnvironment(self.dir)
        plain.add_object("b")
        aliased = AudioEnvironment(self.dir, files={"a": "b"})
        aliased.add_object("a")
        np.testing.assert_array_equal(
            self.waveform(aliased.reset()[0]), self.waveform(plain.reset()[0])
        )

    def test_rejects_wrong_sample_rate(self):
        wavfile.write(self.dir / "slow.wav", 22050, np.zeros(100, np.int16))
        env = AudioEnvironment(self.dir)
        with self.assertRaises(ValueError):
            env.add_object("slow")


if __name__ == "__main__":
    unittest.main()
