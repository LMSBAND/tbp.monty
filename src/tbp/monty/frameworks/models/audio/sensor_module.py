# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""AudioSM: a cochlea (CARFAC) and stabilized auditory image (SAI) as a sensor module.

Monty's Cortical Messaging Protocol wants one feature-at-location per step,
with a pose. Taken seriously, that decides what an auditory object is here:
the SAI's most salient point traces a path through (lag, place) space as the
sound evolves, and the object is that trajectory with features along it. A
steady drone is a single point revisited, a chirp is a curve, a riff is a
shape. Monty already learns objects as graphs of feature-at-location visits;
this hands it a space where sounds are such graphs natively.

Location: (x, y, z) with
    x = log2(lag / 1 ms) * LAG_SCALE     (octaves of period)
    y = channel * CH_SCALE               (place on the basilar membrane)
    z = 0                                (the frame is flat)
Log-lag because pitch is perceived in ratios: an octave is the same distance
everywhere, so the same melodic shape at two transpositions gives two
translated graphs -- a pose change, not a new object.

Pose: pose_vectors rows are a fixed normal out of the auditory plane, then the
local ridge orientation in-plane and its orthogonal. pose_fully_defined=False:
the in-plane orientation is weak evidence, like curvature directions at a
point on a sphere.

Movement: time. The SAI strobes; successive frames are the saccades, so
displacement in the object's frame comes from the sound's evolution.

Scale: Monty's defaults carry YCB-sized constants (graph dedup 1 mm, match
distance 1 cm). An auditory frame has no intrinsic metres, so it is printed at
a scale where those constants are sensible: at 10 cm/octave a 35-cent vibrato
is a ~3 mm cluster and a one-octave siren a 10 cm path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.audio import sai as R
from tbp.monty.frameworks.models.audio.ear import Ear
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState

LAG_SCALE = 0.1  # metres per octave of lag
CH_SCALE = 0.004  # metres per cochlear channel
SALIENCE_FLOOR = 1.6  # SAI peak must beat this x the mean lag marginal to count
TIMBRE_HARMONICS = (2, 3, 4, 5)


class AudioSM(SensorModule):
    """Sensor module turning waveform observations into auditory percepts.

    Expects observation["waveform"]: a mono float chunk at `sample_rate`
    (one Monty step's worth of audio). The cochlea's state streams across
    steps within an episode and resets between episodes.
    """

    def __init__(
        self,
        sensor_module_id: str = "audio",
        sample_rate: float = 44100.0,
        save_raw_obs: bool = False,
        timbre_source: str = "raw",
        attend: tuple[float, float] | None = None,
        min_rms: float = 0.0,
    ) -> None:
        """Initialize the sensor module.

        Args:
            sensor_module_id: Name of the sensor module.
            sample_rate: Audio sample rate in Hz. The SAI is built for 44100.
            save_raw_obs: Unused; kept for interface parity with CameraSM.
            timbre_source: "raw" computes the timbre feature from harmonic
                bands of the raw waveform (CameraSM-style: the ear supplies
                where, the raw observation supplies what). "ear" samples the
                NAP at harmonic places instead, so timbre passes through the
                cochlea; weaker, because the AGC equalizes spectra.
            attend: Optional (f_lo, f_hi) in Hz restricting the pitch search.
            min_rms: Steps quieter than this RMS emit no percept. 0 disables.
                Useful in spatial scenes where level falls off with distance.
        """
        super().__init__()
        self.sensor_module_id = sensor_module_id
        self.features = [
            "pitch_hz",
            "pitch_semis",
            "level",
            "salience",
            "place_centroid",
            "timbre",
        ]
        self.save_raw_obs = save_raw_obs
        self._sample_rate = sample_rate
        self._timbre_source = timbre_source
        self._attend = attend
        self._min_rms = float(min_rms)
        self._ear = Ear(self._sample_rate)
        self.state: SensorState | None = None
        self.is_exploring = False
        self.processed_obs: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._ear = Ear(self._sample_rate)
        self.processed_obs = []
        self.is_exploring = False

    def update_state(self, agent: AgentState) -> None:
        # A mono ear rides the agent unrotated: it has no gaze.
        self.state = SensorState(position=agent.position, rotation=agent.rotation)

    def state_dict(self):
        return {"processed_observations": self.processed_obs}

    def step(self, ctx, observation, motor_only_step: bool = False) -> Message:  # noqa: ARG002
        wave = np.asarray(observation["waveform"], dtype=np.float64)
        frames = self._ear.hear(wave)
        img = frames[-1] if frames else self._ear.image

        percept = self._percept_from_image(img, wave)
        if motor_only_step:
            percept.pass_message = False
        if not self.is_exploring:
            self.processed_obs.append(
                {
                    "location": None
                    if percept.location is None
                    else percept.location.tolist(),
                    "features": dict(percept.non_morphological_features),
                    "confidence": percept.confidence,
                    "pass_message": percept.pass_message,
                }
            )
        return percept

    def _silent(self) -> Message:
        """Silence, or noise with no stable period: say so and pass nothing.

        The LM never hears about sounds the ear could not vouch for.

        Returns:
            A message with no location that is not passed to the LM.
        """
        return Message(
            location=None,
            morphological_features={},
            non_morphological_features={},
            confidence=0.0,
            pass_message=False,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=False,
        )

    def _percept_from_image(self, img: np.ndarray, wave: np.ndarray) -> Message:
        mg = img.sum(axis=0)
        if self._attend is not None:
            f_lo, f_hi = self._attend
            lo = max(R.LO, int(self._sample_rate / float(f_hi)))
            hi = min(R.HI, int(self._sample_rate / float(f_lo)))
        else:
            lo, hi = R.LO, R.HI
        seg = mg[lo : hi + 1]
        mean = float(seg.mean()) if seg.size else 0.0
        lag = _first_peak_window(mg, lo, hi)

        level = float(np.sqrt(np.mean(np.square(wave))))
        salient = (
            lag is not None
            and mean > 0
            and mg[round(float(lag))] > SALIENCE_FLOOR * mean
            and level >= self._min_rms
        )
        if not salient:
            return self._silent()

        # which place on the membrane carries this period: the energy-weighted
        # centroid of per-channel activity at the winning lag
        li = round(float(lag))
        col = np.maximum(img[:, li], 0.0)
        ch_centroid = (
            float((col * np.arange(len(col))).sum() / col.sum())
            if col.sum() > 0
            else float(len(col) / 2)
        )

        x = np.log2((lag / self._sample_rate) / 0.001) * LAG_SCALE
        y = ch_centroid * CH_SCALE
        location = np.array([x, y, 0.0])

        # local ridge orientation in the (lag, channel) plane, from the image
        # gradient around the salient point -- the analog of curvature
        # directions on a surface patch
        g_lag, g_ch = _local_gradient(img, ch_centroid, li)
        in_plane = np.array([g_lag, g_ch, 0.0])
        n = np.linalg.norm(in_plane)
        dir1 = in_plane / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        dir2 = np.cross([0.0, 0.0, 1.0], dir1)
        pose_vectors = np.vstack([[0.0, 0.0, 1.0], dir1, dir2])

        pitch = self._sample_rate / lag
        semis = 69 + 12 * np.log2(pitch / 440.0)
        salience = float(mg[li] / (mean if mean > 0 else 1.0))

        # Timbre: four numbers -- log10 ratios of harmonics 2..5 to the
        # fundamental, averaged over the step's whole chunk (the sustain, not
        # the strobe instant). Level cancels in the ratio; pitch has its own axis.
        if self._timbre_source == "ear":
            timbre = _timbre_harmonics_ear(self._ear.nap_mean, pitch)
        else:
            timbre = _timbre_harmonics(wave, pitch, self._sample_rate)

        return Message(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": False,
                "on_object": 1,
            },
            non_morphological_features={
                "pitch_hz": float(pitch),
                # log pitch, so a flat LM tolerance is the same number of
                # cents in every octave
                "pitch_semis": float(semis),
                "level": level,
                "salience": salience,
                "place_centroid": ch_centroid,
                "timbre": timbre,
            },
            confidence=float(min(salience / (2 * SALIENCE_FLOOR), 1.0)),
            pass_message=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
            process_features_in_lm=True,
        )


def _first_peak_window(mg, lo, hi, thr=0.80):
    """The pitch rule (see sai.first_peak) on an arbitrary lag window.

    Returns:
        Interpolated lag in samples, or None.
    """
    seg = mg[lo : hi + 1]
    if not seg.size:
        return None
    best = seg.max()
    if best <= 0:
        return None
    for i in range(lo + 1, hi):
        if mg[i] >= thr * best and mg[i] > mg[i - 1] and mg[i] >= mg[i + 1]:
            ym, y0, yp = mg[i - 1], mg[i], mg[i + 1]
            den = ym - 2 * y0 + yp
            d = 0.0 if den == 0 else max(-0.5, min(0.5, 0.5 * (ym - yp) / den))
            return i + d
    return None


def _timbre_harmonics_ear(nap_mean: np.ndarray, pitch: float) -> np.ndarray:
    """Timbre through the ear, from the NAP at harmonic places.

    log10 ratios of mean NAP at harmonic k's membrane place vs the
    fundamental's place. Weaker than the raw version (the AGC equalizes
    spectra -- measured 0.095 separation vs 2.157 raw on the same tones),
    but it reflects only what the cochlea passes on.

    Returns:
        One ratio per harmonic in TIMBRE_HARMONICS, clipped to [-2, 1].
    """

    def at(freq):
        i = int(np.argmin(np.abs(R.poles - freq)))
        i0, i1 = max(i - 1, 0), min(i + 2, len(nap_mean))
        return float(nap_mean[i0:i1].max())

    h1 = at(pitch)
    if h1 <= 1e-12:
        return np.zeros(len(TIMBRE_HARMONICS))
    return np.array(
        [
            np.clip(np.log10(max(at(k * pitch), 1e-12) / h1), -2.0, 1.0)
            for k in TIMBRE_HARMONICS
        ]
    )


def _timbre_harmonics(wave: np.ndarray, pitch: float, sr: float) -> np.ndarray:
    """Timbre from the raw waveform: harmonic band energies vs the fundamental.

    log10 of (harmonic k's band energy / the fundamental's), from
    the step's RAW waveform, k = 2..5. Raw on purpose, and with
    precedent: CameraSM computes curvature from raw depth, not from a
    neural code -- the ear supplies WHERE (pitch, place), the raw
    observation supplies WHAT. NAP-based timbre measured flat: the
    AGC equalizes spectra at full level and un-equalizes them at low
    level. Bands are +-4% around k*pitch so vibrato stays inside its own
    harmonic; ratios cancel level exactly; clip [-2, 1] because a
    harmonic 100x down and an absent one are the same fact.

    Returns:
        One ratio per harmonic in TIMBRE_HARMONICS, clipped to [-2, 1].
    """
    spec = np.abs(np.fft.rfft(wave))
    fbin = sr / len(wave)

    def band(freq):
        b0 = int(freq * 0.96 / fbin)
        b1 = int(freq * 1.04 / fbin) + 1
        return float(spec[max(b0, 0) : min(b1, len(spec))].sum())

    h1 = band(pitch)
    if h1 <= 1e-12:
        return np.zeros(len(TIMBRE_HARMONICS))
    return np.array(
        [
            np.clip(np.log10(max(band(k * pitch), 1e-12) / h1), -2.0, 1.0)
            for k in TIMBRE_HARMONICS
        ]
    )


def _local_gradient(img: np.ndarray, ch: float, lag_i: int) -> tuple[float, float]:
    """Central-difference gradient of the image at the salient point.

    Averaged over a small window so one noisy cell cannot set the
    orientation of the percept.

    Returns:
        (d/d lag, d/d channel) at the point.
    """
    c = round(float(ch))
    c0, c1 = max(c - 2, 1), min(c + 2, img.shape[0] - 2)
    l0, l1 = max(lag_i - 2, 1), min(lag_i + 2, img.shape[1] - 2)
    patch = img[c0 - 1 : c1 + 2, l0 - 1 : l1 + 2]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 1.0, 0.0
    g_ch = float((patch[2:, 1:-1] - patch[:-2, 1:-1]).mean())
    g_lag = float((patch[1:-1, 2:] - patch[1:-1, :-2]).mean())
    return g_lag, g_ch
