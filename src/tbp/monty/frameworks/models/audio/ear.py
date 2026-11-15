# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The ear as a streaming object: waveform in, SAI frames out.

CARFAC (carfac, the google/carfac NumPy reference -- the same code
a real-time SAI plugin was verified against) runs in streaming mode: its state
rides in the CarfacParams object across calls, so the AGC and hair
cells stay warm between Monty steps exactly as they would listening to
one continuous world. The SAI layer is sai's arithmetic made
stateful: an input ring per channel, one stabilized image, one frame
out per completed hop.

The frame is the sensor's raw observation. Feature extraction belongs
to the sensor module, not here -- the ear hears, it does not opine.
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.models.audio import carfac as C
from tbp.monty.frameworks.models.audio import sai as R

# The compiled cochlea (fastear): same equations, ~20-60x the speed,
# verified against the reference in the unit tests. If numba is absent
# the reference runs instead: slower, same results.
try:
    from tbp.monty.frameworks.models.audio import fastear as _fast

    _run_segment = _fast.run_segment
    _fast_stabilize = _fast._stabilize
    _fast_hops = _fast._hops
except ImportError:  # pragma: no cover
    _fast = None  # type: ignore[assignment]
    _run_segment = C.run_segment
    _fast_stabilize = None  # type: ignore[assignment]
    _fast_hops = None  # type: ignore[assignment]

import logging as _logging

_logging.getLogger(__name__).debug(
    "ear: %s",
    "compiled (numba)"
    if _fast_stabilize is not None
    else "reference CARFAC (install numba for ~20-60x speed)",
)


class Ear:
    """One cochlea plus one stabilized auditory image, streaming."""

    def __init__(self, sample_rate: float) -> None:
        if int(sample_rate) != R.SR:
            raise ValueError(
                f"sai is built for {R.SR} Hz, got {sample_rate}. "
                "Resample the scene, not the ear."
            )
        self.cfp = C.design_carfac(fs=sample_rate)
        C.carfac_init(self.cfp)
        self.n_ch = self.cfp.ears[0].car_coeffs.n_ch
        if self.n_ch != len(R.poles):
            raise AssertionError(
                f"carfac says {self.n_ch} channels, sai says "
                f"{len(R.poles)} -- pole ladders have diverged"
            )
        self._ib = np.zeros((self.n_ch, R.BUF))
        self._img = np.zeros((self.n_ch, R.S))
        self._pending = np.zeros((self.n_ch, 0))
        ## Only these SAI channels are computed (None = all). A reader that
        ## listens in one band sets its band here; the rest were being built
        ## and thrown away. CARFAC still runs every channel -- it is a cascade.
        self.sai_channels = None
        self.sai_lags = None  # compute lags [0, sai_lags) only; None = all
        self.nap_mean = np.zeros(self.n_ch)  # per-channel mean rectified
        # NAP of the LAST chunk fed

    @property
    def image(self) -> np.ndarray:
        """The current stabilized image [n_ch, lags].

        A live view -- copy before mutating.
        """
        return self._img

    def hear(self, waveform: np.ndarray) -> list[np.ndarray]:
        """Feed a mono chunk; return one SAI frame per completed hop.

        Frames are copies, safe to keep. An empty list just means the
        chunk didn't cross a hop boundary yet -- the samples are not
        lost, they are pending.

        Returns:
            The SAI frames [n_ch, lags] completed by this chunk.
        """
        naps, _, _, _, _ = _run_segment(self.cfp, waveform[:, None])
        nap = naps[:, :, 0].T  # [ch, time]
        # timbre wants the whole chunk's sustain, not one strobe instant
        self.nap_mean = np.maximum(nap, 0.0).mean(axis=1)
        self._pending = np.concatenate([self._pending, nap], axis=1)

        if (
            self.sai_channels is not None or self.sai_lags is not None
        ) and _fast is not None:
            ch = (
                self.sai_channels
                if self.sai_channels is not None
                else np.arange(self.n_ch, dtype=np.int64)
            )
            frames, used = _fast._hops_sel(
                np.ascontiguousarray(self._pending),
                self._ib,
                self._img,
                ch,
                R.win,
                self.n_ch,
                R.NT,
                R.W,
                R.WRS,
                R.ORS,
                R.S,
                float(R.WPV),
                int(R.WPK),
                R.HOP,
                R.BUF,
                -1 if self.sai_lags is None else int(self.sai_lags),
            )
            self._pending = self._pending[:, used:]
            return list(frames)
        if _fast_hops is not None:
            frames, used = _fast_hops(
                np.ascontiguousarray(self._pending),
                self._ib,
                self._img,
                R.win,
                self.n_ch,
                R.NT,
                R.W,
                R.WRS,
                R.ORS,
                R.S,
                float(R.WPV),
                int(R.WPK),
                R.HOP,
                R.BUF,
            )
            self._pending = self._pending[:, used:]
            return list(frames)
        frames = []
        while self._pending.shape[1] >= R.HOP:
            seg, self._pending = (self._pending[:, : R.HOP], self._pending[:, R.HOP :])
            self._ib[:, : R.BUF - R.HOP] = self._ib[:, R.HOP :]
            self._ib[:, R.BUF - R.HOP :] = seg
            self._stabilize()
            frames.append(self._img.copy())
        return frames

    def _stabilize(self) -> None:
        """cpp/sai.cc StabilizeSegment, one frame, over the ring."""
        if _fast_stabilize is not None:
            _fast_stabilize(
                self._ib,
                self._img,
                R.win,
                self.n_ch,
                R.NT,
                R.W,
                R.WRS,
                R.ORS,
                R.S,
                float(R.WPV),
                int(R.WPK),
            )
            return
        for i in range(self.n_ch):
            row = self._ib[i]
            for w in range(R.NT):
                wo = int(w * R.W * 0.5)
                prod = row[R.WRS + wo : R.WRS + wo + R.W] * R.win
                tt = int(prod.argmax())
                pv = prod[tt]
                if pv <= 0:
                    pv, tt = R.WPV, R.WPK
                tt += wo
                a = (0.025 + pv) / (0.5 + pv)
                self._img[i] = (
                    self._img[i] * (1 - a) + a * row[tt + R.ORS : tt + R.ORS + R.S]
                )
