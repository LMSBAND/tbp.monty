# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The stabilized auditory image (SAI), in NumPy.

carfac.py turns audio into a neural activity pattern (NAP); this turns
the NAP into SAI frames and pitch readings. Constants and rules follow
google/carfac's cpp/sai.cc.

Layout matches cpp/sai.cc: hop 512, trigger window hop+1, 2 triggers,
50 ms of lag, future_lags = width-1 so the trigger sits at column 0.
"""

import numpy as np

SR = 44100
HOP = 512
NT = 2
S = int(np.ceil(0.050 * SR))
W = HOP + 1
BUF = S + int((1 + (NT - 1) * 0.5) * W)
WS = int(BUF - W - (NT - 1) * W * 0.5)
WRS = WS - (S - 1)
ORS = 1 + WS - S
win = np.sin(np.pi / W + np.arange(W) * (np.pi - np.pi / W) / (W - 1))
WPV, WPK = win.max(), int(win.argmax())
LO, HI = max(2, int(SR * 0.001)), min(S - 2, int(SR * 0.025))

# CARFAC's pole ladder at its default ERB step (83 channels at 44.1 kHz);
# must match carfac.design_carfac's channels, which Ear asserts.
ERB_STEP = 0.5
erb_q = 1000 / (24.7 * 4.37)
_ladder = []
_p = 0.85 * np.pi * SR / (2 * np.pi)
while _p > 30:
    _ladder.append(_p)
    _p -= ERB_STEP * ((165.3 + _p) / erb_q)
poles = np.array(_ladder)

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def sai_frames(nap):
    """Yield one stabilized image per hop over the whole NAP [ch, time].

    Block-at-once StabilizeSegment, the reference schedule. ear.Ear runs
    the same arithmetic streaming, one hop at a time; the unit tests hold
    the two schedules to agreement.
    """
    n_ch, n_s = nap.shape
    ib = np.zeros((n_ch, BUF))
    img = np.zeros((n_ch, S))
    for f in range(n_s // HOP):
        seg = nap[:, f * HOP : (f + 1) * HOP]
        ib[:, : BUF - HOP] = ib[:, HOP:]
        ib[:, BUF - HOP :] = seg
        for i in range(n_ch):
            row = ib[i]
            for w in range(NT):
                wo = int(w * W * 0.5)
                prod = row[WRS + wo : WRS + wo + W] * win
                tt = int(prod.argmax())
                pv = prod[tt]
                if pv <= 0:
                    pv, tt = WPV, WPK
                tt += wo
                a = (0.025 + pv) / (0.5 + pv)
                img[i] = img[i] * (1 - a) + a * row[tt + ORS : tt + ORS + S]
        yield img


def first_peak(mg, thr=0.80):
    """Pitch rule: first local peak over thr of the tallest, parabola on top.

    Returns:
        Interpolated lag in samples, or None.
    """
    best = mg[LO : HI + 1].max()
    if best <= 0:
        return None
    for i in range(LO + 1, HI):
        if mg[i] >= thr * best and mg[i] > mg[i - 1] and mg[i] >= mg[i + 1]:
            ym, y0, yp = mg[i - 1], mg[i], mg[i + 1]
            den = ym - 2 * y0 + yp
            d = 0.0 if den == 0 else max(-0.5, min(0.5, 0.5 * (ym - yp) / den))
            return i + d
    return None


def wmax(mg, target, tol=0.015):
    """Tallest marginal value within +-tol of a target lag.

    Returns:
        The maximum of the marginal in the window, or 0.0 if it is empty.
    """
    c0, c1 = int(target * (1 - tol)), int(target * (1 + tol)) + 1
    c0 = max(c0, 2)
    c1 = min(c1, len(mg) - 1)
    return mg[c0:c1].max() if c1 > c0 else 0.0


def note_name(m):
    m = round(float(m))
    return f"{NAMES[m % 12]}{m // 12 - 1}"
