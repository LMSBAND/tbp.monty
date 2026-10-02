# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The same cochlea, compiled: carfac's sample loop under numba.

carfac is the google/carfac NumPy reference and stays the truth
-- every equation here is transcribed from it, not rederived, and
tests/unit/frameworks/models/audio holds the two to agreement at float precision on
identical input. What numba buys is the loop: the reference steps
Python bytecode 44100 times a second through dataclass attribute
lookups; this runs the identical arithmetic in one nopython pass.

Scope is exactly the configuration the Ear uses, asserted at call
time: mono, two-cap IHC, no synapse model, ripple (no delay
buffer), closed loop, 3-tap FIR spatial smoothing. Anything else
falls back to the reference rather than pretending.

Precision, measured before deciding: the reference's design
coefficients are float32, and numpy's weak promotion makes its
first AGC period (and assorted terms after) compute in float32
before state reassignment drifts everything to float64 -- a
time-varying dtype mix that is an accident of the port, not
physics. This kernel is uniformly float64, so the two agree to
~float32 epsilon RELATIVE (measured 1.6e-7), not to the last bit,
and the fast path is the more precise of the two. The unit
tests hold that bound.

State lives in the CarfacParams dataclasses, as before: extracted
before the kernel, written back after, so reference and fast calls
interleave on one cfp and the AGC stays warm across Monty steps.
The AGC recursion is unrolled into an up-pass (accumulate, decimate
while each stage's phase hits zero) and a down-pass (deepest stage
smooths first, then feeds its memory to the stage above), which is
the same order the reference's recursion visits.
"""

# Kernel arguments keep the reference's constant names (NT, W, HOP, ... from
# sai.py) so each line can be checked against it; _segment transcribes the
# reference's sample loop as one function on purpose.
# ruff: noqa: N803, C901

from __future__ import annotations

import numpy as np
from numba import njit

from tbp.monty.frameworks.models.audio import carfac as C

IHC_A = 0.175  # ihc_detect's tail offset, from the book


@njit(cache=True, nogil=True)
def _segment(
    wave,
    # car coeffs
    r1,
    a0,
    c0,
    h,
    zr,
    ga,
    gb,
    gc,
    ohc_health,
    velocity_scale,
    v_offset,
    ac_coeff,
    # car state
    z1,
    z2,
    za,
    zb,
    dzb,
    zy,
    g,
    dg,
    ac,
    # ihc coeffs (two_cap)
    lpf_coeff,
    out1_rate,
    in1_rate,
    out2_rate,
    in2_rate,
    output_gain,
    rest_output,
    # ihc state
    cap1,
    cap2,
    lpf1,
    ihc_accum,
    # agc coeffs
    decim,
    eps,
    stage_gain,
    fir,
    detect_scale,
    # agc state
    agc_mem,
    agc_accum,
    decim_phase,
    naps,
):
    n = wave.shape[0]
    n_ch = r1.shape[0]
    n_stage = decim.shape[0]
    stage_in = np.zeros((n_stage, n_ch))
    smoothed = np.zeros(n_ch)

    for k in range(n):
        # ---- CAR, rippled (car_step) --------------------------------
        for i in range(n_ch):
            g[i] += dg[i]
            zb[i] += dzb[i]
        in_out = wave[k]
        for i in range(n_ch):
            v = z2[i] - za[i]
            u = v * velocity_scale + v_offset
            nlf = 1.0 / (1.0 + u * u)
            r = r1[i] + zb[i] * nlf
            za[i] = z2[i]
            z1n = r * (a0[i] * z1[i] - c0[i] * z2[i])
            z2n = r * (c0[i] * z1[i] + a0[i] * z2[i])
            z1[i] = z1n + in_out
            z2[i] = z2n
            in_out = g[i] * (in_out + h[i] * z2n)
            zy[i] = in_out

        # ---- IHC two-cap + AGC input (ihc_step) ---------------------
        for i in range(n_ch):
            car_out = zy[i] - ac[i]
            ac[i] += ac_coeff * car_out
            z = car_out + IHC_A
            if z < 0.0:
                z = 0.0
            z2c = z * z
            z3c = z2c * z
            conductance = z3c / (z3c + z2c + 0.1)
            receptor_current = conductance * cap1[i]
            cap1[i] = (
                cap1[i] - receptor_current * out1_rate + (1.0 - cap1[i]) * in1_rate
            )
            ihc_out = (1.0 - cap1[i]) * cap2[i]
            cap2[i] = cap2[i] - ihc_out * out2_rate + (1.0 - cap2[i]) * in2_rate
            ihc_out *= output_gain
            lpf1[i] += lpf_coeff * (ihc_out - lpf1[i])
            nap = lpf1[i] - rest_output
            ihc_accum[i] += nap
            naps[k, i] = nap
            stage_in[0, i] = detect_scale * nap

        # ---- AGC, recursion unrolled (agc_step) ---------------------
        depth = -1
        for s in range(n_stage):
            decim_phase[s] = (decim_phase[s] + 1) % decim[s]
            for i in range(n_ch):
                agc_accum[s, i] += stage_in[s, i]
            if decim_phase[s] != 0:
                break
            inv = 1.0 / decim[s]
            for i in range(n_ch):
                stage_in[s, i] = agc_accum[s, i] * inv
                agc_accum[s, i] = 0.0
            depth = s
            if s + 1 < n_stage:
                for i in range(n_ch):
                    stage_in[s + 1, i] = stage_in[s, i]

        for s in range(depth, -1, -1):
            for i in range(n_ch):
                a_in = stage_in[s, i]
                if s + 1 < n_stage:
                    a_in += stage_gain[s] * agc_mem[s + 1, i]
                agc_mem[s, i] += eps[s] * (a_in - agc_mem[s, i])
            # 3-tap spatial FIR, edges clamped (shift_*_one)
            for i in range(n_ch):
                lo = i - 1 if i > 0 else 0
                hi = i + 1 if i < n_ch - 1 else n_ch - 1
                smoothed[i] = (
                    fir[s, 0] * agc_mem[s, lo]
                    + fir[s, 1] * agc_mem[s, i]
                    + fir[s, 2] * agc_mem[s, hi]
                )
            for i in range(n_ch):
                agc_mem[s, i] = smoothed[i]

        # ---- close the loop (close_agc_loop) ------------------------
        if depth >= 0:
            inv0 = 1.0 / decim[0]
            for i in range(n_ch):
                u = (1.0 - agc_mem[0, i]) * ohc_health[i]
                new_g = ga[i] * u * u + gb[i] * u + gc[i]
                dzb[i] = (zr[i] * u - zb[i]) * inv0
                dg[i] = (new_g - g[i]) * inv0


@njit(cache=True, nogil=True)
def _stabilize(ib, img, win, n_ch, NT, W, WRS, ORS, S, WPV, WPK):
    """Sai's StabilizeSegment, compiled.

    The cochlea went 63x and left this behind: with CARFAC in numba,
    ~95% of a chirp's cost was this double loop in Python. Same
    arithmetic, same order -- strict > in the argmax so ties keep the
    FIRST index, as numpy does.
    """
    for i in range(n_ch):
        for w in range(NT):
            wo = int(w * W * 0.5)
            tt = 0
            pv = ib[i, WRS + wo] * win[0]
            for j in range(1, W):
                v = ib[i, WRS + wo + j] * win[j]
                if v > pv:
                    pv = v
                    tt = j
            if pv <= 0.0:
                pv = WPV
                tt = WPK
            tt += wo
            a = (0.025 + pv) / (0.5 + pv)
            b = 1.0 - a
            for k in range(S):
                img[i, k] = img[i, k] * b + a * ib[i, tt + ORS + k]


def _supported(cfp) -> bool:
    e = cfp.ears[0]
    return (
        cfp.n_ears == 1
        and e.ihc_coeffs.ihc_style == 2
        and not (cfp.syn_params and cfp.syn_params.do_syn)
        and not e.car_coeffs.use_delay_buffer
        and all(
            a.agc_spatial_n_taps == 3 and a.agc_spatial_iterations == 1
            for a in e.agc_coeffs
        )
    )


def run_segment(cfp, input_waves, open_loop=False, linear_car=False):
    """Drop-in for carfac.run_segment on the Ear's configuration.

    Anything off that path goes to the reference untouched.

    Returns:
        The same tuple as carfac.run_segment (naps first).
    """
    if open_loop or linear_car or not _supported(cfp):
        return C.run_segment(cfp, input_waves, open_loop, linear_car)

    wave = np.ascontiguousarray(np.reshape(input_waves, (-1,)), dtype=np.float64)
    e = cfp.ears[0]
    cc, cs = e.car_coeffs, e.car_state
    ic, ihs = e.ihc_coeffs, e.ihc_state

    def f(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.float64))

    z1, z2, za = f(cs.z1_memory), f(cs.z2_memory), f(cs.za_memory)
    zb, dzb = f(cs.zb_memory), f(cs.dzb_memory)
    zy, g, dg = f(cs.zy_memory), f(cs.g_memory), f(cs.dg_memory)
    ac = f(cs.ac_coupler)
    cap1, cap2 = f(ihs.cap1_voltage), f(ihs.cap2_voltage)
    lpf1, accum = f(ihs.lpf1_state), f(ihs.ihc_accum)

    n_stage = len(e.agc_coeffs)
    decim = np.array([a.decimation for a in e.agc_coeffs], dtype=np.int64)
    eps = np.array([a.agc_epsilon for a in e.agc_coeffs])
    gain = np.array([a.agc_stage_gain for a in e.agc_coeffs], dtype=np.float64)
    fir = np.array([a.agc_spatial_fir for a in e.agc_coeffs])
    agc_mem = np.stack([f(s.agc_memory) for s in e.agc_state])
    agc_accum = np.stack([f(s.input_accum) for s in e.agc_state])
    phase = np.array([s.decim_phase for s in e.agc_state], dtype=np.int64)

    naps = np.zeros((wave.shape[0], cc.n_ch))
    _segment(
        wave,
        f(cc.r1_coeffs),
        f(cc.a0_coeffs),
        f(cc.c0_coeffs),
        f(cc.h_coeffs),
        f(cc.zr_coeffs),
        f(cc.ga_coeffs),
        f(cc.gb_coeffs),
        f(cc.gc_coeffs),
        f(cc.ohc_health),
        float(cc.velocity_scale),
        float(cc.v_offset),
        float(cc.ac_coeff),
        z1,
        z2,
        za,
        zb,
        dzb,
        zy,
        g,
        dg,
        ac,
        float(ic.lpf_coeff),
        float(ic.out1_rate),
        float(ic.in1_rate),
        float(ic.out2_rate),
        float(ic.in2_rate),
        float(ic.output_gain),
        float(ic.rest_output),
        cap1,
        cap2,
        lpf1,
        accum,
        decim,
        eps,
        gain,
        fir,
        float(e.agc_coeffs[0].detect_scale),
        agc_mem,
        agc_accum,
        phase,
        naps,
    )

    cs.z1_memory, cs.z2_memory, cs.za_memory = z1, z2, za
    cs.zb_memory, cs.dzb_memory = zb, dzb
    cs.zy_memory, cs.g_memory, cs.dg_memory = zy, g, dg
    cs.ac_coupler = ac
    ihs.cap1_voltage, ihs.cap2_voltage = cap1, cap2
    ihs.lpf1_state, ihs.ihc_accum = lpf1, accum
    for s in range(n_stage):
        e.agc_state[s].agc_memory = agc_mem[s]
        e.agc_state[s].input_accum = agc_accum[s]
        e.agc_state[s].decim_phase = int(phase[s])

    return naps[:, :, None], cfp, None, None, None


@njit(cache=True, nogil=True)
def _hops(pend, ib, img, win, n_ch, NT, W, WRS, ORS, S, WPV, WPK, HOP, BUF):
    """Every complete hop of a chunk in one call.

    Shift the ring, append the hop, stabilize, keep the frame. Same arithmetic
    and order as Ear.hear's per-hop Python loop; no quadratic re-slicing of the
    pending buffer.

    Returns:
        (frames [n, n_ch, S], samples consumed).
    """
    n = pend.shape[1] // HOP
    frames = np.empty((n, n_ch, S))
    for f in range(n):
        for i in range(n_ch):
            for t in range(BUF - HOP):
                ib[i, t] = ib[i, t + HOP]
            for t in range(HOP):
                ib[i, BUF - HOP + t] = pend[i, f * HOP + t]
        _stabilize(ib, img, win, n_ch, NT, W, WRS, ORS, S, WPV, WPK)
        frames[f] = img
    return frames, n * HOP


@njit(cache=True, nogil=True)
def _hops_sel(
    pend, ib, img, chans, win, n_ch, NT, W, WRS, ORS, S, WPV, WPK, HOP, BUF, S_use=-1
):
    """_hops, but only for the channels someone will read (chans).

    Each SAI channel depends only on its own NAP, so the channels computed are
    bit-identical to a full run; the others stay at zero.

    Returns:
        (frames [n, n_ch, S_use or S], samples consumed).
    """
    n = pend.shape[1] // HOP
    s_use = S if S_use < 0 else S_use  # lags past s_use are never read: not computed
    frames = np.zeros((n, n_ch, s_use))
    for ci in range(chans.shape[0]):
        i = chans[ci]
        for f in range(n):
            for t in range(BUF - HOP):
                ib[i, t] = ib[i, t + HOP]
            for t in range(HOP):
                ib[i, BUF - HOP + t] = pend[i, f * HOP + t]
            for w in range(NT):
                wo = int(w * W * 0.5)
                tt = 0
                pv = ib[i, WRS + wo] * win[0]
                for j in range(1, W):
                    v = ib[i, WRS + wo + j] * win[j]
                    if v > pv:
                        pv = v
                        tt = j
                if pv <= 0.0:
                    pv = WPV
                    tt = WPK
                tt += wo
                a = (0.025 + pv) / (0.5 + pv)
                b = 1.0 - a
                for k in range(s_use):
                    img[i, k] = img[i, k] * b + a * ib[i, tt + ORS + k]
            for k in range(s_use):
                frames[f, i, k] = img[i, k]
    return frames, n * HOP
