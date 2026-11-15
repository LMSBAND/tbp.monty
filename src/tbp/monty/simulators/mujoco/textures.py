# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Texture for a touch sensor, from recorded data.

MuJoCo's meshes carry no surface texture, and the contact solver knows force,
friction and compliance but not what a surface feels like under a sliding
fingertip. Data-driven texture models do: the Penn Haptic Texture Toolkit (HaTT;
Culbertson, Lopez Delgado and Kuchenbecker, IEEE Haptics Symposium 2014) recorded
a tool dragged over 100 real surfaces and fitted, at a set of (speed, normal
force) points, an ARMA model of the tool's vibration (acceleration, m/s^2,
1 kHz). Rendering a texture is interpolating that model at the current speed and
force over the toolkit's Delaunay triangulation (line spectral frequencies,
variance and gain barycentrically, as their renderer does) and driving it with
white noise.

The HaTT model files are not part of this repository (their licence allows
non-profit research use and no redistribution); point ``texture_path`` at a
directory holding the toolkit's ``Models_<material>.xml`` files.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

__all__ = ["HaTTTexture", "load_texture"]


def _values(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.zeros(0)
    return np.array([float(v.text) for v in node.findall("value")])


def lsf_to_poly(lsf: np.ndarray) -> np.ndarray:
    """Line spectral frequencies (rad, ascending) to a monic polynomial.

    For an order-p polynomial A(z), P(z) = A(z) + z^-(p+1) A(1/z) and
    Q(z) = A(z) - z^-(p+1) A(1/z) have their roots on the unit circle at the
    alternating LSFs (the first to P), plus z = -1 for P and z = +1 for Q when p
    is even; A = (P + Q) / 2.

    Returns:
        [1, a1, ..., ap].
    """
    lsf = np.sort(np.asarray(lsf, dtype=np.float64))
    p = len(lsf)
    poly_p = np.array([1.0])
    poly_q = np.array([1.0])
    for i, w in enumerate(lsf):
        factor = np.array([1.0, -2.0 * np.cos(w), 1.0])
        if i % 2 == 0:
            poly_p = np.convolve(poly_p, factor)
        else:
            poly_q = np.convolve(poly_q, factor)
    if p % 2 == 0:
        poly_p = np.convolve(poly_p, [1.0, 1.0])
        poly_q = np.convolve(poly_q, [1.0, -1.0])
    else:
        poly_q = np.convolve(poly_q, [1.0, 0.0, -1.0])
    return (0.5 * (poly_p + poly_q))[: p + 1]


class HaTTTexture:
    """One HaTT texture: the vibration a sliding tool feels on it."""

    def __init__(self, path: str | Path) -> None:
        root = ET.parse(path).getroot()  # noqa: S314  (a local data file)
        self.material = root.findtext("material", default=Path(path).stem)
        self.sample_rate = float(root.findtext("SampleRate", default="1000"))
        self.mu = float(root.findtext("mu", default="nan"))
        self.max_speed = float(root.findtext("maxSpeed"))
        self.max_force = float(root.findtext("maxForce"))
        speeds, forces, ar, ma, var, gain = [], [], [], [], [], []
        for model in root.findall("model"):
            speeds.append(float(model.findtext("speedMod")))
            forces.append(float(model.findtext("forceMod")))
            ar.append(_values(model.find("ARlsf")))
            ma.append(_values(model.find("MAlsf")))
            var.append(float(model.findtext("var")))
            gain.append(float(model.findtext("gain")))
        self.points = np.column_stack([speeds, forces])  # mm/s, N
        self.ar_lsf = np.array(ar)
        self.ma_lsf = np.array(ma)
        self.var = np.array(var)
        self.gain = np.array(gain)
        self.triangles = (
            np.array([_values(t) for t in root.findall("tri")], dtype=int) - 1
        )

    def _weights(self, speed: float, force: float) -> tuple[np.ndarray, np.ndarray]:
        """Model indices and barycentric weights at (speed mm/s, force N).

        Outside the triangulation the nearest model alone is used.

        Returns:
            The indices of the models to blend, and their weights.
        """
        x = np.array(
            [np.clip(speed, 0.0, self.max_speed), np.clip(force, 0.0, self.max_force)]
        )
        for tri in self.triangles:
            a, b, c = self.points[tri]
            m = np.column_stack([b - a, c - a])
            if abs(np.linalg.det(m)) < 1e-12:
                continue
            l1, l2 = np.linalg.solve(m, x - a)
            w = np.array([1.0 - l1 - l2, l1, l2])
            if np.all(w >= -1e-9):
                return tri, np.clip(w, 0.0, 1.0)
        nearest = int(np.argmin(np.linalg.norm(self.points - x, axis=1)))
        return np.array([nearest]), np.array([1.0])

    def vibration(
        self, speed: float, force: float, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """A vibration trace (m/s^2, ``sample_rate``) for a stroke.

        Args:
            speed: stroke speed, mm/s.
            force: normal force, N.
            n: number of samples.
            rng: noise source.

        Returns:
            The acceleration samples.
        """
        idx, w = self._weights(speed, force)
        ar = lsf_to_poly(w @ self.ar_lsf[idx])
        ma = lsf_to_poly(w @ self.ma_lsf[idx]) if self.ma_lsf.shape[1] else np.ones(1)
        var = float(w @ self.var[idx])
        gain = float(w @ self.gain[idx])
        settle = 200  # let the filter forget its zero start
        noise = rng.normal(0.0, np.sqrt(max(var, 0.0)), n + settle)
        return (gain * lfilter(ma, ar, noise))[settle:]


@cache
def load_texture(texture_path: str, material: str) -> HaTTTexture:
    """Loads ``Models_<material>.xml`` from ``texture_path`` (cached).

    Returns:
        The texture.
    """
    return HaTTTexture(Path(texture_path).expanduser() / f"Models_{material}.xml")
