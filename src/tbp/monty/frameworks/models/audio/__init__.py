# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Hearing for Monty: a CARFAC cochlea and stabilized auditory image as a sensor module.

`sensor_module.AudioSM` turns a waveform observation into one feature-at-location
per step, so the stock learning modules (EvidenceGraphLM, DisplacementGraphLM)
learn and recognize sounds as graphs. To plug it in:

1. Have your environment deliver, each step, a mono float chunk at 44.1 kHz:
   ``observations[agent_id]["audio"] = {"waveform": chunk}``.
2. Use the ``audio`` sensor module config (``monty/sensor_module: audio``) and map
   ``audio`` to your agent and learning module in the connectivity config.
3. Give the learning module tolerances for the AudioSM features, e.g.
   ``pitch_semis: 1.362`` (about 136 cents), ``place_centroid: 8.0`` (channels),
   ``timbre: [0.4, 0.4, 0.4, 0.4]`` (log10 harmonic ratios).

``pip install numba`` (or the ``audio`` extra) compiles the cochlea, ~20-60x
faster; without it the vendored google/carfac NumPy reference runs instead.
"""
