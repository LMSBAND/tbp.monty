# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Draw the audio object graphs a Monty LM learned.

    python benchmarks/audio/plot_audio_graphs.py <pretrained/model.pt> [out_dir]

Writes audio_graphs.png (every object's graph: nodes at their LM locations, stored
edges) and audio_transpose.png (best pitch shift between the A- and C-minor takes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def load_graphs(path, lm=0, channel="audio"):
    gm = torch.load(path, weights_only=False)["lm_dict"][lm]["graph_memory"]
    return {o: d[channel]._graph for o, d in gm.items()}


def feature(g, name):
    a, b = g.feature_mapping[name]
    return g.x[:, a:b].numpy().squeeze()


def plot_graphs(graphs, out):
    names = sorted(graphs)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.3 * cols, 3.1 * rows), squeeze=False
    )
    pos_all = np.concatenate([graphs[n].pos.numpy() for n in names])
    lo, hi = pos_all.min(0) - 0.01, pos_all.max(0) + 0.01
    for ax, n in zip(axes.flat, names):
        g = graphs[n]
        pos = g.pos.numpy()
        if getattr(g, "edge_index", None) is not None:
            for a, b in g.edge_index.numpy().T:
                ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=GRID, lw=0.6, zorder=1)
        ax.scatter(
            pos[:, 0], pos[:, 1], s=20, color=BLUE, edgecolor=SURF, lw=1, zorder=3
        )
        ax.set_title(f"{n}  ({len(pos)} nodes)", loc="left", fontsize=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.grid(color=GRID, lw=0.5)
        ax.set_xlabel("log-lag (m, 0.1 per octave)", fontsize=8)
        ax.set_ylabel("cochlear place (m)", fontsize=8)
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


PAIRS = (("swing_am", "swing_cm"), ("chords_am", "chords_cm"))


def plot_transpose(graphs, out, pairs=PAIRS):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.arange(20, 90, 0.5)
    shifts = np.arange(-6, 6.5, 0.5)
    for (a, b), col in zip(pairs, (BLUE, ORANGE)):
        if a not in graphs or b not in graphs:
            continue
        ha = np.histogram(feature(graphs[a], "pitch_semis"), bins)[0].astype(float)
        hb = np.histogram(feature(graphs[b], "pitch_semis"), bins)[0].astype(float)
        sc = np.array([np.dot(np.roll(ha, round(s * 2)), hb) for s in shifts])
        sc /= sc.max()
        best = shifts[int(np.argmax(sc))]
        ax.plot(shifts, sc, color=col, lw=2, label=f"{a} -> {b}: best {best:+.1f}")
    ax.axvline(3, color=INK2, lw=1, ls="--")
    ax.set_xlabel("pitch shift applied to the A-minor graph (semitones)")
    ax.set_ylabel("overlap (normalized)")
    ax.set_title("Key as pose: a minor third is +3", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    model = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path()
    out_dir.mkdir(parents=True, exist_ok=True)
    g = load_graphs(model)
    plot_graphs(g, out_dir / "audio_graphs.png")
    plot_transpose(g, out_dir / "audio_transpose.png")
    print("wrote", out_dir / "audio_graphs.png", out_dir / "audio_transpose.png")
