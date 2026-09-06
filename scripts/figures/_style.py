#!/usr/bin/env python3
"""Shared plotting style for the generated lecture-note figures.

Every lecture script imports this so the computed figures share one palette and
one typographic style, and so they sit comfortably beside the LaTeX figures
exported from the slides (hence Latin Modern and Computer Modern math).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "figures"

INK = "#16202a"
MUTED = "#586874"
LINE = "#c9d4d2"
TEAL = "#0f6c78"
CORAL = "#d2644b"
GOLD = "#c9982d"
PURPLE = "#6d60a8"

RC = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "legend.frameon": False,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def use_style() -> None:
    plt.rcParams.update(RC)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} kB")
