#!/usr/bin/env python3
"""Generate the computed figures used in the Lecture 4 web note.

The spectral and reconstruction claims in the note are checked numerically here
rather than asserted, so students can re-run and modify the experiments. Run
from the repository root:

    python3 scripts/figures/lecture_04_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _style import CORAL, GOLD, MUTED, PURPLE, TEAL, save, use_style

use_style()


def sensor_graph(nodes: int = 220, radius: float = 0.135, seed: int = 4) -> tuple[nx.Graph, dict]:
    """A random geometric graph: the standard stand-in for a sensor network."""
    while True:
        graph = nx.random_geometric_graph(nodes, radius, seed=seed)
        if nx.is_connected(graph):
            break
        seed += 1
    positions = nx.get_node_attributes(graph, "pos")
    return graph, positions


def laplacian_eigen(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    values, vectors = np.linalg.eigh(laplacian)
    return values, vectors


# --------------------------------------------------------------------------
# Figure 1 — the graph Fourier transform separates smooth from noisy
# --------------------------------------------------------------------------
def figure_gft_spectrum() -> None:
    graph, positions = sensor_graph()
    values, vectors = laplacian_eigen(graph)
    coordinates = np.array([positions[node] for node in graph.nodes()])
    rng = np.random.default_rng(0)

    # A smooth field (a spatial gradient) and pure noise on the same graph.
    smooth = np.sin(3.0 * coordinates[:, 0]) + 0.6 * coordinates[:, 1]
    smooth = (smooth - smooth.mean()) / np.linalg.norm(smooth - smooth.mean())
    noisy = rng.normal(size=graph.number_of_nodes())
    noisy = (noisy - noisy.mean()) / np.linalg.norm(noisy - noisy.mean())

    smooth_hat = vectors.T @ smooth
    noisy_hat = vectors.T @ noisy

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    axes[0].plot(values, np.abs(smooth_hat), color=TEAL, linewidth=1.6, label="smooth field")
    axes[0].plot(values, np.abs(noisy_hat), color=CORAL, linewidth=1.2, alpha=0.85, label="white noise")
    axes[0].set_xlabel(r"graph frequency $\lambda$")
    axes[0].set_ylabel(r"$|\hat{x}(\lambda)|$")
    axes[0].set_title("Graph Fourier coefficients")
    axes[0].legend(loc="upper right", fontsize=10)

    order = np.argsort(values)
    cumulative_smooth = np.cumsum(smooth_hat[order] ** 2)
    cumulative_noisy = np.cumsum(noisy_hat[order] ** 2)
    fraction = np.arange(1, len(values) + 1) / len(values)
    axes[1].plot(100 * fraction, 100 * cumulative_smooth, color=TEAL, linewidth=2.3, label="smooth field")
    axes[1].plot(100 * fraction, 100 * cumulative_noisy, color=CORAL, linewidth=2.3, linestyle="--", label="white noise")
    axes[1].plot([0, 100], [0, 100], color=MUTED, linewidth=1.0, linestyle=":")
    index = int(0.1 * len(values))
    axes[1].annotate(
        f"lowest 10% of frequencies\ncarry {100 * cumulative_smooth[index]:.0f}% of the smooth energy",
        xy=(10, 100 * cumulative_smooth[index]),
        xytext=(38, 14),
        fontsize=9.5,
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=1.2),
    )
    axes[1].set_xlabel("lowest fraction of the spectrum (%)")
    axes[1].set_ylabel("cumulative energy (%)")
    axes[1].set_title("Smooth signals are low-pass; noise is not")
    axes[1].legend(loc="lower right", fontsize=10)

    save(fig, "gft-spectrum")


# --------------------------------------------------------------------------
# Figure 2 — the temporal difference is smoother than the signal itself
# --------------------------------------------------------------------------
def build_time_varying(graph: nx.Graph, steps: int = 120, seed: int = 2) -> np.ndarray:
    """A smoothly evolving field on the graph, in the spirit of sensor data."""
    rng = np.random.default_rng(seed)
    values, vectors = laplacian_eigen(graph)
    modes = 12
    # Slowly drifting low-frequency modes plus a little measurement noise.
    phases = rng.uniform(0, 2 * np.pi, size=modes)
    speeds = rng.uniform(0.02, 0.08, size=modes)
    amplitudes = rng.normal(size=modes) / (1.0 + values[1 : modes + 1])
    signal = np.zeros((graph.number_of_nodes(), steps))
    for step in range(steps):
        weights = amplitudes * np.cos(speeds * step + phases)
        signal[:, step] = vectors[:, 1 : modes + 1] @ weights
    signal += 0.01 * rng.normal(size=signal.shape)
    return signal


def figure_temporal_difference() -> None:
    graph, _ = sensor_graph()
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    signal = build_time_varying(graph)

    difference = np.diff(signal, axis=1)

    def per_column_energy(matrix: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->j", matrix, laplacian @ matrix)

    energy_signal = per_column_energy(signal)
    energy_difference = per_column_energy(difference)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    axes[0].plot(energy_signal, color=CORAL, linewidth=1.9, label=r"$\mathbf{x}_s^{\top}\mathbf{L}\mathbf{x}_s$")
    axes[0].plot(range(1, len(energy_difference) + 1), energy_difference, color=TEAL, linewidth=1.9,
                 label=r"$(\mathbf{x}_s-\mathbf{x}_{s-1})^{\top}\mathbf{L}(\mathbf{x}_s-\mathbf{x}_{s-1})$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time step $s$")
    axes[0].set_ylabel("Laplacian quadratic form")
    axes[0].set_title("Smoothness per time step")
    axes[0].set_ylim(top=float(energy_signal.max()) * 4.0)
    axes[0].legend(loc="upper right", fontsize=9.5)

    ratio = float(np.trace(signal.T @ laplacian @ signal) / np.trace(difference.T @ laplacian @ difference))
    axes[1].bar(
        [0, 1],
        [float(np.trace(signal.T @ laplacian @ signal)), float(np.trace(difference.T @ laplacian @ difference))],
        color=[CORAL, TEAL],
        width=0.55,
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([r"$\mathrm{tr}(\mathbf{X}^{\top}\mathbf{L}\mathbf{X})$",
                             r"$\mathrm{tr}((\mathbf{X}\mathbf{D}_h)^{\top}\mathbf{L}\mathbf{X}\mathbf{D}_h)$"],
                            fontsize=11)
    axes[1].set_ylabel("total variation")
    axes[1].set_title(f"The temporal difference is ${ratio:.0f}\\times$ smoother")
    axes[1].grid(axis="x", visible=False)

    save(fig, "temporal-difference-smoothness")


# --------------------------------------------------------------------------
# Figure 3 — what the smoothness prior buys, and why the Sobolev term exists
# --------------------------------------------------------------------------
def figure_reconstruction_smoothness() -> None:
    graph, _ = sensor_graph()
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    signal = build_time_varying(graph, steps=60)
    steps = signal.shape[1]
    norm = np.linalg.norm(signal)

    difference_operator = np.zeros((steps, steps - 1))
    for index in range(steps - 1):
        difference_operator[index, index] = -1.0
        difference_operator[index + 1, index] = 1.0
    dd = difference_operator @ difference_operator.T
    identity_time = np.eye(steps)

    def reconstruct(mask, observed, temporal, nu, iterations=4000):
        # Gradient descent on the quadratic objective, with the step size taken
        # from the Lipschitz constant of its gradient.
        lipschitz = 1.0 + nu * np.linalg.norm(laplacian, 2) * np.linalg.norm(temporal, 2)
        estimate = observed.copy()
        for _ in range(iterations):
            estimate = estimate - (mask * estimate - observed + nu * (laplacian @ estimate @ temporal)) / lipschitz
        return estimate

    rng = np.random.default_rng(11)
    fractions = [0.1, 0.2, 0.3, 0.5, 0.7]
    curves = {"none": [], "static": [], "difference": []}
    for fraction in fractions:
        mask = (rng.random(signal.shape) < fraction).astype(float)
        observed = mask * signal
        curves["none"].append(np.linalg.norm(observed - signal) / norm)
        # The regularization weight is tuned per setting, so the comparison is
        # between the priors rather than between two arbitrary hyperparameters.
        curves["static"].append(
            min(np.linalg.norm(reconstruct(mask, observed, identity_time, nu) - signal) / norm
                for nu in (0.3, 1.0, 3.0, 10.0))
        )
        curves["difference"].append(
            min(np.linalg.norm(reconstruct(mask, observed, dd, nu) - signal) / norm
                for nu in (0.3, 1.0, 3.0, 10.0))
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    percent = [100 * fraction for fraction in fractions]
    axes[0].semilogy(percent, curves["none"], "o:", color=MUTED, linewidth=1.8, markersize=6,
                     label="no reconstruction")
    axes[0].semilogy(percent, curves["static"], "s--", color=CORAL, linewidth=2.2, markersize=6,
                     label=r"smooth $\mathbf{X}$:  $\mathrm{tr}(\mathbf{X}^{\top}\mathbf{L}\mathbf{X})$")
    axes[0].semilogy(percent, curves["difference"], "o-", color=TEAL, linewidth=2.3, markersize=6,
                     label=r"smooth $\mathbf{X}\mathbf{D}_h$:  $\mathrm{tr}((\mathbf{X}\mathbf{D}_h)^{\top}\mathbf{L}\mathbf{X}\mathbf{D}_h)$")
    axes[0].set_xlabel("observed entries (%)")
    axes[0].set_ylabel("relative reconstruction error")
    axes[0].set_title("Regularizing the temporal difference is what works")
    axes[0].legend(loc="lower left", fontsize=9.5)

    epsilons = np.logspace(-3, 0, 40)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    for beta, color, style in ((1.0, TEAL, "-"), (2.0, GOLD, "--")):
        condition = [((eigenvalues.max() + eps) / (eigenvalues[0] + eps)) ** beta for eps in epsilons]
        axes[1].loglog(epsilons, condition, style, color=color, linewidth=2.2, label=rf"$\beta={beta:.0f}$")
    axes[1].set_xlabel(r"$\epsilon$")
    axes[1].set_ylabel(r"$\kappa\left((\mathbf{L}+\epsilon\mathbf{I})^{\beta}\right)$")
    axes[1].set_title(r"$\kappa(\mathbf{L})=\infty$ because $\lambda_{\min}(\mathbf{L})=0$")
    axes[1].legend(loc="upper right", fontsize=10)

    save(fig, "reconstruction-smoothness")


# --------------------------------------------------------------------------
# Figure 4 — the Chebyshev basis
# --------------------------------------------------------------------------
def figure_chebyshev_polynomials() -> None:
    grid = np.linspace(-1, 1, 600)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))

    polynomials = [np.ones_like(grid), grid]
    for _ in range(2, 6):
        polynomials.append(2 * grid * polynomials[-1] - polynomials[-2])
    colors = (MUTED, TEAL, GOLD, CORAL, PURPLE, "#3f7f5f")
    for order, (values, color) in enumerate(zip(polynomials, colors)):
        axes[0].plot(grid, values, color=color, linewidth=2.0, label=f"$T_{order}$")
    axes[0].set_xlabel(r"$\tilde{\lambda}$")
    axes[0].set_ylabel(r"$T_k(\tilde{\lambda})$")
    axes[0].set_ylim(-1.25, 1.25)
    axes[0].set_title("Chebyshev polynomials on $[-1,1]$")
    axes[0].legend(loc="upper center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.22))

    # A ChebConv layer of order k realizes any degree-k response; here are two.
    low_pass = 0.6 * polynomials[0] - 0.5 * polynomials[1] - 0.12 * polynomials[2]
    band_pass = 0.1 * polynomials[0] + 0.15 * polynomials[1] - 0.55 * polynomials[2] + 0.2 * polynomials[3]
    axes[1].plot(grid, low_pass, color=TEAL, linewidth=2.3, label="a low-pass response")
    axes[1].plot(grid, band_pass, color=CORAL, linewidth=2.3, linestyle="--", label="a band-pass response")
    axes[1].axhline(0.0, color=MUTED, linewidth=0.8)
    axes[1].set_xlabel(r"$\tilde{\lambda}=2\lambda/\lambda_N-1$")
    axes[1].set_ylabel(r"$h(\tilde{\lambda})$")
    axes[1].set_title("A learned combination realizes an arbitrary response")
    axes[1].legend(loc="lower right", fontsize=10)

    save(fig, "chebyshev-polynomials")


def main() -> None:
    figure_gft_spectrum()
    figure_temporal_difference()
    figure_reconstruction_smoothness()
    figure_chebyshev_polynomials()


if __name__ == "__main__":
    main()
