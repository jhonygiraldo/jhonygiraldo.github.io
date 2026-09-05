#!/usr/bin/env python3
"""Generate the computed figures used in the Lecture 2 web note.

Every figure in this file is produced from an actual computation, so the
curves and scatter plots in the note report real numbers rather than
illustrations drawn by hand. Run from the repository root:

    python3 scripts/figures/lecture_02_figures.py

Outputs land in assets/figures/ as SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "figures"

INK = "#16202a"
MUTED = "#586874"
LINE = "#c9d4d2"
TEAL = "#0f6c78"
CORAL = "#d2644b"
GOLD = "#c9982d"
PURPLE = "#6d60a8"

plt.rcParams.update(
    {
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
)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} kB")


# --------------------------------------------------------------------------
# Shared graph utilities
# --------------------------------------------------------------------------
def normalized_adjacency(graph: nx.Graph, self_loops: bool = True) -> np.ndarray:
    """Return the GCN propagation operator of a graph."""
    adjacency = nx.to_numpy_array(graph, dtype=float)
    if self_loops:
        adjacency = adjacency + np.eye(adjacency.shape[0])
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def dirichlet_energy(signal: np.ndarray, laplacian: np.ndarray) -> float:
    """Graph Dirichlet energy tr(H^T L H), normalized by the signal norm."""
    numerator = float(np.trace(signal.T @ laplacian @ signal))
    denominator = float(np.linalg.norm(signal) ** 2) + 1e-12
    return numerator / denominator


def edge_homophily(graph: nx.Graph, labels: dict[int, int]) -> float:
    """Node homophily index H(G) as defined in the lecture."""
    ratios = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            continue
        same = sum(labels[node] == labels[other] for other in neighbors)
        ratios.append(same / len(neighbors))
    return float(np.mean(ratios))


# --------------------------------------------------------------------------
# Figure 1 — spectral response of the GCN propagation operator
# --------------------------------------------------------------------------
def figure_spectral_response() -> None:
    rng = np.random.default_rng(7)
    graph = nx.random_partition_graph([60, 60], 0.09, 0.006, seed=11)
    graph = nx.Graph(graph)
    propagation = normalized_adjacency(graph, self_loops=True)

    # Eigen-decomposition of the propagation operator: its eigenvalues live in
    # (-1, 1] and each layer of a GCN raises them to a higher power.
    eigenvalues = np.linalg.eigvalsh(propagation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    grid = np.linspace(-1.0, 1.0, 400)
    axes[0].axvspan(
        eigenvalues.min(),
        eigenvalues.max(),
        color=TEAL,
        alpha=0.08,
        label="spectrum of this graph",
    )
    for layers, color, style in ((1, TEAL, "-"), (2, GOLD, "--"), (8, CORAL, "-."), (32, PURPLE, ":")):
        axes[0].plot(grid, grid**layers, style, color=color, linewidth=2.1, label=f"$L={layers}$")
    axes[0].axhline(0.0, color=MUTED, linewidth=0.8)
    axes[0].set_xlabel(r"eigenvalue $\mu$ of $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$")
    axes[0].set_ylabel(r"filter response $\mu^{L}$")
    axes[0].set_title("Stacking layers sharpens a low-pass filter")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(loc="lower right", fontsize=9.5)

    axes[1].hist(eigenvalues, bins=32, color=TEAL, alpha=0.75, edgecolor="white", linewidth=0.6)
    axes[1].axvline(eigenvalues[0], color=CORAL, linewidth=2.0)
    axes[1].annotate(
        rf"$\mu_{{\max}}={eigenvalues[0]:.2f}$",
        xy=(eigenvalues[0], axes[1].get_ylim()[1] * 0.72),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        color=CORAL,
        fontsize=10,
    )
    axes[1].set_xlabel(r"eigenvalue $\mu$")
    axes[1].set_ylabel("number of eigenvalues")
    axes[1].set_title(f"Spectrum of a two-community graph ($N={graph.number_of_nodes()}$)")

    save(fig, "gcn-spectral-response")


# --------------------------------------------------------------------------
# Figure 2 — homophily and heterophily
# --------------------------------------------------------------------------
def figure_homophily() -> None:
    rng = np.random.default_rng(3)

    homophilous = nx.random_partition_graph([16, 16], 0.34, 0.008, seed=5)
    homophilous = nx.Graph(homophilous)
    labels_homophilous = {node: 0 if node < 16 else 1 for node in homophilous.nodes()}

    # A bipartite-like graph: edges connect the two classes far more often than
    # they connect nodes of the same class.
    heterophilous = nx.Graph()
    heterophilous.add_nodes_from(range(32))
    labels_heterophilous = {node: node % 2 for node in heterophilous.nodes()}
    evens = [n for n in heterophilous if n % 2 == 0]
    odds = [n for n in heterophilous if n % 2 == 1]
    for node in evens:
        for other in rng.choice(odds, size=4, replace=False):
            heterophilous.add_edge(node, int(other))
    for _ in range(6):
        a, b = rng.choice(evens, size=2, replace=False)
        heterophilous.add_edge(int(a), int(b))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    panels = (
        (axes[0], homophilous, labels_homophilous, "Strong homophily"),
        (axes[1], heterophilous, labels_heterophilous, "Strong heterophily"),
    )

    for axis, graph, labels, title in panels:
        positions = nx.spring_layout(graph, seed=17, k=0.55)
        colors = [TEAL if labels[node] == 0 else GOLD for node in graph.nodes()]
        crossing = [(u, v) for u, v in graph.edges() if labels[u] != labels[v]]
        internal = [(u, v) for u, v in graph.edges() if labels[u] == labels[v]]
        nx.draw_networkx_edges(graph, positions, edgelist=internal, ax=axis, edge_color=MUTED, width=1.1, alpha=0.65)
        nx.draw_networkx_edges(graph, positions, edgelist=crossing, ax=axis, edge_color=CORAL, width=1.3, alpha=0.85)
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_color=colors, node_size=175, edgecolors="white", linewidths=1.2
        )
        index = edge_homophily(graph, labels)
        axis.set_title(f"{title}\n$H(G) = {index:.2f}$", fontsize=12)
        axis.set_axis_off()

    handles = [
        Line2D([], [], marker="o", linestyle="", color=TEAL, markersize=9, label="class 1"),
        Line2D([], [], marker="o", linestyle="", color=GOLD, markersize=9, label="class 2"),
        Line2D([], [], color=CORAL, linewidth=2, label="edge between classes"),
        Line2D([], [], color=MUTED, linewidth=2, label="edge inside a class"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    save(fig, "homophily-heterophily")


# --------------------------------------------------------------------------
# Figure 3 — over-smoothing measured on a real propagation
# --------------------------------------------------------------------------
def figure_over_smoothing() -> None:
    rng = np.random.default_rng(0)
    graph = nx.random_partition_graph([70, 70], 0.10, 0.008, seed=13)
    graph = nx.Graph(graph)
    community = np.array([0] * 70 + [1] * 70)

    propagation = normalized_adjacency(graph, self_loops=True)
    laplacian = np.eye(propagation.shape[0]) - propagation

    # Class-dependent random features: initially the two communities are easy
    # to separate. Each layer applies the GCN propagation operator.
    features = rng.normal(size=(140, 16)) * 0.35
    features[community == 0, 0] += 1.6
    features[community == 1, 1] += 1.6

    depths = list(range(0, 41))
    energies = []
    snapshots: dict[int, np.ndarray] = {}
    state = features.copy()
    for depth in depths:
        energies.append(dirichlet_energy(state, laplacian))
        if depth in (1, 5, 20):
            snapshots[depth] = state.copy()
        state = propagation @ state

    fig = plt.figure(figsize=(11.0, 3.1))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.55, 1, 1, 1], wspace=0.28)

    axis = fig.add_subplot(grid[0, 0])
    axis.semilogy(depths, energies, color=TEAL, linewidth=2.2)
    for depth, color in ((1, GOLD), (5, CORAL), (20, PURPLE)):
        axis.plot([depth], [energies[depth]], "o", color=color, markersize=7, zorder=3)
    axis.set_xlabel("number of propagation steps $L$")
    axis.set_ylabel("normalized Dirichlet energy")
    axis.set_title("Feature variation collapses")

    # One fixed projection, shared axis limits: the panels are directly
    # comparable, so the shrinking cloud is the actual embedding collapse and
    # not an artifact of rescaling each panel separately.
    reference = snapshots[1] - snapshots[1].mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(reference, full_matrices=False)
    basis = components[:2].T
    span = np.abs(reference @ basis).max() * 1.3

    for column, (depth, color) in enumerate(((1, GOLD), (5, CORAL), (20, PURPLE)), start=1):
        state = snapshots[depth]
        projected = (state - state.mean(axis=0, keepdims=True)) @ basis
        panel = fig.add_subplot(grid[0, column])
        for value, marker_color in ((0, TEAL), (1, GOLD)):
            mask = community == value
            panel.scatter(
                projected[mask, 0], projected[mask, 1], s=15, color=marker_color, edgecolors="white", linewidths=0.4
            )
        panel.set_xlim(-span, span)
        panel.set_ylim(-span, span)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_aspect("equal")
        panel.grid(False)
        panel.set_title(f"$L={depth}$", color=color)
        for spine in panel.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)

    save(fig, "over-smoothing-collapse")


# --------------------------------------------------------------------------
# Figure 4 — PairNorm and DropEdge slow the collapse
# --------------------------------------------------------------------------
def figure_over_smoothing_remedies() -> None:
    rng = np.random.default_rng(1)
    graph = nx.random_partition_graph([70, 70], 0.10, 0.008, seed=13)
    graph = nx.Graph(graph)
    community = np.array([0] * 70 + [1] * 70)

    features = rng.normal(size=(140, 16)) * 0.35
    features[community == 0, 0] += 1.6
    features[community == 1, 1] += 1.6

    propagation = normalized_adjacency(graph, self_loops=True)
    laplacian = np.eye(propagation.shape[0]) - propagation

    def pair_norm(signal: np.ndarray, scale: float = 1.0) -> np.ndarray:
        centered = signal - signal.mean(axis=0, keepdims=True)
        rescale = np.sqrt(np.mean(np.sum(centered**2, axis=1))) + 1e-12
        return scale * centered / rescale

    def drop_edge_operator(rate: float, seed: int) -> np.ndarray:
        local = nx.Graph(graph)
        generator = np.random.default_rng(seed)
        edges = list(local.edges())
        keep = generator.random(len(edges)) >= rate
        thinned = nx.Graph()
        thinned.add_nodes_from(local.nodes())
        thinned.add_edges_from([edge for edge, flag in zip(edges, keep) if flag])
        return normalized_adjacency(thinned, self_loops=True)

    depths = list(range(0, 41))
    curves: dict[str, list[float]] = {"plain": [], "pairnorm": [], "dropedge": []}

    state = features.copy()
    for _ in depths:
        curves["plain"].append(dirichlet_energy(state, laplacian))
        state = propagation @ state

    state = features.copy()
    for _ in depths:
        curves["pairnorm"].append(dirichlet_energy(state, laplacian))
        state = pair_norm(propagation @ state)

    state = features.copy()
    for depth in depths:
        curves["dropedge"].append(dirichlet_energy(state, laplacian))
        state = drop_edge_operator(0.5, seed=100 + depth) @ state

    fig, axis = plt.subplots(figsize=(7.4, 4.0))
    axis.semilogy(depths, curves["plain"], color=CORAL, linewidth=2.2, label="plain propagation")
    axis.semilogy(depths, curves["dropedge"], color=GOLD, linewidth=2.2, linestyle="--", label="DropEdge ($p=0.5$)")
    axis.semilogy(depths, curves["pairnorm"], color=TEAL, linewidth=2.2, linestyle="-.", label="PairNorm")
    axis.set_xlabel("number of propagation steps $L$")
    axis.set_ylabel("normalized Dirichlet energy")
    axis.set_title("Normalization and edge dropping keep the embeddings apart")
    axis.legend(loc="lower left", fontsize=10)
    save(fig, "over-smoothing-remedies")


# --------------------------------------------------------------------------
# Figure 5 — receptive-field growth
# --------------------------------------------------------------------------
def figure_receptive_field() -> None:
    graphs = {
        "Barabási–Albert ($N=2000$)": nx.barabasi_albert_graph(2000, 3, seed=4),
        "Erdős–Rényi ($N=2000$)": nx.gnm_random_graph(2000, 6000, seed=4),
        "2-D grid ($45 \\times 45$)": nx.grid_2d_graph(45, 45),
    }
    colors = (TEAL, CORAL, GOLD)
    styles = ("-", "--", "-.")

    fig, axis = plt.subplots(figsize=(7.4, 4.1))
    rng = np.random.default_rng(2)
    for (title, graph), color, style in zip(graphs.items(), colors, styles):
        nodes = list(graph.nodes())
        sample = [nodes[index] for index in rng.choice(len(nodes), size=60, replace=False)]
        max_hops = 8
        sizes = np.zeros(max_hops + 1)
        for node in sample:
            lengths = nx.single_source_shortest_path_length(graph, node, cutoff=max_hops)
            counts = np.zeros(max_hops + 1)
            for distance in lengths.values():
                counts[distance] += 1
            sizes += np.cumsum(counts)
        sizes /= len(sample)
        axis.semilogy(range(max_hops + 1), sizes, style, color=color, linewidth=2.2, marker="o", markersize=4.5, label=title)

    axis.set_xlabel("hop distance $k$")
    axis.set_ylabel("nodes within $k$ hops (average)")
    axis.set_title("A $k$-layer message-passing model reads a $k$-hop neighborhood")
    axis.legend(loc="lower right", fontsize=10)
    save(fig, "receptive-field-growth")


# --------------------------------------------------------------------------
# Figure 6 — GCN weights against GAT attention on one neighborhood
# --------------------------------------------------------------------------
def figure_attention_weights() -> None:
    graph = nx.Graph()
    center = 0
    neighbors = [1, 2, 3, 4]
    graph.add_edges_from((center, node) for node in neighbors)
    # Give the neighbors very different degrees so the GCN coefficients differ.
    extra = 5
    for node, degree in zip(neighbors, (1, 4, 9, 2)):
        for _ in range(degree):
            graph.add_edge(node, extra)
            extra += 1

    degrees = dict(graph.degree())
    gcn_weights = np.array(
        [1.0 / np.sqrt((degrees[center] + 1) * (degrees[node] + 1)) for node in [center] + neighbors]
    )
    gcn_weights = gcn_weights / gcn_weights.sum()

    # A learned attention head can weight the same neighborhood differently:
    # these scores come from a fixed random attention vector on random features.
    rng = np.random.default_rng(6)
    features = rng.normal(size=(len(neighbors) + 1, 6))
    attention_vector = rng.normal(size=12)
    scores = []
    for index in range(len(neighbors) + 1):
        pair = np.concatenate([features[0], features[index]])
        scores.append(np.maximum(0.2 * (attention_vector @ pair), attention_vector @ pair))
    attention = np.exp(scores - np.max(scores))
    attention = attention / attention.sum()

    positions = {center: (0.0, 0.0)}
    for index, node in enumerate(neighbors):
        angle = np.pi / 2 + index * 2 * np.pi / len(neighbors)
        positions[node] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})

    # Left: the neighborhood that both models see.
    axis = axes[0]
    for node in neighbors:
        axis.plot(
            [positions[center][0], positions[node][0]],
            [positions[center][1], positions[node][1]],
            color=MUTED,
            linewidth=1.4,
            zorder=1,
        )
    for node in neighbors:
        axis.scatter(*positions[node], s=620, color="white", edgecolors=MUTED, linewidths=1.5, zorder=3)
        axis.annotate(rf"$\mathbf{{h}}_{node}$", xy=positions[node], ha="center", va="center", fontsize=11.5, zorder=4)
        axis.annotate(
            rf"$d_{node}={degrees[node]}$",
            xy=positions[node],
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=MUTED,
        )
    axis.scatter(*positions[center], s=760, color="#e2eff0", edgecolors=TEAL, linewidths=2.0, zorder=3)
    axis.annotate(r"$\mathbf{h}_i$", xy=positions[center], ha="center", va="center", fontsize=12.5, zorder=4)
    axis.set_xlim(-1.75, 1.75)
    axis.set_ylim(-1.75, 1.6)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(rf"One neighborhood ($d_i={degrees[center]}$), seen by both models", fontsize=11.5)

    # Right: the coefficients each model assigns to that neighborhood.
    axis = axes[1]
    names = [r"self $i$"] + [rf"$j={node}$  ($d={degrees[node]}$)" for node in neighbors]
    offsets = np.arange(len(names))
    height = 0.36
    axis.barh(offsets + height / 2, gcn_weights, height=height, color=TEAL, label="GCN (fixed by degrees)")
    axis.barh(offsets - height / 2, attention, height=height, color=CORAL, label="GAT (learned attention)")
    for position, (left, right) in enumerate(zip(gcn_weights, attention)):
        axis.annotate(f"{left:.2f}", xy=(left, position + height / 2), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=9.5, color=TEAL)
        axis.annotate(f"{right:.2f}", xy=(right, position - height / 2), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=9.5, color=CORAL)
    axis.set_yticks(offsets)
    axis.set_yticklabels(names, fontsize=10)
    axis.invert_yaxis()
    axis.set_xlabel("aggregation coefficient")
    axis.set_xlim(0, max(gcn_weights.max(), attention.max()) * 1.22)
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right", fontsize=10)
    axis.set_title("Both sets of coefficients sum to one", fontsize=11.5)

    save(fig, "gcn-vs-gat-weights")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_spectral_response()
    figure_homophily()
    figure_over_smoothing()
    figure_over_smoothing_remedies()
    figure_receptive_field()
    figure_attention_weights()


if __name__ == "__main__":
    main()
