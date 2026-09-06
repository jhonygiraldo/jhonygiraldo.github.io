#!/usr/bin/env python3
"""Generate the computed figures used in the Lecture 6 web note.

The lecture's central empirical claim is that Erdos-Renyi graphs match real
networks on two properties and fail on two others. That comparison is measured
here rather than quoted, on a real network shipped with networkx plus a
preferential-attachment model. Run from the repository root:

    python3 scripts/figures/lecture_06_figures.py
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _style import CORAL, GOLD, MUTED, PURPLE, TEAL, save, use_style

use_style()


def giant_component(graph: nx.Graph) -> nx.Graph:
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def average_path_length(graph: nx.Graph, sample: int = 400, seed: int = 0) -> float:
    """Mean shortest-path length inside the giant component, sampled if large."""
    component = giant_component(graph)
    nodes = list(component.nodes())
    rng = np.random.default_rng(seed)
    if len(nodes) > sample:
        nodes = [nodes[i] for i in rng.choice(len(nodes), size=sample, replace=False)]
    total, count = 0, 0
    for node in nodes:
        for distance in nx.single_source_shortest_path_length(component, node).values():
            if distance > 0:
                total += distance
                count += 1
    return total / max(count, 1)


def matched_erdos_renyi(graph: nx.Graph, seed: int = 3) -> nx.Graph:
    """G(n, p) with the same node count and the same average degree."""
    nodes = graph.number_of_nodes()
    p = 2 * graph.number_of_edges() / (nodes * (nodes - 1))
    return nx.gnp_random_graph(nodes, p, seed=seed)


# --------------------------------------------------------------------------
# Figure 1 — where Erdos-Renyi matches real networks, and where it fails
# --------------------------------------------------------------------------
def figure_real_vs_random() -> None:
    # Les Miserables co-appearance is genuinely observed data but small, so the
    # models are drawn larger: the contrast we care about is in the tail, and a
    # 77-node histogram cannot show a tail.
    real = nx.Graph(nx.les_miserables_graph())
    scale_free = nx.powerlaw_cluster_graph(5000, 3, 0.4, seed=5)
    random_graph = matched_erdos_renyi(scale_free)

    graphs = (
        (f"Les Misérables, real ($N={real.number_of_nodes()}$)", real, TEAL, "-"),
        (f"preferential attachment ($N=5000$)", scale_free, GOLD, "--"),
        (f"Erdős–Rényi $G_{{np}}$ ($N=5000$)", random_graph, CORAL, "-."),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    # The complementary CDF, P(K >= k), is the readable way to compare tails:
    # it needs no binning choice and stays smooth on a small sample.
    for label, graph, color, style in graphs:
        degrees = np.array([degree for _, degree in graph.degree()])
        ks = np.arange(1, degrees.max() + 1)
        ccdf = [(degrees >= k).mean() for k in ks]
        axes[0].loglog(ks, ccdf, style, color=color, linewidth=2.2, label=label)
    axes[0].set_xlabel("degree $k$")
    axes[0].set_ylabel(r"$P(K \geq k)$")
    axes[0].set_title("Degree distribution (complementary CDF)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].annotate("random graphs have\nno heavy tail", xy=(13, 6e-4), xytext=(26, 8e-2),
                     fontsize=9.5, color=CORAL,
                     arrowprops=dict(arrowstyle="->", color=CORAL, linewidth=1.1))

    labels = ["clustering\ncoefficient $C$", "avg. path\nlength $\\bar{h}$",
              "largest component\n(fraction of $N$)"]
    positions = np.arange(len(labels))
    width = 0.26
    matched = graphs[1:]
    for index, (label, graph, color, _) in enumerate(matched):
        values = [
            nx.average_clustering(graph),
            average_path_length(graph),
            giant_component(graph).number_of_nodes() / graph.number_of_nodes(),
        ]
        offset = (index - 0.5) * width
        bars = axes[1].bar(positions + offset, values, width=width, color=color)
        for bar, value in zip(bars, values):
            axes[1].annotate(f"{value:.2f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                             xytext=(0, 3), textcoords="offset points", ha="center",
                             fontsize=8.5, color=color)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=9.5)
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylim(0, 6.4)
    axes[1].legend([label for label, *_ in matched], loc="upper left", fontsize=9)
    axes[1].set_title("Same $N$ and same average degree: only clustering differs")

    save(fig, "real-vs-random-graphs")


# --------------------------------------------------------------------------
# Figure 2 — the giant component appears at average degree one
# --------------------------------------------------------------------------
def figure_giant_component() -> None:
    nodes = 2000
    degrees = np.linspace(0.1, 3.0, 40)
    rng = np.random.default_rng(1)

    fractions, clustering = [], []
    for average_degree in degrees:
        p = average_degree / (nodes - 1)
        sizes, coefficients = [], []
        for repeat in range(3):
            graph = nx.gnp_random_graph(nodes, p, seed=int(rng.integers(1e6)))
            sizes.append(len(max(nx.connected_components(graph), key=len)) / nodes)
            coefficients.append(nx.average_clustering(graph))
        fractions.append(np.mean(sizes))
        clustering.append(np.mean(coefficients))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    axes[0].plot(degrees, [100 * value for value in fractions], color=TEAL, linewidth=2.3)
    axes[0].axvline(1.0, color=CORAL, linewidth=1.8, linestyle="--")
    axes[0].annotate(r"$\bar{k}=1$", xy=(1.0, 82), xytext=(8, 0), textcoords="offset points",
                     color=CORAL, fontsize=11)
    axes[0].set_xlabel(r"average degree $\bar{k}=p(N-1)$")
    axes[0].set_ylabel("largest component (% of $N$)")
    axes[0].set_title(f"The giant component emerges at $\\bar{{k}}=1$  ($N={nodes}$)")

    # C = k/N: fix the average degree and clustering vanishes as the graph grows.
    sizes = np.logspace(2, 5, 40)
    for average_degree, color, style in ((5.0, TEAL, "-"), (20.0, GOLD, "--")):
        axes[1].loglog(sizes, average_degree / sizes, style, color=color, linewidth=2.2,
                       label=rf"$\bar{{k}}={average_degree:.0f}$")
    axes[1].axhline(0.11, color=CORAL, linewidth=1.8, linestyle=":")
    axes[1].annotate("MSN Messenger: $C=0.11$", xy=(2.5e3, 0.16), fontsize=9.5, color=CORAL)
    axes[1].set_xlabel("number of nodes $N$")
    axes[1].set_ylabel(r"$\mathbb{E}[C]=\bar{k}/N$")
    axes[1].set_title("Random-graph clustering vanishes with size")
    axes[1].legend(loc="lower left", fontsize=10)

    save(fig, "erdos-renyi-properties")


# --------------------------------------------------------------------------
# Figure 3 — the cost of generating a graph one edge decision at a time
# --------------------------------------------------------------------------
def figure_generation_cost() -> None:
    nodes = np.arange(10, 2001)

    fig, axis = plt.subplots(figsize=(7.6, 4.2))
    axis.loglog(nodes, nodes * (nodes + 1) / 2, color=CORAL, linewidth=2.4,
                label=r"GraphRNN: $\mathcal{O}(N^2/2)$ edge decisions")
    axis.loglog(nodes, nodes * np.log2(nodes), color=TEAL, linewidth=2.4, linestyle="--",
                label=r"local expansion: sub-quadratic")
    axis.loglog(nodes, nodes, color=MUTED, linewidth=1.4, linestyle=":", label=r"$N$ (for reference)")

    for size in (100, 1000):
        quadratic = size * (size + 1) / 2
        axis.plot([size], [quadratic], "o", color=CORAL, markersize=7, zorder=3)
        axis.annotate(f"$N={size}$: {quadratic:,.0f}", xy=(size, quadratic), xytext=(-8, 10),
                      textcoords="offset points", ha="right", fontsize=9.5, color=CORAL)

    axis.set_xlabel("number of nodes $N$")
    axis.set_ylabel("generation steps")
    axis.set_title("Why sequential edge generation does not reach large graphs")
    axis.legend(loc="upper left", fontsize=10)
    save(fig, "graph-generation-cost")


def main() -> None:
    figure_real_vs_random()
    figure_giant_component()
    figure_generation_cost()


if __name__ == "__main__":
    main()
