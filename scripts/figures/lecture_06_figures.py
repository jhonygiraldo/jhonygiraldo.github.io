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

from _style import CORAL, GOLD, INK, MUTED, PURPLE, TEAL, save, use_style

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

    # Clustering is O(1e-3) for G(n,p) while path length is O(5). On a shared
    # linear axis the clustering bars vanish, so plot the ratio instead: how
    # many times larger the model's value is than the random graph's.
    matched = graphs[1:]
    labels = ["clustering\ncoefficient $C$", "avg. path\nlength $\\bar{h}$",
              "largest component\n(fraction of $N$)"]
    scale_free_values = [
        nx.average_clustering(matched[0][1]),
        average_path_length(matched[0][1]),
        giant_component(matched[0][1]).number_of_nodes() / matched[0][1].number_of_nodes(),
    ]
    random_values = [
        nx.average_clustering(matched[1][1]),
        average_path_length(matched[1][1]),
        giant_component(matched[1][1]).number_of_nodes() / matched[1][1].number_of_nodes(),
    ]
    ratios = [a / b if b > 0 else np.inf for a, b in zip(scale_free_values, random_values)]
    positions = np.arange(len(labels))
    colors = [CORAL if r > 3 or r < 1 / 3 else MUTED for r in ratios]
    axes[1].bar(positions, ratios, width=0.5, color=colors)
    axes[1].axhline(1.0, color=TEAL, linewidth=1.6)
    axes[1].annotate("equal", xy=(2.42, 1.0), xytext=(0, 4), textcoords="offset points",
                     fontsize=9, color=TEAL, ha="right")
    for position, (ratio, a, b) in enumerate(zip(ratios, scale_free_values, random_values)):
        axes[1].annotate(f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.2f}×",
                         xy=(position, ratio), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=9.5, color=INK)
        labels[position] = f"{labels[position]}\n{a:.3g} vs {b:.3g}"
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=9.5)
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylabel("preferential attachment $/$ Erdős–Rényi")
    axes[1].set_title("Two synthetic models, same $N$ and mean degree")

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
    axis.loglog(nodes, nodes * (nodes - 1) / 2, color=CORAL, linewidth=2.4,
                label=r"GraphRNN worst case: $N(N-1)/2$ decisions")
    axis.loglog(nodes, nodes * np.log2(nodes), color=TEAL, linewidth=2.4, linestyle="--",
                label=r"$N\log_2 N$ (reference slope, not a measurement)")
    axis.loglog(nodes, nodes, color=MUTED, linewidth=1.4, linestyle=":", label=r"$N$ (for reference)")

    for size in (100, 1000):
        quadratic = size * (size - 1) / 2
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
