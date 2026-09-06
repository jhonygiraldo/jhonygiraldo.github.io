#!/usr/bin/env python3
"""Generate the computed figures used in the Lecture 3 web note.

Each figure reports a measurement rather than an illustration, so the claims in
the note about mini-batching, computational-graph growth, and clustering can be
checked and re-run. Run from the repository root:

    python3 scripts/figures/lecture_03_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _style import CORAL, GOLD, MUTED, PURPLE, TEAL, save, use_style

use_style()


def load_graph(seed: int = 12) -> nx.Graph:
    """A graph with communities and a heavy-tailed degree distribution.

    Both properties matter here: communities are what Cluster-GCN exploits, and
    the degree tail is what makes hub nodes blow up a computational graph.
    """
    graph = nx.powerlaw_cluster_graph(4000, 5, 0.35, seed=seed)
    return graph


# --------------------------------------------------------------------------
# Figure 1 — uniform mini-batches destroy the edges
# --------------------------------------------------------------------------
def figure_minibatch_edge_loss() -> None:
    graph = load_graph()
    rng = np.random.default_rng(0)
    nodes = np.array(graph.nodes())
    total_edges = graph.number_of_edges()

    sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    uniform_kept, uniform_isolated = [], []
    for size in sizes:
        kept, isolated = [], []
        for _ in range(12):
            batch = rng.choice(nodes, size=size, replace=False)
            induced = graph.subgraph(batch)
            kept.append(induced.number_of_edges() / total_edges)
            isolated.append(sum(1 for _, d in induced.degree() if d == 0) / size)
        uniform_kept.append(np.mean(kept))
        uniform_isolated.append(np.mean(isolated))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    axes[0].loglog(sizes, uniform_kept, "o-", color=CORAL, linewidth=2.2, markersize=6, label="measured")
    # A uniform sample of M of N nodes keeps an edge only when both endpoints
    # are drawn, which happens with probability of roughly (M/N)^2.
    reference = [(size / graph.number_of_nodes()) ** 2 for size in sizes]
    axes[0].loglog(sizes, reference, "--", color=MUTED, linewidth=1.6, label=r"$(M/N)^2$")
    axes[0].set_xlabel("mini-batch size $M$")
    axes[0].set_ylabel("fraction of edges inside the batch")
    axes[0].set_title("A uniform mini-batch keeps almost no edges")
    axes[0].legend(loc="upper left", fontsize=10)

    axes[1].semilogx(sizes, [100 * value for value in uniform_isolated], "o-", color=CORAL, linewidth=2.2, markersize=6)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("mini-batch size $M$")
    axes[1].set_ylabel("isolated nodes in the batch (%)")
    axes[1].set_title(f"Most sampled nodes have no neighbor\n($N={graph.number_of_nodes()}$, "
                      f"$|\\mathcal{{E}}|={total_edges}$)")

    save(fig, "minibatch-edge-loss")


# --------------------------------------------------------------------------
# Figure 2 — how large a computational graph really gets
# --------------------------------------------------------------------------
def figure_computational_graph_growth() -> None:
    graph = load_graph()
    rng = np.random.default_rng(1)
    nodes = list(graph.nodes())
    sample = [nodes[i] for i in rng.choice(len(nodes), size=120, replace=False)]

    depths = range(1, 6)
    exact = []
    for depth in depths:
        sizes = []
        for node in sample:
            reached = nx.single_source_shortest_path_length(graph, node, cutoff=depth)
            sizes.append(len(reached))
        exact.append(np.mean(sizes))

    fig, axis = plt.subplots(figsize=(7.6, 4.2))
    axis.semilogy(list(depths), exact, "o-", color=CORAL, linewidth=2.3, markersize=6.5,
                  label="full $K$-hop neighborhood (measured)")
    for fan_out, color, style in ((2, TEAL, "--"), (5, GOLD, "-."), (10, PURPLE, ":")):
        bound = [sum(fan_out**k for k in range(depth + 1)) for depth in depths]
        axis.semilogy(list(depths), bound, style, color=color, linewidth=2.1, marker="s", markersize=4.5,
                      label=f"neighbor sampling, $H={fan_out}$")
    axis.axhline(graph.number_of_nodes(), color=MUTED, linewidth=1.2, linestyle="-")
    axis.annotate(f"whole graph, $N={graph.number_of_nodes()}$", xy=(1.05, graph.number_of_nodes()),
                  xytext=(0, 6), textcoords="offset points", fontsize=9.5, color=MUTED)
    axis.set_xlabel("number of GNN layers $K$")
    axis.set_ylabel("nodes in one node's computational graph")
    axis.set_xticks(list(depths))
    axis.set_title("Sampling caps the fan-out, but the growth stays exponential in $K$")
    axis.legend(loc="upper left", fontsize=9.5)
    save(fig, "computational-graph-growth")


# --------------------------------------------------------------------------
# Figure 3 — what a Cluster-GCN partition throws away
# --------------------------------------------------------------------------
def figure_cluster_partition_loss() -> None:
    graph = nx.powerlaw_cluster_graph(1500, 5, 0.5, seed=7)
    total_edges = graph.number_of_edges()
    rng = np.random.default_rng(3)
    nodes = list(graph.nodes())

    def kept_fraction(partition: list[set]) -> float:
        label = {}
        for index, part in enumerate(partition):
            for node in part:
                label[node] = index
        inside = sum(1 for u, v in graph.edges() if label[u] == label[v])
        return inside / total_edges

    # Community detection at several resolutions gives partitions of different
    # sizes; a random partition of the same sizes is the control.
    community_points, random_points = [], []
    for resolution in (0.4, 0.7, 1.0, 1.6, 2.6, 4.0, 7.0, 12.0):
        communities = nx.community.greedy_modularity_communities(graph, resolution=resolution)
        count = len(communities)
        community_points.append((count, kept_fraction(list(communities))))

        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(np.array(shuffled), count)
        random_points.append((count, kept_fraction([set(chunk.tolist()) for chunk in chunks])))

    community_points.sort()
    random_points.sort()

    fig, axis = plt.subplots(figsize=(7.6, 4.2))
    axis.plot([c for c, _ in community_points], [100 * f for _, f in community_points], "o-",
              color=TEAL, linewidth=2.3, markersize=6.5, label="community partition (Cluster-GCN)")
    axis.plot([c for c, _ in random_points], [100 * f for _, f in random_points], "s--",
              color=CORAL, linewidth=2.1, markersize=5.5, label="random partition of the same sizes")
    axis.set_xscale("log")
    axis.set_ylim(0, 100)
    axis.set_xlabel("number of partitions $C$")
    axis.set_ylabel("edges kept inside the sub-graphs (%)")
    axis.set_title(f"Partitioning by community preserves far more edges\n"
                   f"($N={graph.number_of_nodes()}$, $|\\mathcal{{E}}|={total_edges}$)")
    axis.legend(loc="lower left", fontsize=10)
    save(fig, "cluster-partition-edge-loss")


# --------------------------------------------------------------------------
# Figure 4 — why SGC works, and when it stops working
# --------------------------------------------------------------------------
def figure_sgc_homophily() -> None:
    rng = np.random.default_rng(5)
    depths = [0, 1, 2, 3, 4, 6, 8]
    # H=0.05 is included deliberately: near-bipartite graphs have large negative
    # propagation eigenvalues, so separability RISES there. Omitting it invites
    # the false reading that low homophily always kills the signal.
    homophily_levels = [
        (0.95, TEAL, "-"),
        (0.70, GOLD, "--"),
        (0.30, PURPLE, ":"),
        (0.05, CORAL, "-."),
    ]

    fig, axis = plt.subplots(figsize=(7.6, 4.2))

    for target, color, style in homophily_levels:
        # Two equal communities; p_in / (p_in + p_out) fixes the homophily.
        density = 0.06
        p_in = 2 * density * target
        p_out = 2 * density * (1 - target)
        graph = nx.Graph(nx.random_partition_graph([150, 150], p_in, p_out, seed=21))
        community = np.array([0] * 150 + [1] * 150)

        adjacency = nx.to_numpy_array(graph) + np.eye(300)
        degree = adjacency.sum(axis=1)
        inverse = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        propagation = inverse[:, None] * adjacency * inverse[None, :]

        features = rng.normal(size=(300, 32)) * 1.0
        features[community == 0, :4] += 0.9
        features[community == 1, 4:8] += 0.9

        separations = []
        state = features.copy()
        for depth in range(max(depths) + 1):
            if depth in depths:
                # Fisher-style separability: between-class distance over
                # within-class spread, i.e. how easy a linear model has it.
                mean_a = state[community == 0].mean(axis=0)
                mean_b = state[community == 1].mean(axis=0)
                between = np.linalg.norm(mean_a - mean_b)
                within = np.sqrt(state[community == 0].var(axis=0).sum() + state[community == 1].var(axis=0).sum())
                separations.append(between / (within + 1e-12))
            state = propagation @ state

        axis.plot(depths, separations, style, color=color, linewidth=2.2, marker="o", markersize=5,
                  label=f"$H(\\mathcal{{G}}) \\approx {target:.2f}$")

    axis.set_xlabel(r"diffusion steps $K$ in $\hat{\mathbf{A}}^{K}\mathbf{X}$")
    axis.set_ylabel("class separability of the features")
    axis.set_title("Diffusion depth interacts with homophily — in both directions")
    axis.legend(loc="upper left", fontsize=10)
    save(fig, "sgc-homophily")


def main() -> None:
    figure_minibatch_edge_loss()
    figure_computational_graph_growth()
    figure_cluster_partition_loss()
    figure_sgc_homophily()


if __name__ == "__main__":
    main()
