#!/usr/bin/env python3
"""Generate the computed figures used in the Lecture 5 web note.

The LightGCN experiment trains the embeddings *through* the diffusion, which is
what the model actually does — diffusing embeddings that were trained without
diffusion is a different (and much weaker) thing. Run from the repository root:

    python3 scripts/figures/lecture_05_figures.py

The LightGCN sweep takes a few minutes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import CORAL, INK, MUTED, TEAL, save, use_style

use_style()


# --------------------------------------------------------------------------
# A synthetic user-item graph with latent taste groups
# --------------------------------------------------------------------------
def make_interactions(
    users: int = 400, items: int = 300, groups: int = 6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Interactions driven by hidden taste groups, with heavy-tailed popularity.

    Collaborative filtering can only work if users who agreed in the past agree
    in the future, so the ground truth gives every user a taste group. Item
    popularity and user activity are Pareto-distributed, as they are in a real
    catalogue.
    """
    rng = np.random.default_rng(seed)
    user_group = rng.integers(0, groups, size=users)
    item_group = rng.integers(0, groups, size=items)

    popularity = rng.pareto(1.4, size=items) + 1.0
    popularity /= popularity.mean()
    activity = rng.pareto(1.4, size=users) + 1.0
    activity /= activity.mean()

    base = np.where(user_group[:, None] == item_group[None, :], 0.16, 0.006)
    probability = base * popularity[None, :] ** 0.9 * activity[:, None] ** 0.6
    observed = rng.random((users, items)) < np.clip(probability, 0.0, 0.95)

    # Hold out a quarter of each user's interactions as the future to predict.
    train = np.zeros_like(observed)
    test = np.zeros_like(observed)
    for user in range(users):
        positives = np.flatnonzero(observed[user])
        rng.shuffle(positives)
        if len(positives) < 4:
            continue
        split = int(0.75 * len(positives))
        train[user, positives[:split]] = True
        test[user, positives[split:]] = True
    return train, test


def recall_at_k(scores: np.ndarray, train: np.ndarray, test: np.ndarray, k: int = 10) -> float:
    """Mean over users of |P_u ∩ R_u| / |P_u|, excluding already-seen items."""
    masked = np.where(train, -np.inf, scores)
    ranked = np.argsort(-masked, axis=1)[:, :k]
    recalls = []
    for user in range(scores.shape[0]):
        positives = set(np.flatnonzero(test[user]).tolist())
        if not positives:
            continue
        recalls.append(len(positives & set(ranked[user].tolist())) / len(positives))
    return float(np.mean(recalls))


def propagation_operator(train: np.ndarray) -> np.ndarray:
    """LightGCN's normalized bipartite adjacency, without self-loops."""
    users, items = train.shape
    adjacency = np.zeros((users + items, users + items))
    adjacency[:users, users:] = train
    adjacency[users:, :users] = train.T
    degree = adjacency.sum(axis=1)
    inverse = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    return inverse[:, None] * adjacency * inverse[None, :]


def train_lightgcn(
    train: np.ndarray,
    test: np.ndarray,
    layers: int,
    dimension: int = 32,
    steps: int = 1200,
    learning_rate: float = 0.5,
    negatives: int = 4,
    weight_decay: float = 1e-4,
    k: int = 10,
    seed: int = 1,
) -> float:
    """LightGCN under the BPR loss, differentiating through the diffusion.

    Forward:  E_final = mean_{i<=K} S^i E
    Backward: dL/dE = mean_{i<=K} S^i (dL/dE_final), since S is symmetric.
    `layers = 0` is the shallow, undiffused baseline.
    """
    rng = np.random.default_rng(seed)
    users, items = train.shape
    operator = propagation_operator(train)
    embedding = rng.normal(scale=0.1, size=(users + items, dimension))
    positive_pairs = np.argwhere(train)

    for _ in range(steps):
        state = embedding.copy()
        total = embedding.copy()
        for _ in range(layers):
            state = operator @ state
            total = total + state
        final = total / (layers + 1)

        chosen = rng.choice(len(positive_pairs), size=min(4096, len(positive_pairs)), replace=False)
        u_index = positive_pairs[chosen, 0]
        p_index = positive_pairs[chosen, 1] + users
        sampled = rng.integers(0, items, size=(len(chosen), negatives))
        valid = ~train[u_index[:, None], sampled]
        n_index = sampled + users

        z_user = final[u_index]
        z_pos = final[p_index]
        z_neg = final[n_index]
        difference = np.einsum("ij,ij->i", z_user, z_pos)[:, None] - np.einsum("ijk,ik->ij", z_neg, z_user)
        weight = (1.0 / (1.0 + np.exp(difference))) * valid

        gradient = np.zeros_like(final)
        np.add.at(gradient, u_index, -(weight[:, :, None] * (z_pos[:, None, :] - z_neg)).sum(1) / negatives)
        np.add.at(gradient, p_index, -(weight.sum(1) / negatives)[:, None] * z_user)
        np.add.at(gradient, n_index.ravel(),
                  (weight / negatives).ravel()[:, None] * np.repeat(z_user, negatives, axis=0))
        gradient /= len(chosen)

        through = gradient.copy()
        accumulated = gradient.copy()
        for _ in range(layers):
            accumulated = operator @ accumulated
            through = through + accumulated
        embedding -= learning_rate * (through / (layers + 1) + weight_decay * embedding)

    state = embedding.copy()
    total = embedding.copy()
    for _ in range(layers):
        state = operator @ state
        total = total + state
    final = total / (layers + 1)
    return recall_at_k(final[:users] @ final[users:].T, train, test, k)


# --------------------------------------------------------------------------
# Figure 1 — the two-stage architecture of a deployed recommender
# --------------------------------------------------------------------------
def figure_two_stage() -> None:
    fig, axis = plt.subplots(figsize=(10.4, 3.2))

    boxes = [
        (0.5, "All items\n$10^{6}$–$10^{9}$", "#eef3f4", MUTED),
        (3.1, "Candidate generation\ncheap, approximate", "#e2eff0", TEAL),
        (5.7, "Candidates\n$\\sim10^{3}$", "#eef3f4", MUTED),
        (8.3, "Ranking\nslow, accurate", "#f9e9e2", CORAL),
        (10.9, "Top-$k$\n$k\\approx10$–$100$", "#eef3f4", MUTED),
    ]
    for x, label, face, edge in boxes:
        axis.add_patch(
            FancyBboxPatch((x, 0.6), 2.0, 1.1, boxstyle="round,pad=0.08",
                           facecolor=face, edgecolor=edge, linewidth=1.6)
        )
        axis.annotate(label, xy=(x + 1.0, 1.15), ha="center", va="center", fontsize=10.5, color=INK)

    for x in (2.5, 5.1, 7.7, 10.3):
        axis.add_patch(
            FancyArrowPatch((x, 1.15), (x + 0.6, 1.15), arrowstyle="-|>",
                            mutation_scale=17, linewidth=1.6, color=MUTED)
        )

    axis.annotate(
        r"scoring every pair costs $|\mathcal{U}|\times|\mathcal{V}|$ — the reason for two stages",
        xy=(6.45, 0.15), ha="center", fontsize=10.5, color=MUTED,
    )
    axis.set_xlim(0, 13.2)
    axis.set_ylim(-0.1, 2.1)
    axis.set_axis_off()
    save(fig, "two-stage-recommender")


# --------------------------------------------------------------------------
# Figure 2 — what the diffusion is worth
# --------------------------------------------------------------------------
def figure_lightgcn_depth() -> None:
    train, test = make_interactions()
    depths = [0, 1, 2, 3, 4, 6, 8, 10, 12]
    recalls = [100 * train_lightgcn(train, test, layers) for layers in depths]

    fig, axis = plt.subplots(figsize=(7.8, 4.3))
    axis.plot(depths, recalls, "o-", color=TEAL, linewidth=2.4, markersize=6.5)
    axis.plot([0], [recalls[0]], "o", color=CORAL, markersize=11, zorder=3)
    axis.annotate(
        f"$k=0$: shallow embeddings only\n({recalls[0]:.1f}%)",
        xy=(0, recalls[0]),
        xytext=(16, 4),
        textcoords="offset points",
        fontsize=10,
        color=CORAL,
    )
    axis.annotate(
        f"$k={depths[-1]}$: {recalls[-1]:.1f}%",
        xy=(depths[-1], recalls[-1]),
        xytext=(-6, -20),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=TEAL,
    )
    axis.set_xlabel("number of diffusion steps $k$")
    axis.set_ylabel("test Recall@10 (%)")
    axis.set_title("Diffusion is the whole of what LightGCN adds")
    save(fig, "lightgcn-diffusion-depth")


def main() -> None:
    figure_two_stage()
    figure_lightgcn_depth()


if __name__ == "__main__":
    main()
