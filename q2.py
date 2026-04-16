from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


# ============================================================================
# SETTINGS
# ============================================================================

# K used for the initial exploratory spaghetti plot in 2a
EXPLORE_K = 4

# Range of K values to evaluate in 2b
CANDIDATE_K = [2, 3, 4, 5]

# Final chosen K after reviewing policy outcomes in 2c
CHOSEN_K = 3

# Maximum sessions in the funded pathway
SESSION_LIMIT = 12

# One colour per cluster (supports up to 5)
PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]

# All Q2 outputs go here
OUT_DIR = Path("output/q2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def read_unlabeled_scores() -> pd.DataFrame:
    # scores from the unlabeled pipeline output (JSON)
    with open("output/q1/scored_notes.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        cid = str(rec["client_id"]).strip()
        scores = rec.get("estimated_trajectory_vector", [])
        for idx, val in enumerate(scores, start=1):
            rows.append({"client_id": cid, "session": idx, "score": int(val)})

    return pd.DataFrame(rows, columns=["client_id", "session", "score"])


def read_labeled_scores() -> pd.DataFrame:
    # extract estimated vectors from the labeled evaluation JSON
    with open("output/q1/evaluated_labeled_results.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        cid = str(rec["client_id"]).strip()
        scores = rec.get("estimated_trajectory_vector", [])
        for idx, val in enumerate(scores, start=1):
            rows.append({"client_id": cid, "session": idx, "score": int(val)})

    return pd.DataFrame(rows, columns=["client_id", "session", "score"])


def pivot_to_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # reshape long-format scores to clients x sessions matrix
    wide = df.pivot_table(index="client_id", columns="session", values="score", aggfunc="first")
    wide = wide.reindex(columns=range(1, SESSION_LIMIT))
    return wide.values.astype(float), list(wide.index)


def build_trajectories(score_matrix: np.ndarray) -> np.ndarray:
    # cumulative sum of scores across sessions, starting at 0
    n_clients = score_matrix.shape[0]
    cum = np.zeros((n_clients, SESSION_LIMIT), dtype=float)
    cum[:, 1:] = np.cumsum(score_matrix, axis=1)
    return cum


def load_all_data() -> Tuple[np.ndarray, List[str]]:
    # merge labeled and unlabeled clients into one trajectory matrix
    unlabeled = read_unlabeled_scores()
    labeled = read_labeled_scores()

    combined = pd.concat([unlabeled, labeled], ignore_index=True)
    combined = combined.drop_duplicates(subset=["client_id", "session"])

    score_matrix, client_ids = pivot_to_matrix(combined)
    trajectories = build_trajectories(score_matrix)
    return trajectories, client_ids


# ============================================================================
# 2A — CLUSTERING
# ============================================================================

def fit_kmeans(data: np.ndarray, k: int) -> np.ndarray:
    # standardize then fit K-means, return cluster labels
    scaled = StandardScaler().fit_transform(data)
    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    return model.fit_predict(scaled)


def make_spaghetti_plot(
    trajectories: np.ndarray,
    assignments: np.ndarray,
    k: int,
    save_to: Path = OUT_DIR / "spaghetti_plots.png",
) -> None:
    # faint individual lines + bold cluster mean, one panel per cluster
    x_axis = np.arange(1, SESSION_LIMIT + 1)

    ncols = 2
    nrows = (k + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for ax in axes[k:]:
        ax.set_visible(False)

    for c in range(k):
        ax = axes[c]
        colour = PALETTE[c]
        in_cluster = trajectories[assignments == c]

        for traj in in_cluster:
            ax.plot(x_axis, traj, color=colour, alpha=0.25, linewidth=0.8)

        ax.plot(x_axis, in_cluster.mean(axis=0), colour, linewidth=2.5,
                label=f"Cluster {c + 1} mean")

        ax.set_title(f"Cluster {c + 1}  (n={in_cluster.shape[0]})", fontsize=11)
        ax.set_xlabel("Session", fontsize=9)
        if c % 2 == 0:
            ax.set_ylabel("Cumulative Progress Score", fontsize=9)
        ax.set_xticks(x_axis)
        ax.set_xticklabels([str(s) for s in x_axis], fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cumulative Progress Trajectories by Cluster (K={k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")


def run_2a(trajectories: np.ndarray, k: int) -> None:
    assignments = fit_kmeans(trajectories, k)
    make_spaghetti_plot(trajectories, assignments, k)


# ============================================================================
# 2B — OPTIMAL REASSESSMENT POLICY (NEWSVENDOR)
# ============================================================================

def get_stopping_points(trajectories: np.ndarray) -> np.ndarray:
    # t* = earliest session where cumulative progress >= 90% of total
    stopping = np.full(trajectories.shape[0], SESSION_LIMIT, dtype=int)
    for i, traj in enumerate(trajectories):
        total = traj[-1]
        if total == 0:
            continue
        cutoff = 0.9 * total
        for t in range(SESSION_LIMIT):
            if traj[t] >= cutoff:
                stopping[i] = t + 1
                break
    return stopping


def compute_expected_savings(
    stopping_points: np.ndarray,
    assignments: np.ndarray,
    k: int,
) -> np.ndarray:
    # E[savings](Q) = F_c(Q) * (SESSION_LIMIT - Q) for each cluster and Q
    Q_range = np.arange(1, SESSION_LIMIT + 1)
    savings = np.zeros((k, SESSION_LIMIT))
    for c in range(k):
        cluster_stops = stopping_points[assignments == c]
        for i, Q in enumerate(Q_range):
            frac_done = np.mean(cluster_stops <= Q)
            savings[c, i] = frac_done * (SESSION_LIMIT - Q)
    return savings


def get_optimal_Q(savings_matrix: np.ndarray) -> np.ndarray:
    # Q* = session that maximises expected savings per cluster
    return np.argmax(savings_matrix, axis=1) + 1


def plot_all_k_savings(
    savings_by_k: Dict[int, np.ndarray],
    q_stars_by_k: Dict[int, np.ndarray],
    save_to: Path = OUT_DIR / "all_savings_curves.png",
) -> None:
    # savings curves for every K evaluated, one panel per K
    k_list = sorted(savings_by_k.keys())
    ncols = 2
    nrows = (len(k_list) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=True)
    axes = np.array(axes).flatten()
    for ax in axes[len(k_list):]:
        ax.set_visible(False)

    Q_range = np.arange(1, SESSION_LIMIT + 1)
    for ax, k in zip(axes, k_list):
        curves = savings_by_k[k]
        q_stars = q_stars_by_k[k]
        for c in range(k):
            ax.plot(Q_range, curves[c], color=PALETTE[c], linewidth=2,
                    label=f"Cluster {c + 1} (Q*={q_stars[c]})")
            ax.axvline(q_stars[c], color=PALETTE[c], linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"K = {k}", fontsize=12)
        ax.set_xlabel("Reassessment session Q", fontsize=10)
        ax.set_xticks(Q_range)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for row in range(nrows):
        axes[row * ncols].set_ylabel("E[sessions saved per child]", fontsize=10)
    fig.suptitle("Newsvendor E[Savings](Q) — All K Values", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")


def run_2b(trajectories: np.ndarray) -> None:
    print("\n--- 2b: Optimal Reassessment Policy ---")

    stopping_points = get_stopping_points(trajectories)
    savings_by_k: Dict[int, np.ndarray] = {}
    q_stars_by_k: Dict[int, np.ndarray] = {}

    for k in CANDIDATE_K:
        assignments = fit_kmeans(trajectories, k)
        savings = compute_expected_savings(stopping_points, assignments, k)
        q_stars = get_optimal_Q(savings)
        savings_by_k[k] = savings
        q_stars_by_k[k] = q_stars
        print_policy_table(stopping_points, assignments, savings, q_stars, k)
        make_spaghetti_plot(
            trajectories, assignments, k,
            save_to=OUT_DIR / f"spaghetti_plots_k={k}.png",
        )

    plot_all_k_savings(savings_by_k, q_stars_by_k)


# ============================================================================
# 2D — FINAL PLOTS FOR CHOSEN K
# ============================================================================

def plot_stopping_histograms(
    stopping_points: np.ndarray,
    assignments: np.ndarray,
    k: int,
    save_to: Path = OUT_DIR / "t_star_distributions.png",
) -> None:
    # histogram of t* values per cluster
    bins = np.arange(0.5, SESSION_LIMIT + 1.5, 1)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=False, sharex=True)
    if k == 1:
        axes = [axes]

    for c in range(k):
        ax = axes[c]
        cluster_stops = stopping_points[assignments == c]
        ax.hist(cluster_stops, bins=bins, color=PALETTE[c], edgecolor="white", alpha=0.85)
        ax.set_title(f"Cluster {c + 1}  (n={cluster_stops.size})", fontsize=11)
        ax.set_xlabel("t* (session)", fontsize=9)
        ax.set_xticks(range(1, SESSION_LIMIT + 1))
        mean_val = np.mean(cluster_stops)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.1f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        if c == 0:
            ax.set_ylabel("Count", fontsize=9)

    fig.suptitle("Distribution of Stopping Points (t*) by Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")


def plot_savings_by_Q(
    savings: np.ndarray,
    q_stars: np.ndarray,
    k: int,
    save_to: Path = OUT_DIR / "savings_curves.png",
) -> None:
    # E[savings] vs Q for all clusters, dashed lines at Q*
    Q_range = np.arange(1, SESSION_LIMIT + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for c in range(k):
        ax.plot(Q_range, savings[c], color=PALETTE[c], linewidth=2,
                label=f"Cluster {c + 1} (Q*={q_stars[c]})")
        ax.axvline(q_stars[c], color=PALETTE[c], linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Reassessment session Q", fontsize=11)
    ax.set_ylabel("E[sessions saved per child]", fontsize=11)
    ax.set_title("Expected Sessions Saved vs Reassessment Timing", fontsize=13)
    ax.set_xticks(Q_range)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")


def plot_optimized_vs_baseline(
    stopping_points: np.ndarray,
    assignments: np.ndarray,
    savings: np.ndarray,
    q_stars: np.ndarray,
    k: int,
    save_to: Path = OUT_DIR / "grouped_bar_savings.png",
) -> None:
    # grouped bar: optimized Q* savings vs naive mean-t* baseline per cluster
    optimized_vals = np.array([savings[c, q_stars[c] - 1] for c in range(k)])
    baseline_vals = np.zeros(k)
    for c in range(k):
        cluster_stops = stopping_points[assignments == c]
        naive_Q = int(np.clip(round(np.mean(cluster_stops)), 1, SESSION_LIMIT))
        baseline_vals[c] = savings[c, naive_Q - 1]

    x = np.arange(k)
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(6 + k, 5))

    bars_opt = ax.bar(x - bar_w / 2, optimized_vals, bar_w,
                      color=PALETTE[:k], edgecolor="white", label="Optimized Q*")
    bars_base = ax.bar(x + bar_w / 2, baseline_vals, bar_w,
                       color=PALETTE[:k], edgecolor="black", alpha=0.5, label="Baseline mean(t*)")

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("E[sessions saved per child]", fontsize=11)
    ax.set_title("Optimized Policy vs Mean-t* Baseline", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c + 1}" for c in range(k)], fontsize=10)

    legend_patches = [
        Patch(facecolor="gray", edgecolor="white", label="Optimized Q*"),
        Patch(facecolor="gray", alpha=0.5, edgecolor="white", label="Baseline mean(t*)"),
    ]
    ax.legend(handles=legend_patches, fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars_opt:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")


def run_2d(trajectories: np.ndarray, client_ids: List[str], k: int) -> None:
    print(f"\n--- 2d: Final Plots (K={k}) ---")

    assignments = fit_kmeans(trajectories, k)
    stopping_points = get_stopping_points(trajectories)
    savings = compute_expected_savings(stopping_points, assignments, k)
    q_stars = get_optimal_Q(savings)

    plot_stopping_histograms(stopping_points, assignments, k)
    plot_savings_by_Q(savings, q_stars, k)
    plot_optimized_vs_baseline(stopping_points, assignments, savings, q_stars, k)

    # total savings comparison: optimized vs baseline across all clients
    total_opt = 0.0
    total_base = 0.0
    for c in range(k):
        n_c = int((assignments == c).sum())
        naive_Q = int(np.clip(round(np.mean(stopping_points[assignments == c])), 1, SESSION_LIMIT))
        total_opt += savings[c, q_stars[c] - 1] * n_c
        total_base += savings[c, naive_Q - 1] * n_c

    print(f"\nTotal sessions saved (baseline):  {total_base:.1f}")
    print(f"Total sessions saved (optimized): {total_opt:.1f}")
    print(f"Delta: {total_opt - total_base:.1f} sessions\n")

    # save per-client cluster and t* for Q3
    pd.DataFrame({
        "client_id": client_ids,
        "cluster": assignments + 1,
        "t_star": stopping_points,
    }).to_csv(OUT_DIR / "t_star_assignments.csv", index=False)
    print(f"Saved: {OUT_DIR / 't_star_assignments.csv'}")


# ============================================================================
# HELPER — SUMMARY TABLE
# ============================================================================

def print_policy_table(
    stopping_points: np.ndarray,
    assignments: np.ndarray,
    savings: np.ndarray,
    q_stars: np.ndarray,
    k: int,
) -> None:
    # print cluster-level policy summary: size, Q*, expected savings, % saved
    header = f"{'Cluster':>8}  {'Size':>6}  {'Q*':>4}  {'E[saved/child]':>16}  {'% sessions saved':>18}"
    divider = "=" * len(header)
    print(f"\n{divider}\n{header}\n{divider}")

    total_n = 0
    total_weighted_savings = 0.0

    for c in range(k):
        n_c = int((assignments == c).sum())
        q_c = int(q_stars[c])
        exp_saved = savings[c, q_c - 1]
        pct = exp_saved / SESSION_LIMIT * 100
        print(f"  {c + 1:>6}  {n_c:>6}  {q_c:>4}  {exp_saved:>16.3f}  {pct:>17.1f}%")
        total_n += n_c
        total_weighted_savings += n_c * exp_saved

    overall_pct = total_weighted_savings / (total_n * SESSION_LIMIT) * 100 if total_n > 0 else 0.0
    print("-" * len(header))
    print(f"  {'Total':>6}  {'—':>6}  {'—':>4}  {'—':>16}  {overall_pct:>17.1f}%")
    print(divider)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    trajectories, client_ids = load_all_data()

    # 2a: exploratory spaghetti plot
    run_2a(trajectories, EXPLORE_K)

    # 2b: evaluate policy across all candidate K values
    run_2b(trajectories)

    # 2d: final plots and savings comparison for chosen K
    run_2d(trajectories, client_ids, CHOSEN_K)