#!/usr/bin/env python3
"""
Script 18: Unsupervised validation (cluster quality + stability)
===============================================================

Unsupervised outputs (UMAP/clustering) should be *validated* differently than
supervised models:

- Internal validity: silhouette, Davies-Bouldin, Calinski-Harabasz
- Stability: how consistent clusters are under resampling / different seeds
- Optional post-hoc label association (NOT training): ARI/NMI + permutation test

This script operates on patient-level embeddings produced by Script 14
(`results/unsupervised/patient_embeddings.csv`) and produces:

Outputs
-------
results/unsupervised/cluster_metrics_by_k.csv
results/unsupervised/cluster_stability_by_k.csv
results/unsupervised/unsupervised_validation_report.txt
figures/paper_final/figureS15_unsupervised_validation.png/.pdf
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: str
    results_unsup: str
    figures_paper: str
    data_processed: str


def _paths() -> Paths:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return Paths(
        project_root=project_root,
        results_unsup=os.path.join(project_root, "results", "unsupervised"),
        figures_paper=os.path.join(project_root, "figures", "paper_final"),
        data_processed=os.path.join(project_root, "data_processed"),
    )


def _load_patient_embeddings(results_unsup: str) -> pd.DataFrame:
    path = os.path.join(results_unsup, "patient_embeddings.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def _load_patient_labels(data_processed: str) -> Optional[pd.Series]:
    path = os.path.join(data_processed, "deeptcr_trb_ready.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["patient_id", "response_binary"])
    y = df.groupby("patient_id")["response_binary"].first()
    y.index = y.index.astype(str)
    return y


def _cluster_labels(X: np.ndarray, *, k: int, seed: int, algo: str) -> np.ndarray:
    if algo == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(X)
    if algo == "gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(n_components=k, random_state=seed).fit_predict(X)
    raise ValueError(f"Unknown algo: {algo}")


def main() -> None:
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    p = _paths()
    os.makedirs(p.results_unsup, exist_ok=True)
    os.makedirs(p.figures_paper, exist_ok=True)

    emb = _load_patient_embeddings(p.results_unsup)
    X = emb.to_numpy().astype(np.float32)
    patient_ids = emb.index.to_numpy()

    y = _load_patient_labels(p.data_processed)
    y_aligned = y.reindex(patient_ids).astype(float) if y is not None else None

    # Evaluate a small range of k values (bounded by n_patients)
    n_patients = X.shape[0]
    k_values = [k for k in range(2, min(7, n_patients))]  # 2..6 for n>=7

    algo = "kmeans"
    seeds = list(range(10))  # stability runs
    subsample_frac = 0.80
    n_subsamples = 50
    rng = np.random.default_rng(13)

    metrics_rows = []
    stability_rows = []

    for k in k_values:
        # One representative clustering for internal metrics
        c = _cluster_labels(X, k=k, seed=13, algo=algo)
        sil = silhouette_score(X, c)
        db = davies_bouldin_score(X, c)
        ch = calinski_harabasz_score(X, c)

        row = {
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        }

        # Optional post-hoc association with labels (not training)
        if y_aligned is not None and y_aligned.notna().all():
            yy = y_aligned.astype(int).to_numpy()
            row["ARI_vs_response"] = adjusted_rand_score(yy, c)
            row["NMI_vs_response"] = normalized_mutual_info_score(yy, c)
        metrics_rows.append(row)

        # Stability: agreement across different seeds + subsamples
        # Seed stability (full data)
        seed_labels = [_cluster_labels(X, k=k, seed=s, algo=algo) for s in seeds]
        seed_aris = []
        for i in range(len(seed_labels)):
            for j in range(i + 1, len(seed_labels)):
                seed_aris.append(adjusted_rand_score(seed_labels[i], seed_labels[j]))
        seed_ari_mean = float(np.mean(seed_aris)) if seed_aris else float("nan")

        # Subsample stability
        sub_aris = []
        for _ in range(n_subsamples):
            idx = rng.choice(n_patients, size=max(2, int(n_patients * subsample_frac)), replace=False)
            Xs = X[idx]
            c1 = _cluster_labels(Xs, k=k, seed=int(rng.integers(1_000_000)), algo=algo)
            c2 = _cluster_labels(Xs, k=k, seed=int(rng.integers(1_000_000)), algo=algo)
            sub_aris.append(adjusted_rand_score(c1, c2))
        sub_ari_mean = float(np.mean(sub_aris)) if sub_aris else float("nan")
        sub_ari_std = float(np.std(sub_aris)) if sub_aris else float("nan")

        stability_rows.append(
            {
                "k": k,
                "seed_ARI_mean": seed_ari_mean,
                "subsample_ARI_mean": sub_ari_mean,
                "subsample_ARI_std": sub_ari_std,
                "n_subsamples": n_subsamples,
                "subsample_frac": subsample_frac,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("k")
    stability_df = pd.DataFrame(stability_rows).sort_values("k")

    metrics_csv = os.path.join(p.results_unsup, "cluster_metrics_by_k.csv")
    stability_csv = os.path.join(p.results_unsup, "cluster_stability_by_k.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    # Permutation test for k=2 association (optional)
    perm_p = None
    if y_aligned is not None and y_aligned.notna().all() and 2 in k_values:
        from scipy.stats import fisher_exact

        c2 = _cluster_labels(X, k=2, seed=13, algo=algo)
        yy = y_aligned.astype(int).to_numpy()

        def fisher_p(y_true: np.ndarray, c_lab: np.ndarray) -> float:
            # 2x2 table: response (0/1) x cluster (0/1)
            a = int(((y_true == 1) & (c_lab == 1)).sum())
            b = int(((y_true == 1) & (c_lab == 0)).sum())
            c = int(((y_true == 0) & (c_lab == 1)).sum())
            d = int(((y_true == 0) & (c_lab == 0)).sum())
            _, pval = fisher_exact([[a, b], [c, d]])
            return float(pval)

        p_obs = fisher_p(yy, c2)
        n_perm = 5000
        perm = []
        for _ in range(n_perm):
            yperm = rng.permutation(yy)
            perm.append(fisher_p(yperm, c2))
        perm = np.array(perm)
        perm_p = float((np.sum(perm <= p_obs) + 1) / (n_perm + 1))
    else:
        p_obs = None

    # Write report
    report_path = os.path.join(p.results_unsup, "unsupervised_validation_report.txt")
    lines = []
    lines.append("Unsupervised validation report")
    lines.append("=" * 80)
    lines.append(f"Patients: {n_patients}")
    lines.append(f"Embedding dims: {X.shape[1]}")
    lines.append(f"Algorithm: {algo}")
    lines.append("")
    lines.append("Internal cluster quality by k (higher silhouette/CH better; lower DB better):")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("Cluster stability by k (ARI agreement; higher is better):")
    lines.append(stability_df.to_string(index=False))
    if perm_p is not None:
        lines.append("")
        lines.append("Post-hoc association with response labels (k=2; Fisher exact):")
        lines.append(f"Observed p-value: {p_obs:.4g}")
        lines.append(f"Permutation p-value (n=5000): {perm_p:.4g}")
        lines.append("Note: this does NOT train on labels; it only tests association after clustering.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Figure S15: metrics plots
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#2C3E50")
    axes[0].set_title("Silhouette (↑ better)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(metrics_df["k"], metrics_df["davies_bouldin"], marker="o", color="#E74C3C")
    axes[1].set_title("Davies-Bouldin (↓ better)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("score")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(stability_df["k"], stability_df["subsample_ARI_mean"], marker="o", color="#27AE60", label="subsample ARI")
    axes[2].plot(stability_df["k"], stability_df["seed_ARI_mean"], marker="o", color="#3498DB", label="seed ARI")
    axes[2].set_title("Stability (ARI ↑ better)")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("ARI")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend(loc="best")

    fig.suptitle("Figure S15: Unsupervised clustering validation (quality + stability)", fontweight="bold", y=1.02)
    fig.tight_layout()

    out_png = os.path.join(p.figures_paper, "figureS15_unsupervised_validation.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  - {metrics_csv}")
    print(f"  - {stability_csv}")
    print(f"  - {report_path}")
    print(f"  - {out_png} (+.pdf)")


if __name__ == "__main__":
    main()

