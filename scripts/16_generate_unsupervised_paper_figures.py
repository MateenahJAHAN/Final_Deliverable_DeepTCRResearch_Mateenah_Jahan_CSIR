#!/usr/bin/env python3
"""
Script 16: Generate paper-ready unsupervised figures (Supplementary)
===================================================================

Creates publication-style PNG+PDF figures in figures/paper_final/ from the
outputs of scripts 14 and 15.

Outputs
-------
- figures/paper_final/figureS10_unsupervised_patient_clusters.png/.pdf
- figures/paper_final/figureS11_unsupervised_sequence_space.png/.pdf
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _save(fig, out_png: str) -> None:
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_png.lower().endswith(".png"):
        fig.savefig(out_png[:-4] + ".pdf", bbox_inches="tight")


def generate_patient_cluster_figure(project_root: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    results_dir = os.path.join(project_root, "results", "unsupervised")
    out_dir = os.path.join(project_root, "figures", "paper_final")

    emb_path = os.path.join(results_dir, "patient_embeddings.csv")
    clus_path = os.path.join(results_dir, "patient_clusters.csv")

    emb = pd.read_csv(emb_path, index_col=0)
    clus = pd.read_csv(clus_path, index_col=0)
    df = clus.join(emb, how="inner")

    X = df.to_numpy()
    xy = PCA(n_components=2, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    clusters = df["cluster"].astype(int).to_numpy()
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=clusters, cmap="tab10", s=90, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.set_title("Figure S10: Unsupervised patient clustering\n(CDR3 + V/J, no labels used for clustering)", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(*sc.legend_elements(), title="Cluster", loc="best")

    # Annotate patient IDs lightly (optional; helps interpret 34 points)
    for i, pid in enumerate(df.index.tolist()):
        ax.annotate(str(pid), (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figureS10_unsupervised_patient_clusters.png"))
    plt.close(fig)


def generate_sequence_space_figure(project_root: str) -> None:
    import matplotlib.pyplot as plt

    results_dir = os.path.join(project_root, "results", "unsupervised")
    out_dir = os.path.join(project_root, "figures", "paper_final")

    path = os.path.join(results_dir, "sequence_space_2d.csv")
    df = pd.read_csv(path)

    # Use attention_weight if available; else fallback color.
    has_att = df["attention_weight"].notna().any() if "attention_weight" in df.columns else False
    att = df["attention_weight"].fillna(0.0).to_numpy() if has_att else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: all points colored by attention (proxy for "predictive sequences")
    ax = axes[0]
    if has_att:
        sc = ax.scatter(df["x"], df["y"], c=np.log10(att + 1e-12), cmap="viridis", s=4, alpha=0.25)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(attention_weight)")
        ax.set_title("A. Unsupervised sequence space\ncolored by attention (proxy importance)", fontweight="bold")
    else:
        ax.scatter(df["x"], df["y"], s=4, alpha=0.25, color="#2C3E50")
        ax.set_title("A. Unsupervised sequence space", fontweight="bold")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.2)

    # Panel B: top vs bottom attention fraction
    ax = axes[1]
    if has_att:
        top_frac = 0.10
        q_hi = np.quantile(att, 1.0 - top_frac)
        q_lo = np.quantile(att, top_frac)
        is_top = att >= q_hi
        is_bottom = att <= q_lo

        ax.scatter(df.loc[~(is_top | is_bottom), "x"], df.loc[~(is_top | is_bottom), "y"], s=3, alpha=0.12, color="#95A5A6", label="middle 80%")
        ax.scatter(df.loc[is_bottom, "x"], df.loc[is_bottom, "y"], s=8, alpha=0.55, color="#E74C3C", label="bottom 10%")
        ax.scatter(df.loc[is_top, "x"], df.loc[is_top, "y"], s=8, alpha=0.70, color="#27AE60", label="top 10%")
        ax.set_title("B. Top vs bottom 10% sequences\n(by attention proxy)", fontweight="bold")
        ax.legend(loc="best", frameon=True)
    else:
        ax.scatter(df["x"], df["y"], s=4, alpha=0.25, color="#2C3E50")
        ax.set_title("B. Top/bottom overlay unavailable", fontweight="bold")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.2)

    fig.suptitle("Figure S11: Unsupervised sequence-space visualization (Sidhom-style adapted)\n(CDR3 + V/J; UMAP on unsupervised latent embedding)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "figureS11_unsupervised_sequence_space.png"))
    plt.close(fig)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    _ensure_dirs(os.path.join(project_root, "figures", "paper_final"))

    print("=" * 80)
    print("GENERATING UNSUPERVISED PAPER FIGURES (SCRIPT 16)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    generate_patient_cluster_figure(project_root)
    generate_sequence_space_figure(project_root)

    print("Saved figures to figures/paper_final/")


if __name__ == "__main__":
    main()

