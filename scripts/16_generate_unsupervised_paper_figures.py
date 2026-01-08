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
    # Color by response (Responder=green, Non-responder=orange) for clarity.
    # Response labels are available in data_processed/deeptcr_trb_ready.csv.
    resp_path = os.path.join(project_root, "data_processed", "deeptcr_trb_ready.csv")
    resp_df = pd.read_csv(resp_path)
    resp = resp_df.groupby("patient_id")["response_binary"].first().astype(int)
    resp = resp.reindex(df.index.astype(str))
    is_r = resp.fillna(0).astype(int).to_numpy() == 1
    is_nr = ~is_r

    ax.scatter(xy[is_nr, 0], xy[is_nr, 1], s=95, alpha=0.85, color="#F39C12", edgecolor="black", linewidth=0.4, label="Non-responder")
    ax.scatter(xy[is_r, 0], xy[is_r, 1], s=95, alpha=0.85, color="#27AE60", edgecolor="black", linewidth=0.4, label="Responder")
    ax.set_title("Figure S10: Unsupervised patient embedding (PCA)\ncolored by response label", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", title="Response")

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

    has_att = df["attention_weight"].notna().any() if "attention_weight" in df.columns else False
    att = df["attention_weight"].fillna(0.0).to_numpy() if has_att else None
    has_resp = "response_binary" in df.columns and df["response_binary"].notna().any()

    # Colors requested by user
    COLOR_R = "#27AE60"   # green
    COLOR_NR = "#F39C12"  # orange

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: all points colored by response (green/orange) when available
    ax = axes[0]
    if has_resp:
        r_mask = df["response_binary"].astype(int) == 1
        nr_mask = ~r_mask
        ax.scatter(df.loc[nr_mask, "x"], df.loc[nr_mask, "y"], s=4, alpha=0.18, color=COLOR_NR, label="Non-responder")
        ax.scatter(df.loc[r_mask, "x"], df.loc[r_mask, "y"], s=4, alpha=0.18, color=COLOR_R, label="Responder")
        ax.set_title("A. Unsupervised sequence space\ncolored by response label", fontweight="bold")
        ax.legend(loc="best", frameon=True)
    else:
        ax.scatter(df["x"], df["y"], s=4, alpha=0.25, color="#2C3E50")
        ax.set_title("A. Unsupervised sequence space", fontweight="bold")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.2)

    # Panel B: highlight top-attention sequences as an overlay (keeps response colors if present)
    ax = axes[1]
    if has_att:
        top_frac = 0.10
        q_hi = np.quantile(att, 1.0 - top_frac)
        is_top = att >= q_hi

        # Base layer
        if has_resp:
            r_mask = df["response_binary"].astype(int) == 1
            nr_mask = ~r_mask
            ax.scatter(df.loc[nr_mask, "x"], df.loc[nr_mask, "y"], s=3, alpha=0.10, color=COLOR_NR, label="Non-responder")
            ax.scatter(df.loc[r_mask, "x"], df.loc[r_mask, "y"], s=3, alpha=0.10, color=COLOR_R, label="Responder")
        else:
            ax.scatter(df.loc[~is_top, "x"], df.loc[~is_top, "y"], s=3, alpha=0.12, color="#95A5A6", label="other")

        # Top attention overlay as black outlined points
        ax.scatter(
            df.loc[is_top, "x"],
            df.loc[is_top, "y"],
            s=14,
            alpha=0.85,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label="Top 10% attention (overlay)",
        )

        ax.set_title("B. Response colors + top-attention overlay\n(attention proxy from supervised MIL)", fontweight="bold")
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

