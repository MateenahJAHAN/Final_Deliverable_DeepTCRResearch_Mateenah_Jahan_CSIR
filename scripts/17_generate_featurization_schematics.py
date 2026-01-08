#!/usr/bin/env python3
"""
Script 17: Generate featurization / model schematic figures (Sidhom-style)
=========================================================================

Creates simple schematic diagrams showing how sequences are featurized and used in:
  (A) Unsupervised embedding + UMAP visualization
  (B) Supervised MIL + attention + patient-level prediction

NOTE: These are conceptual diagrams to communicate the pipeline; they do not
execute the DeepTCR model itself.

Outputs (PNG + PDF)
-------------------
- figures/paper_final/figureS12_featurization_schematics.png/.pdf
- figures/paper_final/figureS13_unsupervised_pipeline.png/.pdf
"""

from __future__ import annotations

import os
from datetime import datetime


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, out_png: str) -> None:
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_png.lower().endswith(".png"):
        fig.savefig(out_png[:-4] + ".pdf", bbox_inches="tight")


def _box(ax, x, y, w, h, text, *, fc="#FFFFFF", ec="#2C3E50", lw=1.2, fontsize=10, weight="normal"):
    import matplotlib.patches as patches

    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#2C3E50")
    return rect


def _arrow(ax, x1, y1, x2, y2, *, color="#2C3E50"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.4, color=color, shrinkA=0, shrinkB=0),
    )


def main() -> None:
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dir = os.path.join(project_root, "figures", "paper_final")
    _ensure_dir(out_dir)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5))

    # -------------------------------------------------------------------------
    # Panel A: Unsupervised pipeline schematic
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.01, 0.96, "A. Unsupervised representation + UMAP (no labels used for embedding)", fontsize=12, fontweight="bold", color="#2C3E50")

    _box(ax, 0.02, 0.58, 0.20, 0.22, "Input\nCDR3β + V/J", fc="#ECF0F1", weight="bold")
    _box(ax, 0.27, 0.62, 0.22, 0.14, "Featurization\n(one-hot + V/J)\n+ k-mers", fc="#E8F6F3")
    _box(ax, 0.53, 0.62, 0.18, 0.14, "Latent space\n(128-D)", fc="#FEF5E7")
    _box(ax, 0.76, 0.62, 0.20, 0.14, "UMAP (2-D)\nvisualization", fc="#FDEDEC")

    _arrow(ax, 0.22, 0.69, 0.27, 0.69)
    _arrow(ax, 0.49, 0.69, 0.53, 0.69)
    _arrow(ax, 0.71, 0.69, 0.76, 0.69)

    _box(ax, 0.27, 0.34, 0.22, 0.14, "Pool to patient\n(mean of sequences)", fc="#E8F6F3")
    _box(ax, 0.53, 0.34, 0.18, 0.14, "Patient embedding", fc="#FEF5E7")
    _box(ax, 0.76, 0.34, 0.20, 0.14, "Cluster patients\n(k-means)", fc="#FDEDEC")

    _arrow(ax, 0.49, 0.41, 0.53, 0.41)
    _arrow(ax, 0.71, 0.41, 0.76, 0.41)
    _arrow(ax, 0.38, 0.62, 0.38, 0.48)

    ax.text(
        0.02,
        0.10,
        "Note: This is an unsupervised structure/visualization tool.\nIt does not assign 'Responder/Non-responder' without labeled outcomes.",
        fontsize=9,
        color="#2C3E50",
    )

    # -------------------------------------------------------------------------
    # Panel B: Supervised MIL schematic
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.01, 0.96, "B. Supervised DeepTCR MIL (CNN featurization + attention → patient prediction)", fontsize=12, fontweight="bold", color="#2C3E50")

    _box(ax, 0.02, 0.60, 0.20, 0.22, "Patient repertoire\n(bag of TCRs)", fc="#ECF0F1", weight="bold")
    _box(ax, 0.27, 0.64, 0.18, 0.14, "CDR3 one-hot", fc="#E8F6F3")
    _box(ax, 0.47, 0.64, 0.18, 0.14, "CNN encoder\n(featurization)", fc="#D6EAF8")
    _box(ax, 0.67, 0.64, 0.16, 0.14, "Concat V/J", fc="#E8F6F3")
    _box(ax, 0.85, 0.64, 0.13, 0.14, "Seq embed", fc="#FEF5E7")

    _arrow(ax, 0.22, 0.71, 0.27, 0.71)
    _arrow(ax, 0.45, 0.71, 0.47, 0.71)
    _arrow(ax, 0.65, 0.71, 0.67, 0.71)
    _arrow(ax, 0.83, 0.71, 0.85, 0.71)

    _box(ax, 0.30, 0.34, 0.22, 0.14, "Attention\n(learn weights)", fc="#FADBD8")
    _box(ax, 0.56, 0.34, 0.20, 0.14, "Weighted pooling\n(patient vector)", fc="#FEF5E7")
    _box(ax, 0.80, 0.34, 0.18, 0.14, "Classifier\nResponder vs NR", fc="#D5F5E3", weight="bold")

    _arrow(ax, 0.91, 0.64, 0.41, 0.48)
    _arrow(ax, 0.52, 0.41, 0.56, 0.41)
    _arrow(ax, 0.76, 0.41, 0.80, 0.41)

    ax.text(
        0.02,
        0.10,
        "The CNN featurization + attention layer identifies which clonotypes contribute most to the patient-level prediction.",
        fontsize=9,
        color="#2C3E50",
    )

    fig.suptitle("Figure S12: Featurization schematics for unsupervised and supervised analyses", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = os.path.join(out_dir, "figureS12_featurization_schematics.png")
    _save(fig, out_png)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure S13: dedicated unsupervised pipeline (input -> featurization -> embedding)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.01, 0.92, "Figure S13: Unsupervised pipeline (input → featurization → embedding → UMAP)", fontsize=12, fontweight="bold", color="#2C3E50")

    _box(ax, 0.03, 0.50, 0.18, 0.28, "Input\nCDR3β + V/J", fc="#ECF0F1", weight="bold")
    _box(ax, 0.27, 0.52, 0.20, 0.24, "Featurization\nCDR3 k-mers\n+ V/J one-hot", fc="#E8F6F3")
    _box(ax, 0.53, 0.52, 0.16, 0.24, "Embedding\n(128-D latent)", fc="#FEF5E7")
    _box(ax, 0.74, 0.52, 0.22, 0.24, "UMAP (2-D)\nsequence space\nplot", fc="#FDEDEC")

    _arrow(ax, 0.21, 0.64, 0.27, 0.64)
    _arrow(ax, 0.47, 0.64, 0.53, 0.64)
    _arrow(ax, 0.69, 0.64, 0.74, 0.64)

    # Patient-level branch
    _box(ax, 0.53, 0.12, 0.16, 0.22, "Pool to patient\n(mean)", fc="#E8F6F3")
    _box(ax, 0.74, 0.12, 0.22, 0.22, "Patient embedding\n+ clustering", fc="#FDEDEC")
    _arrow(ax, 0.61, 0.52, 0.61, 0.34)
    _arrow(ax, 0.69, 0.23, 0.74, 0.23)

    ax.text(0.03, 0.03, "Note: clustering is unsupervised; response labels are used only for coloring plots.", fontsize=9, color="#2C3E50")

    fig.tight_layout()
    out_png2 = os.path.join(out_dir, "figureS13_unsupervised_pipeline.png")
    _save(fig, out_png2)
    plt.close(fig)

    print("=" * 80)
    print("GENERATED FEATURIZATION SCHEMATIC FIGURE")
    print(f"Saved: {out_png} (+.pdf)")
    print(f"Saved: {out_png2} (+.pdf)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

