#!/usr/bin/env python3
"""
Script 15: Unsupervised sequence-space visualization (Sidhom-style, adapted)
============================================================================

What the 2022 Science Advances paper does
----------------------------------------
They train a VAE (class-agnostic) to produce a ~128-D latent representation of
TCR sequences (in their case, including CDR3 + V/D/J + HLA), then visualize with
UMAP. They then overlay highly predictive sequences (top/bottom %) identified by
the supervised MIL model.

What we can do with your data
-----------------------------
You have: CDR3 (beta) + V gene + J gene, and (optionally) response labels.
You do NOT have HLA. So we implement a practical, reproducible approximation:

1) Build an unsupervised sequence embedding from:
   - CDR3 k-mer counts (k=3 by default)
   - V/J one-hot
2) Reduce to 128-D with TruncatedSVD (acts like a "latent space")
3) Reduce to 2-D with UMAP if available; otherwise PCA fallback
4) Optionally overlay "top/bottom predictive sequences" using:
   - attention_weight from results/attention_weights_all.csv (your proxy importance)

Important limitation
--------------------
This script does NOT create a biologically faithful VAE; it provides an
unsupervised latent space + 2D visualization that you can use similarly to the
paper to inspect whether high-importance sequences cluster into shared modes.

Outputs
-------
results/unsupervised/sequence_space_2d.csv
figures/unsupervised/sequence_space_all.png (+.pdf)
figures/unsupervised/sequence_space_top_bottom.png (+.pdf)
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _load_ready_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "aminoAcid", "vGeneName", "jGeneName"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _load_attention(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    att = pd.read_csv(path)
    # Minimal sanity check
    required = {"aminoAcid", "vGeneName", "jGeneName", "patient_id", "attention_weight"}
    if not required.issubset(set(att.columns)):
        return None
    # IMPORTANT: attention file can contain duplicate keys (same CDR3+V/J within patient).
    # If we merge it raw we can create a many-to-many join and explode memory.
    # Collapse to a unique key by taking max attention weight within each key.
    key = ["patient_id", "aminoAcid", "vGeneName", "jGeneName"]
    att = att.copy()
    att["patient_id"] = att["patient_id"].astype(str)
    att = att[key + ["attention_weight"]].groupby(key, as_index=False)["attention_weight"].max()
    return att


def _build_sequence_latent(
    df: pd.DataFrame,
    *,
    kmer_k: int,
    svd_dim: int,
    random_state: int,
) -> np.ndarray:
    from scipy.sparse import hstack
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import OneHotEncoder

    seqs = df["aminoAcid"].astype(str).fillna("").values
    v = df["vGeneName"].astype(str).fillna("UNK").values
    j = df["jGeneName"].astype(str).fillna("UNK").values

    def kmer_analyzer(s: str) -> list[str]:
        s = s.strip().upper()
        if len(s) < kmer_k:
            return []
        return [s[i : i + kmer_k] for i in range(len(s) - kmer_k + 1)]

    vec = CountVectorizer(analyzer=kmer_analyzer, lowercase=False, min_df=2)
    X_kmer = vec.fit_transform(seqs)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_vj = ohe.fit_transform(np.column_stack([v, j]))

    X = hstack([X_kmer, X_vj], format="csr")
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    return svd.fit_transform(X).astype(np.float32)


def _to_2d(latent: np.ndarray, *, random_state: int) -> np.ndarray:
    # Prefer UMAP (paper), fallback to PCA.
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(latent).astype(np.float32)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(latent).astype(np.float32)


def _plot_scatter(
    out_png: str,
    *,
    df2d: pd.DataFrame,
    title: str,
    color_by: str,
    alpha: float,
    s: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    vals = df2d[color_by].to_numpy()
    sc = ax.scatter(df2d["x"], df2d["y"], c=vals, cmap="coolwarm", s=s, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_by)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_png.lower().endswith(".png"):
        fig.savefig(out_png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_processed = os.path.join(project_root, "data_processed")
    results_unsup = os.path.join(project_root, "results", "unsupervised")
    figures_unsup = os.path.join(project_root, "figures", "unsupervised")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=os.path.join(data_processed, "deeptcr_trb_ready.csv"),
        help="DeepTCR-ready CSV (CDR3 + V/J + patient_id).",
    )
    parser.add_argument(
        "--attention-csv",
        default=os.path.join(project_root, "results", "attention_weights_all.csv"),
        help="Optional attention CSV to overlay (default: results/attention_weights_all.csv).",
    )
    parser.add_argument("--kmer-k", type=int, default=3)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--top-frac", type=float, default=0.10, help="Top/bottom fraction for overlay plots.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=50000,
        help="Downsample sequences for visualization (keeps memory bounded).",
    )
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()

    _ensure_dirs(results_unsup, figures_unsup)

    df = _load_ready_csv(args.input_csv).copy()
    df["patient_id"] = df["patient_id"].astype(str)

    # DeepTCR convention: ignore sequences > 40aa
    max_len = 40
    mask = df["aminoAcid"].astype(str).str.len().le(max_len)
    df = df.loc[mask].reset_index(drop=True)

    # Merge attention weights if present (gives us a "predictive importance" proxy).
    att = _load_attention(args.attention_csv)
    if att is not None:
        key = ["patient_id", "aminoAcid", "vGeneName", "jGeneName"]
        df = df.merge(att, on=key, how="left", validate="m:1")
    else:
        df["attention_weight"] = np.nan

    # Downsample for visualization (paper-style UMAP doesn't require plotting all 200k+ points).
    # Keep all very high-attention sequences if available, then fill remaining with random sample.
    rng = np.random.default_rng(args.random_state)
    if args.max_sequences is not None and len(df) > args.max_sequences:
        if df["attention_weight"].notna().any():
            keep_top = df["attention_weight"] >= df["attention_weight"].quantile(0.99)
            df_top = df.loc[keep_top]
            remaining = max(args.max_sequences - len(df_top), 0)
            df_rest = df.loc[~keep_top]
            if remaining > 0 and len(df_rest) > remaining:
                take_idx = rng.choice(df_rest.index.to_numpy(), size=remaining, replace=False)
                df = pd.concat([df_top, df_rest.loc[take_idx]], axis=0).sample(frac=1.0, random_state=args.random_state)
            else:
                df = df_top
        else:
            take_idx = rng.choice(df.index.to_numpy(), size=args.max_sequences, replace=False)
            df = df.loc[take_idx]
        df = df.reset_index(drop=True)

    latent = _build_sequence_latent(
        df,
        kmer_k=args.kmer_k,
        svd_dim=args.svd_dim,
        random_state=args.random_state,
    )
    xy = _to_2d(latent, random_state=args.random_state)

    df2d = df[["patient_id", "aminoAcid", "vGeneName", "jGeneName"]].copy()
    if "response_binary" in df.columns:
        df2d["response_binary"] = df["response_binary"].astype(float)
    df2d["attention_weight"] = df["attention_weight"].astype(float)
    df2d["x"] = xy[:, 0]
    df2d["y"] = xy[:, 1]

    out_csv = os.path.join(results_unsup, "sequence_space_2d.csv")
    df2d.to_csv(out_csv, index=False)

    # Plot 1: all sequences colored by attention (or density proxy)
    out_all = os.path.join(figures_unsup, "sequence_space_all.png")
    color_by = "attention_weight" if df2d["attention_weight"].notna().any() else ("response_binary" if "response_binary" in df2d.columns else "x")
    _plot_scatter(
        out_all,
        df2d=df2d.fillna({color_by: 0.0}),
        title="Unsupervised sequence space (all sequences)",
        color_by=color_by,
        alpha=0.15,
        s=6,
    )

    # Plot 2: top/bottom by attention_weight (paper-style overlay idea)
    out_tb = os.path.join(figures_unsup, "sequence_space_top_bottom.png")
    if df2d["attention_weight"].notna().any():
        q_hi = df2d["attention_weight"].quantile(1.0 - args.top_frac)
        q_lo = df2d["attention_weight"].quantile(args.top_frac)
        overlay = df2d[(df2d["attention_weight"] >= q_hi) | (df2d["attention_weight"] <= q_lo)].copy()
        overlay["is_top"] = (overlay["attention_weight"] >= q_hi).astype(float)
        _plot_scatter(
            out_tb,
            df2d=overlay.fillna({"is_top": 0.0}),
            title=f"Top vs bottom {args.top_frac:.0%} by attention (proxy predictive sequences)",
            color_by="is_top",
            alpha=0.7,
            s=10,
        )
    else:
        # Fallback: just write an empty-ish plot from df2d
        _plot_scatter(
            out_tb,
            df2d=df2d.assign(is_top=0.0),
            title="Top/bottom overlay unavailable (no attention weights found)",
            color_by="is_top",
            alpha=0.4,
            s=8,
        )

    print("Saved:")
    print(f"  - {out_csv}")
    print(f"  - {out_all}")
    print(f"  - {out_tb}")


if __name__ == "__main__":
    main()

