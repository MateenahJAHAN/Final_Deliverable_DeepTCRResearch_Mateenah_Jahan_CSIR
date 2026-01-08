#!/usr/bin/env python3
"""
Script 14: Unsupervised Patient Stratification (No labels required)
===================================================================

Goal
----
Perform unsupervised learning on TCR-beta repertoires to stratify patients into
clusters using only:
  - CDR3 amino acid sequence (aminoAcid)
  - V gene (vGeneName)
  - J gene (jGeneName)

Why this exists
---------------
Your current pipeline (scripts 01–13) is supervised (patient labels are used to
train an attention/MIL model). DeepTCR (Sidhom et al.) also supports unsupervised
representation learning (e.g., DeepTCR_U / VAE-style embeddings) that can be used
to cluster sequences or patients without using response labels.

Important limitation (scientific)
---------------------------------
Unsupervised clustering can separate patients into groups, but **cannot** know
which group corresponds to "Responder" vs "Non-responder" unless you:
  - have at least some labeled outcomes (even a small subset), or
  - have an external biological rule to assign meaning to clusters.

Implementation strategy
-----------------------
This script provides two modes:
  1) DeepTCR_U (if installed and importable): learns embeddings with DeepTCR.
  2) Fallback (default, always works): k-mer + V/J one-hot -> SVD embedding
     -> patient-level pooling -> KMeans/GMM clustering.

Outputs
-------
Creates:
  - results/unsupervised/patient_embeddings.csv
  - results/unsupervised/patient_clusters.csv
  - results/unsupervised/cluster_report.txt
  - figures/unsupervised/patient_clusters_pca.png

Run
---
python scripts/14_unsupervised_patient_stratification.py
python scripts/14_unsupervised_patient_stratification.py --n-clusters 3
python scripts/14_unsupervised_patient_stratification.py --method deeptcr_u
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: str
    data_processed: str
    results_dir: str
    figures_dir: str


def _project_paths() -> Paths:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results", "unsupervised")
    figures_dir = os.path.join(project_root, "figures", "unsupervised")
    return Paths(
        project_root=project_root,
        data_processed=os.path.join(project_root, "data_processed"),
        results_dir=results_dir,
        figures_dir=figures_dir,
    )


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _load_deeptcr_ready_csv(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"patient_id", "aminoAcid", "vGeneName", "jGeneName"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")
    return df


def _maybe_get_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    # Your dataset contains patient-level labels replicated per sequence as response_binary.
    if "response_binary" not in df.columns:
        return None
    # Use first label per patient (should be constant within patient)
    return df.groupby("patient_id")["response_binary"].first()


def _sklearn_sequence_to_patient_embedding(
    df: pd.DataFrame,
    *,
    kmer_k: int,
    svd_dim: int,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      patient_meta_df: index=patient_id, columns include patient_seq_count
      patient_embeddings: shape (n_patients, svd_dim)
    """
    from scipy.sparse import hstack  # optional dependency via scikit-learn
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

    # CDR3 k-mers (sparse counts)
    vec = CountVectorizer(analyzer=kmer_analyzer, lowercase=False, min_df=2)
    X_kmer = vec.fit_transform(seqs)

    # V/J (sparse one-hot)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_vj = ohe.fit_transform(np.column_stack([v, j]))

    X = hstack([X_kmer, X_vj], format="csr")

    # SVD -> dense sequence embeddings
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    Z_seq = svd.fit_transform(X)  # (n_sequences, svd_dim)

    # Patient pooling: simple mean of sequence embeddings per patient
    pid = df["patient_id"].astype(str).values
    patient_ids, inv = np.unique(pid, return_inverse=True)
    Z_patient = np.zeros((len(patient_ids), svd_dim), dtype=np.float32)
    counts = np.bincount(inv).astype(np.float32)
    for d in range(svd_dim):
        Z_patient[:, d] = np.bincount(inv, weights=Z_seq[:, d]).astype(np.float32)
    Z_patient /= np.maximum(counts[:, None], 1.0)

    patient_meta = pd.DataFrame(
        {"patient_seq_count": counts.astype(int)},
        index=pd.Index(patient_ids, name="patient_id"),
    )

    return patient_meta, Z_patient


def _cluster_patients(
    Z_patient: np.ndarray,
    *,
    n_clusters: int,
    algorithm: str,
    random_state: int,
) -> np.ndarray:
    if algorithm == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        return model.fit_predict(Z_patient)
    if algorithm == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        return model.fit_predict(Z_patient)
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def _write_report(
    report_path: str,
    *,
    patient_clusters: pd.DataFrame,
    labels_by_patient: Optional[pd.Series],
) -> None:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    lines: list[str] = []
    lines.append("Unsupervised clustering report")
    lines.append("=" * 80)
    lines.append(f"Patients: {len(patient_clusters):,}")
    lines.append(f"Clusters: {patient_clusters['cluster'].nunique()}")
    lines.append("")
    lines.append("Cluster sizes:")
    lines.append(patient_clusters["cluster"].value_counts().sort_index().to_string())
    lines.append("")

    # Silhouette is undefined for 1 cluster; skip if needed.
    try:
        sil = silhouette_score(
            patient_clusters.filter(regex=r"^emb_").to_numpy(),
            patient_clusters["cluster"].to_numpy(),
        )
        lines.append(f"Silhouette score (higher is better): {sil:.4f}")
    except Exception as e:
        lines.append(f"Silhouette score: unavailable ({e})")

    if labels_by_patient is not None:
        # Evaluate cluster agreement with response labels (NOT used for training).
        common = patient_clusters.index.intersection(labels_by_patient.index)
        y = labels_by_patient.loc[common].astype(int).to_numpy()
        c = patient_clusters.loc[common, "cluster"].astype(int).to_numpy()
        ari = adjusted_rand_score(y, c)
        nmi = normalized_mutual_info_score(y, c)
        lines.append("")
        lines.append("Optional evaluation vs known response labels (post-hoc only):")
        lines.append(f"Adjusted Rand Index (ARI): {ari:.4f}")
        lines.append(f"Normalized Mutual Information (NMI): {nmi:.4f}")

        # Cluster -> label mapping accuracy (only meaningful for n_clusters=2).
        if patient_clusters["cluster"].nunique() == 2 and len(np.unique(y)) == 2:
            # Try both mappings
            pred0 = (c == 0).astype(int)
            pred1 = (c == 1).astype(int)
            acc0 = (pred0 == y).mean()
            acc1 = (pred1 == y).mean()
            lines.append(f"Best 2-cluster mapping accuracy: {max(acc0, acc1):.4f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_pca(
    out_png: str,
    *,
    patient_clusters: pd.DataFrame,
    labels_by_patient: Optional[pd.Series],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = patient_clusters.filter(regex=r"^emb_").to_numpy()
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    clusters = patient_clusters["cluster"].astype(int).to_numpy()

    # User-requested colors: Responder=green, Non-responder=orange (when labels exist)
    if labels_by_patient is not None:
        lab = labels_by_patient.reindex(patient_clusters.index).astype(float)
        is_r = lab.fillna(0).astype(int).to_numpy() == 1
        is_nr = ~is_r

        # Color by response, outline by cluster id for extra context
        ax.scatter(
            X2[is_nr, 0],
            X2[is_nr, 1],
            c="#F39C12",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label="Non-responder",
        )
        ax.scatter(
            X2[is_r, 0],
            X2[is_r, 1],
            c="#27AE60",
            s=70,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            label="Responder",
        )
        ax.set_title("Patient embeddings (PCA) colored by response")
        ax.legend(loc="best", title="Response")
    else:
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap="tab10", s=60, alpha=0.9)
        ax.set_title("Patient clusters (PCA of unsupervised embeddings)")
        ax.legend(*sc.legend_elements(), title="cluster", loc="best")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)

    # annotate with R/NR + cluster id for readability
    if labels_by_patient is not None:
        lab = labels_by_patient.reindex(patient_clusters.index)
        for i, pid in enumerate(patient_clusters.index.tolist()):
            if pd.isna(lab.loc[pid]):
                continue
            txt = ("R" if int(lab.loc[pid]) == 1 else "NR") + f" (c{clusters[i]})"
            ax.annotate(txt, (X2[i, 0], X2[i, 1]), fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _run_deeptcr_u(df: pd.DataFrame, *, n_clusters: int, random_state: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Best-effort DeepTCR_U path. If DeepTCR_U is not installed/available, raise.

    Notes:
    - DeepTCR API can differ between versions. This function intentionally keeps a
      narrow surface area so the fallback method remains available.
    - We do not use response labels (Y) for training here.
    """
    try:
        from DeepTCR.DeepTCR import DeepTCR_U  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DeepTCR_U is not available in this environment. "
            "Install DeepTCR (and compatible TensorFlow) or run with --method sklearn."
        ) from e

    beta_sequences = df["aminoAcid"].astype(str).values
    v_beta = df["vGeneName"].astype(str).values
    j_beta = df["jGeneName"].astype(str).values
    pids = df["patient_id"].astype(str).values

    # DeepTCR expects alpha placeholders.
    n = len(df)
    alpha_sequences = np.array(["AAA"] * n)
    v_alpha = np.array(["TRAV1-1"] * n)
    j_alpha = np.array(["TRAJ1"] * n)

    # Initialize and load data (no Y passed).
    dt = DeepTCR_U("DeepTCR_Unsupervised")
    dt.Load_Data(
        beta_sequences=beta_sequences,
        v_beta=v_beta,
        j_beta=j_beta,
        alpha_sequences=alpha_sequences,
        v_alpha=v_alpha,
        j_alpha=j_alpha,
        sample_labels=pids,
    )

    # Train unsupervised model (VAE-style). Parameters vary by version; keep minimal.
    # If your DeepTCR version exposes different args, update here.
    dt.Train(epochs_min=10, size_of_net="small", batch_size=256, suppress_output=True)  # type: ignore

    # Attempt to extract patient-level features. If not available, fall back to sequence features + pooling.
    if hasattr(dt, "Get_Seq_Features"):
        Z_seq = dt.Get_Seq_Features()  # type: ignore
    elif hasattr(dt, "seq_features"):
        Z_seq = dt.seq_features  # type: ignore
    else:  # pragma: no cover
        raise RuntimeError("Could not extract features from DeepTCR_U object.")

    # Pool to patient embeddings
    patient_ids, inv = np.unique(pids, return_inverse=True)
    Z_patient = np.zeros((len(patient_ids), Z_seq.shape[1]), dtype=np.float32)
    counts = np.bincount(inv).astype(np.float32)
    for d in range(Z_seq.shape[1]):
        Z_patient[:, d] = np.bincount(inv, weights=Z_seq[:, d]).astype(np.float32)
    Z_patient /= np.maximum(counts[:, None], 1.0)

    meta = pd.DataFrame(
        {"patient_seq_count": counts.astype(int)},
        index=pd.Index(patient_ids, name="patient_id"),
    )
    return meta, Z_patient


def main() -> None:
    paths = _project_paths()

    parser = argparse.ArgumentParser(description="Unsupervised patient stratification from TCR data.")
    parser.add_argument(
        "--input-csv",
        default=os.path.join(paths.data_processed, "deeptcr_trb_ready.csv"),
        help="DeepTCR-ready CSV (default: data_processed/deeptcr_trb_ready.csv)",
    )
    parser.add_argument("--method", choices=["sklearn", "deeptcr_u"], default="sklearn")
    parser.add_argument("--kmer-k", type=int, default=3, help="k for k-mer features (sklearn method)")
    parser.add_argument("--svd-dim", type=int, default=64, help="SVD embedding dimension (sklearn method)")
    parser.add_argument("--n-clusters", type=int, default=2, help="Number of patient clusters")
    parser.add_argument("--cluster-algorithm", choices=["kmeans", "gmm"], default="kmeans")
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument(
        "--eval-with-labels",
        action="store_true",
        help="If response_binary exists, compute post-hoc agreement metrics (ARI/NMI).",
    )

    args = parser.parse_args()

    _ensure_dirs(paths.results_dir, paths.figures_dir)

    print("=" * 80)
    print("UNSUPERVISED PATIENT STRATIFICATION (SCRIPT 14)")
    print("=" * 80)
    print(f"Input: {args.input_csv}")
    print(f"Method: {args.method}")
    print(f"Clusters: {args.n_clusters} ({args.cluster_algorithm})")
    print("")

    df = _load_deeptcr_ready_csv(args.input_csv)
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)

    # Optional: filter very long sequences for DeepTCR compatibility and consistent k-mers.
    max_len = 40
    if "aminoAcid" in df.columns:
        mask = df["aminoAcid"].astype(str).str.len().le(max_len)
        filtered = int((~mask).sum())
        if filtered > 0:
            print(f"Filtering {filtered} sequences > {max_len} AA (DeepTCR convention).")
        df = df.loc[mask].reset_index(drop=True)

    # Use labels ONLY for plotting (colors) if present; clustering never uses labels.
    labels_for_plot = _maybe_get_labels(df)
    labels_for_eval = labels_for_plot if args.eval_with_labels else None

    if args.method == "deeptcr_u":
        patient_meta, Z_patient = _run_deeptcr_u(df, n_clusters=args.n_clusters, random_state=args.random_state)
    else:
        patient_meta, Z_patient = _sklearn_sequence_to_patient_embedding(
            df,
            kmer_k=args.kmer_k,
            svd_dim=args.svd_dim,
            random_state=args.random_state,
        )

    clusters = _cluster_patients(
        Z_patient,
        n_clusters=args.n_clusters,
        algorithm=args.cluster_algorithm,
        random_state=args.random_state,
    )

    # Assemble output DF
    emb_cols = [f"emb_{i:03d}" for i in range(Z_patient.shape[1])]
    patient_clusters = patient_meta.copy()
    patient_clusters["cluster"] = clusters.astype(int)
    patient_clusters = pd.concat(
        [patient_clusters, pd.DataFrame(Z_patient, index=patient_clusters.index, columns=emb_cols)],
        axis=1,
    )

    # Save outputs
    out_embeddings = os.path.join(paths.results_dir, "patient_embeddings.csv")
    out_clusters = os.path.join(paths.results_dir, "patient_clusters.csv")
    out_report = os.path.join(paths.results_dir, "cluster_report.txt")
    out_fig = os.path.join(paths.figures_dir, "patient_clusters_pca.png")

    patient_clusters[emb_cols].to_csv(out_embeddings, index=True)
    patient_clusters[["patient_seq_count", "cluster"]].to_csv(out_clusters, index=True)
    _write_report(out_report, patient_clusters=patient_clusters, labels_by_patient=labels_for_eval)
    _plot_pca(out_fig, patient_clusters=patient_clusters, labels_by_patient=labels_for_plot)

    print("Saved:")
    print(f"  - {out_embeddings}")
    print(f"  - {out_clusters}")
    print(f"  - {out_report}")
    print(f"  - {out_fig}")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()

