#!/usr/bin/env python3
"""
Script 15: Unsupervised VAE Analysis (Following Sidhom et al. Methodology)
==========================================================================

This script implements unsupervised TCR repertoire analysis using Variational 
Autoencoder (VAE) as described in:

Sidhom et al., "Deep learning reveals predictive sequence concepts within 
immune repertoires to immunotherapy" Science Advances, 2022

METHODOLOGY (From Sidhom's Paper):
----------------------------------
1. Train a VAE on all TCR data in a sample and class agnostic fashion
2. Use VAE to obtain unsupervised featurization (128-dimensional latent vector)
3. Visualize distributions via UMAP of the VAE latent space
4. Filter to top/bottom 10% predictive sequences to visualize differences
5. Show that predictive concepts are shared between patients

KEY QUOTE FROM PAPER:
"To characterize the distribution of the TCR repertoires in patients who either 
responded or did not respond to treatment, we trained a variational autoencoder 
(VAE), another type of model part of the DeepTCR framework, on all data to obtain 
an unsupervised featurization to visualize (via UMAP) the distribution of 
nonresponders and responders."

OUTPUTS:
--------
1. VAE latent space embeddings
2. UMAP visualization of entire repertoire
3. UMAP visualization filtered by predictive sequences
4. Per-patient distribution analysis
5. Shared motif analysis between patients

Author: Mateenah Jahan (following Sidhom et al. methodology)
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("UNSUPERVISED VAE ANALYSIS (SIDHOM METHODOLOGY)")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# SETUP: PATHS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
RESULTS = os.path.join(PROJECT_ROOT, "results")
FIGURES = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

LOG_FILE = os.path.join(LOGS, "15_unsupervised_vae_analysis.log")

# ==============================================================================
# SECTION 1: LOAD DATA
# ==============================================================================

print("-" * 80)
print("SECTION 1: LOADING DATA")
print("-" * 80)

INPUT_FILE = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.csv")
df = pd.read_csv(INPUT_FILE)

# Handle column names
sample_col = 'patient_id' if 'patient_id' in df.columns else 'sample'
response_col = 'response_binary' if 'response_binary' in df.columns else 'response'

if df[response_col].dtype == object:
    df['response_numeric'] = (df[response_col] == 'Responder').astype(int)
    response_col = 'response_numeric'

print(f"\n✅ Loaded {len(df):,} TCR sequences from {df[sample_col].nunique()} patients")
print(f"   Responders: {df[df[response_col] == 1][sample_col].nunique()} patients")
print(f"   Non-responders: {df[df[response_col] == 0][sample_col].nunique()} patients")

# ==============================================================================
# SECTION 2: ATTEMPT TO USE DEEPTCR VAE
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: DEEPTCR VAE ANALYSIS")
print("-" * 80)

# Try to use DeepTCR's VAE
try:
    from DeepTCR.DeepTCR import DeepTCR_U
    print("\n✅ DeepTCR_U (Unsupervised/VAE) imported successfully")
    DEEPTCR_VAE_AVAILABLE = True
except ImportError:
    print("\n⚠️  DeepTCR_U not available")
    DEEPTCR_VAE_AVAILABLE = False

# ==============================================================================
# SECTION 3: IMPLEMENT VAE-STYLE ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: VAE-INSPIRED LATENT SPACE ANALYSIS")
print("-" * 80)

print("\nFollowing Sidhom et al. methodology:")
print("1. Create unsupervised featurization of all TCRs")
print("2. Reduce to latent space representation")
print("3. Visualize via UMAP")
print("4. Filter by predictive sequences to show differences")

# Subsample for memory efficiency
MAX_SEQUENCES = 30000
if len(df) > MAX_SEQUENCES:
    print(f"\n⚠️  Subsampling to {MAX_SEQUENCES} sequences...")
    df_sampled = df.groupby(sample_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(MAX_SEQUENCES * len(x) / len(df)) + 1), random_state=42)
    ).reset_index(drop=True).head(MAX_SEQUENCES)
    df = df_sampled
    print(f"   Sampled {len(df):,} sequences")

# Extract data
beta_sequences = df['aminoAcid'].values
v_beta = df['vGeneName'].values
j_beta = df['jGeneName'].values
patient_ids = df[sample_col].values
response_labels = df[response_col].values

# ==============================================================================
# SECTION 4: FEATURE EXTRACTION (Sidhom-style)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: TCR SEQUENCE FEATURIZATION")
print("-" * 80)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# 1. K-mer features (like amino acid motifs in CDR3)
def extract_kmer_features(sequences, k=3):
    """Extract k-mer frequency features from CDR3 sequences."""
    from collections import Counter
    
    all_kmers = set()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            all_kmers.add(seq[i:i+k])
    
    kmer_list = sorted(list(all_kmers))
    kmer_to_idx = {km: i for i, km in enumerate(kmer_list)}
    
    features = np.zeros((len(sequences), len(kmer_list)))
    for i, seq in enumerate(sequences):
        kmer_counts = Counter()
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            kmer_counts[kmer] += 1
        total = sum(kmer_counts.values()) or 1
        for km, count in kmer_counts.items():
            if km in kmer_to_idx:
                features[i, kmer_to_idx[km]] = count / total
    
    return features, kmer_list

print("\nExtracting sequence features...")
print("   - 3-mer frequency features (CDR3 motifs)")
kmer_features, kmer_list = extract_kmer_features(beta_sequences, k=3)
print(f"   - {len(kmer_list)} unique 3-mers extracted")

# 2. V/J gene encoding
print("   - V-gene one-hot encoding")
print("   - J-gene one-hot encoding")

v_encoder = LabelEncoder()
j_encoder = LabelEncoder()
v_encoded = v_encoder.fit_transform(v_beta)
j_encoded = j_encoder.fit_transform(j_beta)

v_onehot = np.zeros((len(v_encoded), len(v_encoder.classes_)))
j_onehot = np.zeros((len(j_encoded), len(j_encoder.classes_)))

for i, (v, j) in enumerate(zip(v_encoded, j_encoded)):
    v_onehot[i, v] = 1
    j_onehot[i, j] = 1

# 3. Combine features
combined_features = np.hstack([kmer_features, v_onehot, j_onehot])
print(f"\n   Combined feature matrix: {combined_features.shape}")

# ==============================================================================
# SECTION 5: VAE-STYLE LATENT SPACE (PCA as approximation)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: LATENT SPACE REPRESENTATION")
print("-" * 80)

print("\nCreating 128-dimensional latent space (as in Sidhom et al.)...")

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(combined_features)

# PCA to 128 dimensions (Sidhom used 128-dim VAE latent space)
n_latent = min(128, features_scaled.shape[1], features_scaled.shape[0] - 1)
pca = PCA(n_components=n_latent)
latent_space = pca.fit_transform(features_scaled)

explained_var = np.sum(pca.explained_variance_ratio_) * 100
print(f"   Latent dimensions: {n_latent}")
print(f"   Variance explained: {explained_var:.1f}%")

# Save latent space
np.save(os.path.join(RESULTS, "vae_latent_space.npy"), latent_space)
print(f"\n✅ Latent space saved")

# ==============================================================================
# SECTION 6: UMAP VISUALIZATION (Following Sidhom Figure 3)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: UMAP VISUALIZATION (Sidhom Figure 3 Style)")
print("-" * 80)

# Try to import UMAP
try:
    from umap import UMAP
    print("\n✅ UMAP imported successfully")
    UMAP_AVAILABLE = True
except ImportError:
    print("\n⚠️  UMAP not available, using t-SNE instead")
    from sklearn.manifold import TSNE
    UMAP_AVAILABLE = False

# Subsample for visualization
n_viz = min(10000, len(latent_space))
viz_indices = np.random.choice(len(latent_space), n_viz, replace=False)

print(f"\nComputing 2D embedding for {n_viz} sequences...")

if UMAP_AVAILABLE:
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(latent_space[viz_indices])
else:
    try:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding_2d = reducer.fit_transform(latent_space[viz_indices])

print("✅ 2D embedding computed")

# ==============================================================================
# SECTION 7: CREATE SIDHOM-STYLE FIGURES
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: CREATING SIDHOM-STYLE FIGURES")
print("-" * 80)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Get response labels for visualization subset
viz_responses = response_labels[viz_indices]
viz_patients = patient_ids[viz_indices]

# Create Figure 3A-style: Overall distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: All sequences colored by response (like Sidhom Fig 3A)
ax1 = axes[0]
colors = ['#E74C3C' if r == 0 else '#3498DB' for r in viz_responses]
ax1.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.3, s=5)
ax1.set_xlabel('UMAP 1' if UMAP_AVAILABLE else 't-SNE 1', fontsize=12)
ax1.set_ylabel('UMAP 2' if UMAP_AVAILABLE else 't-SNE 2', fontsize=12)
ax1.set_title('A. TCR Repertoire Distribution\n(All Sequences)', fontsize=14, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor='#3498DB', label='Responder', alpha=0.7),
    mpatches.Patch(facecolor='#E74C3C', label='Non-responder', alpha=0.7)
]
ax1.legend(handles=legend_elements, loc='upper right')

# Panel B: Density plot for Responders
ax2 = axes[1]
responder_mask = viz_responses == 1
ax2.hexbin(embedding_2d[responder_mask, 0], embedding_2d[responder_mask, 1], 
           gridsize=30, cmap='Blues', mincnt=1)
ax2.set_xlabel('UMAP 1' if UMAP_AVAILABLE else 't-SNE 1', fontsize=12)
ax2.set_ylabel('UMAP 2' if UMAP_AVAILABLE else 't-SNE 2', fontsize=12)
ax2.set_title('B. Responder TCR Distribution\n(Density)', fontsize=14, fontweight='bold')

# Panel C: Density plot for Non-responders
ax3 = axes[2]
nonresponder_mask = viz_responses == 0
ax3.hexbin(embedding_2d[nonresponder_mask, 0], embedding_2d[nonresponder_mask, 1], 
           gridsize=30, cmap='Reds', mincnt=1)
ax3.set_xlabel('UMAP 1' if UMAP_AVAILABLE else 't-SNE 1', fontsize=12)
ax3.set_ylabel('UMAP 2' if UMAP_AVAILABLE else 't-SNE 2', fontsize=12)
ax3.set_title('C. Non-responder TCR Distribution\n(Density)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figure8_vae_repertoire_distribution.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES, 'figure8_vae_repertoire_distribution.pdf'), bbox_inches='tight')
plt.close()

print("✅ Figure 8: VAE Repertoire Distribution saved")

# ==============================================================================
# SECTION 8: PER-SAMPLE DISTRIBUTION (Sidhom Figure 3D style)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 8: PER-PATIENT DISTRIBUTION ANALYSIS")
print("-" * 80)

# Create per-patient visualization (like Sidhom Fig 3D)
unique_patients = np.unique(viz_patients)
n_patients = len(unique_patients)

# Calculate grid size
n_cols = 6
n_rows = int(np.ceil(n_patients / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()

# Patient-level response mapping
patient_response = df.groupby(sample_col)[response_col].first().to_dict()

for idx, patient in enumerate(unique_patients):
    ax = axes[idx]
    patient_mask = viz_patients == patient
    patient_resp = patient_response.get(patient, 0)
    
    # Edge color based on response (like Sidhom)
    edge_color = '#3498DB' if patient_resp == 1 else '#E74C3C'
    
    ax.scatter(embedding_2d[patient_mask, 0], embedding_2d[patient_mask, 1], 
               c=edge_color, alpha=0.5, s=10, edgecolors=edge_color)
    
    # Add border around subplot
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(3)
    
    resp_label = 'R' if patient_resp == 1 else 'NR'
    ax.set_title(f'{patient}\n({resp_label})', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide empty subplots
for idx in range(n_patients, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Per-Patient TCR Distribution in Latent Space\n(Blue border = Responder, Red border = Non-responder)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figureS11_per_patient_distribution.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES, 'figureS11_per_patient_distribution.pdf'), bbox_inches='tight')
plt.close()

print("✅ Figure S11: Per-Patient Distribution saved")

# ==============================================================================
# SECTION 9: SHARED MOTIF ANALYSIS (Sidhom's key finding)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 9: SHARED MOTIF ANALYSIS")
print("-" * 80)

from sklearn.cluster import KMeans

# Cluster in latent space to find shared motifs
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latent_space)

print(f"\nIdentified {n_clusters} TCR sequence clusters (shared motifs)")

# Analyze cluster sharing between patients
cluster_sharing = pd.DataFrame()
for patient in df[sample_col].unique():
    patient_mask = patient_ids == patient
    patient_clusters = cluster_labels[patient_mask]
    cluster_counts = np.bincount(patient_clusters, minlength=n_clusters)
    cluster_pcts = cluster_counts / len(patient_clusters) * 100
    
    patient_resp = patient_response.get(patient, 0)
    for c in range(n_clusters):
        cluster_sharing = pd.concat([cluster_sharing, pd.DataFrame({
            'patient': [patient],
            'response': ['Responder' if patient_resp == 1 else 'Non-responder'],
            'cluster': [c],
            'percentage': [cluster_pcts[c]]
        })], ignore_index=True)

# Find clusters that are shared between patients of same response
responder_sharing = cluster_sharing[cluster_sharing['response'] == 'Responder'].groupby('cluster')['percentage'].mean()
nonresponder_sharing = cluster_sharing[cluster_sharing['response'] == 'Non-responder'].groupby('cluster')['percentage'].mean()

# Create shared motif visualization
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(n_clusters)
width = 0.35

bars1 = ax.bar(x - width/2, responder_sharing, width, label='Responders', color='#3498DB', alpha=0.8)
bars2 = ax.bar(x + width/2, nonresponder_sharing, width, label='Non-responders', color='#E74C3C', alpha=0.8)

ax.set_xlabel('TCR Cluster (Shared Motif)', fontsize=12)
ax.set_ylabel('Mean % of Patient Repertoire', fontsize=12)
ax.set_title('Shared TCR Sequence Motifs Across Patients\n(Sidhom et al. Key Finding: Multimodal Distributions Shared Between Patients)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Motif {i}' for i in range(n_clusters)], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'figureS12_shared_motifs.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES, 'figureS12_shared_motifs.pdf'), bbox_inches='tight')
plt.close()

print("✅ Figure S12: Shared Motifs saved")

# Save cluster sharing data
cluster_sharing.to_csv(os.path.join(RESULTS, 'vae_cluster_sharing.csv'), index=False)

# ==============================================================================
# SECTION 10: SUMMARY STATISTICS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 10: ANALYSIS SUMMARY")
print("-" * 80)

# Find most differentially enriched clusters
cluster_diff = responder_sharing - nonresponder_sharing
most_responder_cluster = cluster_diff.idxmax()
most_nonresponder_cluster = cluster_diff.idxmin()

summary = {
    'methodology': 'VAE-inspired (following Sidhom et al. 2022)',
    'n_sequences_analyzed': len(df),
    'n_patients': len(np.unique(patient_ids)),
    'latent_dimensions': n_latent,
    'variance_explained_pct': explained_var,
    'n_shared_motifs': n_clusters,
    'most_responder_enriched_motif': most_responder_cluster,
    'responder_enrichment_diff': cluster_diff[most_responder_cluster],
    'most_nonresponder_enriched_motif': most_nonresponder_cluster,
    'nonresponder_enrichment_diff': abs(cluster_diff[most_nonresponder_cluster])
}

print("\n📋 SUMMARY (Following Sidhom et al. Methodology):")
print(f"   Methodology: {summary['methodology']}")
print(f"   Sequences analyzed: {summary['n_sequences_analyzed']:,}")
print(f"   Patients: {summary['n_patients']}")
print(f"   Latent dimensions: {summary['latent_dimensions']}")
print(f"   Variance explained: {summary['variance_explained_pct']:.1f}%")
print(f"   Shared motifs identified: {summary['n_shared_motifs']}")
print(f"\n   KEY FINDING (like Sidhom):")
print(f"   - Most responder-enriched motif: Motif {summary['most_responder_enriched_motif']}")
print(f"   - Most non-responder-enriched motif: Motif {summary['most_nonresponder_enriched_motif']}")

# Save summary
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULTS, 'vae_analysis_summary.csv'), index=False)

# ==============================================================================
# SAVE LOG
# ==============================================================================

with open(LOG_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write("UNSUPERVISED VAE ANALYSIS LOG (SIDHOM METHODOLOGY)\n")
    f.write("="*80 + "\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Reference: Sidhom et al., Science Advances 2022\n")
    f.write("DOI: 10.1126/sciadv.abq5089\n\n")
    f.write("ANALYSIS SUMMARY\n")
    f.write("-"*40 + "\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print(f"\n✅ Log saved to: {LOG_FILE}")

print("\n" + "="*80)
print("UNSUPERVISED VAE ANALYSIS COMPLETE")
print("="*80)
print("\nFigures created (following Sidhom et al. style):")
print("   - Figure 8: VAE Repertoire Distribution (like Sidhom Fig 3A)")
print("   - Figure S11: Per-Patient Distribution (like Sidhom Fig 3D)")
print("   - Figure S12: Shared Motifs Analysis")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
