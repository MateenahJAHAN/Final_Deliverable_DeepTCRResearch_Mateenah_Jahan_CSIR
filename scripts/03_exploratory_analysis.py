#!/usr/bin/env python3
"""
Script 03: Exploratory Data Analysis and Visualizations
=======================================================

This script corresponds to Cells 17-26 of the original Untitled1.ipynb notebook.

PURPOSE:
--------
1. Perform comprehensive exploratory data analysis (EDA) on TCR repertoire data
2. Generate publication-quality visualizations with proper legends and labels
3. Analyze sequence characteristics (length, V/J gene usage, repertoire size)
4. Compare responders vs non-responders
5. Create patient-level summary statistics

BIOLOGICAL QUESTIONS ADDRESSED:
-------------------------------
1. What is the distribution of CDR3 sequence lengths?
2. Which V genes are most commonly used?
3. Do responders have different repertoire sizes than non-responders?
4. Are there differences in sequence characteristics between response groups?
5. What is the overall response rate in the dataset?

VISUALIZATIONS CREATED:
-----------------------
1. top10_v_genes.png - Bar chart of most common V genes
2. sequence_length_distribution.png - Histogram of CDR3 lengths
3. length_hist_by_response.png - Length distribution by response group
4. response_pie.png - Pie chart of responder vs non-responder distribution
5. repertoire_size_by_response.png - Box plot comparing repertoire sizes
6. patient_summary_all.csv - Per-patient statistics table

STATISTICAL ANALYSES:
--------------------
- Sequence length statistics (mean, median, std)
- V/J gene usage frequencies
- Repertoire size per patient
- Response label distribution
- Patient-level aggregations

OUTPUTS:
--------
All figures saved to: ../figures/
Patient summary saved to: ../results/patient_summary_all.csv
Log file saved to: ../logs/03_exploratory_analysis.log

Author: Converted from Untitled1.ipynb
Date: December 24, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plotting style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)

# ==============================================================================
# SETUP: PATHS AND CONFIGURATION
# ==============================================================================

print("="*80)
print("EXPLORATORY DATA ANALYSIS - SCRIPT 03")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
FIGURES = os.path.join(PROJECT_ROOT, "figures")
RESULTS = os.path.join(PROJECT_ROOT, "results")
LOGS = os.path.join(PROJECT_ROOT, "logs")

# Define file paths
INPUT_FILE = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.csv")
LOG_FILE = os.path.join(LOGS, "03_exploratory_analysis.log")
PATIENT_SUMMARY_FILE = os.path.join(RESULTS, "patient_summary_all.csv")

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==============================================================================
# SECTION 1: LOAD DATA
# ==============================================================================

print("-" * 80)
print("SECTION 1: LOADING DATA")
print("-" * 80)

print(f"\nReading data from: {INPUT_FILE}\n")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Data loaded successfully!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Patients: {df['patient_id'].nunique()}")

except FileNotFoundError:
    print(f"❌ ERROR: File not found at {INPUT_FILE}")
    print("   Please run script 02_data_loading.py first")
    sys.exit(1)

# ==============================================================================
# SECTION 2: OVERALL DATASET STATISTICS (Notebook Cells 17-19)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: OVERALL DATASET STATISTICS")
print("-" * 80)

print("\n1. SEQUENCE STATUS DISTRIBUTION:")
print("-" * 80)
sequence_status = df['sequenceStatus'].value_counts()
for status, count in sequence_status.items():
    percentage = (count / len(df)) * 100
    print(f"   {status:10s}: {count:7,} sequences ({percentage:5.2f}%)")

print("\n2. SEQUENCE LENGTH STATISTICS:")
print("-" * 80)
# Calculate sequence lengths
df['cdr3_length'] = df['aminoAcid'].str.len()

length_stats = df['cdr3_length'].describe()
print(f"   Count:  {length_stats['count']:,.0f}")
print(f"   Mean:   {length_stats['mean']:.2f} amino acids")
print(f"   Median: {length_stats['50%']:.0f} amino acids")
print(f"   Std:    {length_stats['std']:.2f}")
print(f"   Min:    {length_stats['min']:.0f} amino acids")
print(f"   Max:    {length_stats['max']:.0f} amino acids")
print(f"   Q1:     {length_stats['25%']:.0f} amino acids")
print(f"   Q3:     {length_stats['75%']:.0f} amino acids")

print("\n3. V GENE USAGE:")
print("-" * 80)
v_gene_counts = df['vGeneName'].value_counts()
print(f"   Total unique V genes: {len(v_gene_counts)}")
print(f"\n   Top 10 V genes:")
for i, (gene, count) in enumerate(v_gene_counts.head(10).items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i:2d}. {gene:15s}: {count:7,} ({percentage:5.2f}%)")

print("\n4. J GENE USAGE:")
print("-" * 80)
j_gene_counts = df['jGeneName'].value_counts()
print(f"   Total unique J genes: {len(j_gene_counts)}")
print(f"\n   Top 10 J genes:")
for i, (gene, count) in enumerate(j_gene_counts.head(10).items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i:2d}. {gene:15s}: {count:7,} ({percentage:5.2f}%)")

print("\n5. PATIENT-LEVEL STATISTICS:")
print("-" * 80)
patient_seq_counts = df.groupby('patient_id').size()
print(f"   Total patients: {len(patient_seq_counts)}")
print(f"   Sequences per patient:")
print(f"   - Mean:   {patient_seq_counts.mean():,.0f}")
print(f"   - Median: {patient_seq_counts.median():,.0f}")
print(f"   - Min:    {patient_seq_counts.min():,}")
print(f"   - Max:    {patient_seq_counts.max():,}")

# ==============================================================================
# SECTION 3: VISUALIZATIONS (Notebook Cells 20-22)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: CREATING VISUALIZATIONS")
print("-" * 80)

# ----------------------
# FIGURE 1: Top 10 V Genes
# ----------------------

print("\n1. Creating 'Top 10 V Genes' bar chart...")

fig, ax = plt.subplots(figsize=(12, 6))

top10_v = v_gene_counts.head(10)
colors = sns.color_palette("viridis", len(top10_v))

bars = ax.barh(range(len(top10_v)), top10_v.values, color=colors)
ax.set_yticks(range(len(top10_v)))
ax.set_yticklabels(top10_v.index)
ax.set_xlabel('Number of Sequences', fontsize=12, fontweight='bold')
ax.set_ylabel('V Gene', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Common TRB V Genes\nT-Cell Receptor Beta Chain Variable Gene Usage',
             fontsize=14, fontweight='bold', pad=20)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top10_v.values)):
    percentage = (value / len(df)) * 100
    ax.text(value + 200, bar.get_y() + bar.get_height()/2,
            f'{value:,} ({percentage:.1f}%)',
            va='center', fontsize=10)

ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIGURES, 'top10_v_genes.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Saved: {fig_path}")

# ----------------------
# FIGURE 2: Sequence Length Distribution
# ----------------------

print("\n2. Creating 'Sequence Length Distribution' histogram...")

fig, ax = plt.subplots(figsize=(10, 6))

n, bins, patches = ax.hist(df['cdr3_length'], bins=range(5, 30),
                           color='steelblue', edgecolor='black', alpha=0.7)

# Color bars by frequency
norm = plt.Normalize(n.min(), n.max())
colors_grad = plt.cm.viridis(norm(n))
for patch, color in zip(patches, colors_grad):
    patch.set_facecolor(color)

ax.set_xlabel('CDR3 Sequence Length (amino acids)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
ax.set_title('Distribution of CDR3 Sequence Lengths\nTCR Beta Chain CDR3 Hypervariable Region',
             fontsize=14, fontweight='bold', pad=20)

# Add statistics text box
stats_text = f'Mean: {length_stats["mean"]:.1f} aa\nMedian: {length_stats["50%"]:.0f} aa\nStd: {length_stats["std"]:.2f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIGURES, 'sequence_length_distribution.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Saved: {fig_path}")

# ----------------------
# FIGURE 3: Length Distribution by Response
# ----------------------

print("\n3. Creating 'Length Distribution by Response' faceted histogram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Responders
responder_lengths = df[df['response_binary'] == 1]['cdr3_length']
axes[0].hist(responder_lengths, bins=range(5, 30),
             color='#2ecc71', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('CDR3 Length (amino acids)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Number of Sequences', fontsize=11, fontweight='bold')
axes[0].set_title('Responders\n(Response to Immunotherapy)',
                  fontsize=12, fontweight='bold', color='#27ae60')
axes[0].axvline(responder_lengths.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {responder_lengths.mean():.1f}')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3)

# Non-responders
nonresponder_lengths = df[df['response_binary'] == 0]['cdr3_length']
axes[1].hist(nonresponder_lengths, bins=range(5, 30),
             color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('CDR3 Length (amino acids)', fontsize=11, fontweight='bold')
axes[1].set_title('Non-Responders\n(No Response to Immunotherapy)',
                  fontsize=12, fontweight='bold', color='#c0392b')
axes[1].axvline(nonresponder_lengths.mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean: {nonresponder_lengths.mean():.1f}')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('CDR3 Length Distribution by Treatment Response\nComparing Immunotherapy Responders vs Non-Responders',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
fig_path = os.path.join(FIGURES, 'length_hist_by_response.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Saved: {fig_path}")

# ==============================================================================
# SECTION 4: RESPONSE LABEL ANALYSIS (Notebook Cells 23-26)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: RESPONSE LABEL ANALYSIS")
print("-" * 80)

# Sequence-level response distribution
print("\n1. SEQUENCE-LEVEL RESPONSE DISTRIBUTION:")
print("-" * 80)
seq_response = df['response_binary'].value_counts()
for label in [1, 0]:
    label_name = "Responder" if label == 1 else "Non-responder"
    count = seq_response[label]
    percentage = (count / len(df)) * 100
    print(f"   {label_name:15s} ({label}): {count:7,} sequences ({percentage:5.2f}%)")

# Patient-level response distribution
print("\n2. PATIENT-LEVEL RESPONSE DISTRIBUTION:")
print("-" * 80)
patient_response = df.groupby('patient_id')['response_binary'].first().value_counts()
for label in [1, 0]:
    label_name = "Responder" if label == 1 else "Non-responder"
    count = patient_response[label]
    percentage = (count / len(patient_response)) * 100
    print(f"   {label_name:15s} ({label}): {count:2d} patients ({percentage:5.2f}%)")

# ----------------------
# FIGURE 4: Response Pie Chart
# ----------------------

print("\n3. Creating 'Response Distribution' pie chart...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#2ecc71', '#e74c3c']
labels = [f'Responders\n({patient_response[1]} patients)',
          f'Non-Responders\n({patient_response[0]} patients)']
sizes = [patient_response[1], patient_response[0]]
explode = (0.05, 0)  # Explode responders slice

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

ax.set_title('Patient Response to Immunotherapy\nBasal Cell Carcinoma Checkpoint Blockade Treatment',
             fontsize=14, fontweight='bold', pad=20)

# Add legend with statistics
legend_text = [
    f'Responders: {patient_response[1]} patients ({patient_response[1]/len(patient_response)*100:.1f}%)',
    f'Non-Responders: {patient_response[0]} patients ({patient_response[0]/len(patient_response)*100:.1f}%)'
]
ax.legend(legend_text, loc='upper left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)

plt.tight_layout()
fig_path = os.path.join(FIGURES, 'response_pie.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Saved: {fig_path}")

# ==============================================================================
# SECTION 5: REPERTOIRE SIZE COMPARISON (Notebook Cell 26)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: REPERTOIRE SIZE ANALYSIS")
print("-" * 80)

# Calculate repertoire statistics per patient
print("\n1. Calculating per-patient repertoire statistics...")

patient_stats = df.groupby('patient_id').agg({
    'aminoAcid': 'count',  # Number of unique sequences
    'cdr3_length': ['mean', 'median'],
    'response_binary': 'first'
}).reset_index()

patient_stats.columns = ['patient_id', 'unique_sequences', 'avg_len', 'median_len', 'response_binary']

print(f"   ✅ Statistics calculated for {len(patient_stats)} patients")

# Compare repertoire sizes between responders and non-responders
responder_sizes = patient_stats[patient_stats['response_binary'] == 1]['unique_sequences']
nonresponder_sizes = patient_stats[patient_stats['response_binary'] == 0]['unique_sequences']

print("\n2. REPERTOIRE SIZE COMPARISON:")
print("-" * 80)
print(f"   Responders:")
print(f"   - Mean:   {responder_sizes.mean():,.0f} sequences")
print(f"   - Median: {responder_sizes.median():,.0f} sequences")
print(f"   - Std:    {responder_sizes.std():,.0f}")
print(f"\n   Non-Responders:")
print(f"   - Mean:   {nonresponder_sizes.mean():,.0f} sequences")
print(f"   - Median: {nonresponder_sizes.median():,.0f} sequences")
print(f"   - Std:    {nonresponder_sizes.std():,.0f}")

size_diff = responder_sizes.mean() - nonresponder_sizes.mean()
print(f"\n   Difference (Responder - Non-Responder): {size_diff:+,.0f} sequences")

# ----------------------
# FIGURE 5: Repertoire Size Box Plot
# ----------------------

print("\n3. Creating 'Repertoire Size by Response' box plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data for box plot
plot_data = []
plot_labels = []
for response_val, label_name, color in [(1, 'Responder', '#2ecc71'), (0, 'Non-Responder', '#e74c3c')]:
    sizes = patient_stats[patient_stats['response_binary'] == response_val]['unique_sequences']
    plot_data.append(sizes)
    plot_labels.append(f'{label_name}\n(n={len(sizes)} patients)')

# Create box plot
bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True,
                widths=0.6, notch=True, showmeans=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='D', markerfacecolor='yellow',
                              markeredgecolor='black', markersize=8),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Color boxes
for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual data points
for i, (data, label) in enumerate(zip(plot_data, plot_labels), 1):
    y = data
    x = np.random.normal(i, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.4, s=100, edgecolors='black', linewidths=0.5)

ax.set_ylabel('Repertoire Size (Unique TCR Sequences)', fontsize=12, fontweight='bold')
ax.set_xlabel('Response Group', fontsize=12, fontweight='bold')
ax.set_title('TCR Repertoire Diversity by Treatment Response\nComparison of Unique Sequence Counts',
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='_', color='red', linewidth=2, markersize=10, label='Median'),
    plt.Line2D([0], [0], marker='D', color='yellow', markeredgecolor='black',
               markersize=8, linestyle='None', label='Mean')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Add statistics text
stats_text = (f'Responders: Mean = {responder_sizes.mean():.0f}\n'
              f'Non-Responders: Mean = {nonresponder_sizes.mean():.0f}\n'
              f'Difference: {size_diff:+.0f}')
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIGURES, 'repertoire_size_by_response.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Saved: {fig_path}")

# ==============================================================================
# SECTION 6: SAVE PATIENT SUMMARY
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: SAVING PATIENT SUMMARY")
print("-" * 80)

print(f"\nSaving patient summary to: {PATIENT_SUMMARY_FILE}")

patient_stats.to_csv(PATIENT_SUMMARY_FILE, index=False)

print(f"✅ Saved! ({len(patient_stats)} patients)")

print("\nPatient summary preview:")
print("-" * 80)
print(patient_stats.head(10).to_string())

# ==============================================================================
# SECTION 7: SAVE LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: SAVING ANALYSIS LOG")
print("-" * 80)

try:
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total sequences: {len(df):,}\n")
        f.write(f"Total patients: {df['patient_id'].nunique()}\n")
        f.write(f"Unique V genes: {df['vGeneName'].nunique()}\n")
        f.write(f"Unique J genes: {df['jGeneName'].nunique()}\n\n")

        f.write("-" * 80 + "\n")
        f.write("SEQUENCE LENGTH STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {length_stats['mean']:.2f} aa\n")
        f.write(f"Median: {length_stats['50%']:.0f} aa\n")
        f.write(f"Std: {length_stats['std']:.2f}\n")
        f.write(f"Min: {length_stats['min']:.0f} aa\n")
        f.write(f"Max: {length_stats['max']:.0f} aa\n\n")

        f.write("-" * 80 + "\n")
        f.write("RESPONSE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Responders: {patient_response[1]} patients ({patient_response[1]/len(patient_response)*100:.1f}%)\n")
        f.write(f"Non-Responders: {patient_response[0]} patients ({patient_response[0]/len(patient_response)*100:.1f}%)\n\n")

        f.write("-" * 80 + "\n")
        f.write("REPERTOIRE SIZE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Responders - Mean: {responder_sizes.mean():,.0f}, Median: {responder_sizes.median():,.0f}\n")
        f.write(f"Non-Responders - Mean: {nonresponder_sizes.mean():,.0f}, Median: {nonresponder_sizes.median():,.0f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FIGURES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write("1. top10_v_genes.png\n")
        f.write("2. sequence_length_distribution.png\n")
        f.write("3. length_hist_by_response.png\n")
        f.write("4. response_pie.png\n")
        f.write("5. repertoire_size_by_response.png\n\n")

        f.write("-" * 80 + "\n")
        f.write("FILES CREATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"Patient summary: {PATIENT_SUMMARY_FILE}\n")

    print(f"\n✅ Log saved to: {LOG_FILE}")

except Exception as e:
    print(f"\n⚠️  Could not save log file: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("="*80)

print("\n📋 SUMMARY:")
print(f"   ✅ Analyzed {len(df):,} TCR sequences from {df['patient_id'].nunique()} patients")
print(f"   ✅ Generated 5 publication-quality figures")
print(f"   ✅ Created patient summary table")
print(f"   ✅ All outputs saved to figures/ and results/")

print("\n📊 KEY FINDINGS:")
print(f"   • Most common V gene: {v_gene_counts.index[0]} ({v_gene_counts.values[0]:,} sequences)")
print(f"   • Average CDR3 length: {length_stats['mean']:.1f} amino acids")
print(f"   • Responders: {patient_response[1]} patients ({patient_response[1]/len(patient_response)*100:.1f}%)")
print(f"   • Average repertoire size: {patient_stats['unique_sequences'].mean():,.0f} sequences/patient")

print("\n📊 NEXT STEPS:")
print("   → Run Script 04: Feature encoding (one-hot encoding)")
print("   → Command: python scripts/04_feature_encoding.py")

print("\n" + "="*80)
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
