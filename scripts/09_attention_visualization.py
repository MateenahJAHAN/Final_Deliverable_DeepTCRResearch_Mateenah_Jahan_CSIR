#!/usr/bin/env python3
"""
Script 09: Attention Weight Visualization
==========================================

This script creates publication-quality visualizations of attention weights
extracted from the trained DeepTCR models.

VISUALIZATIONS:
--------------
1. Attention distribution histogram (log scale)
2. Attention by response group comparison
3. Per-patient attention heatmap
4. Attention vs CDR3 length scatter

Author: Post-training analysis pipeline
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

print("="*80)
print("ATTENTION WEIGHT VISUALIZATION - SCRIPT 09")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "09_attention_visualization.log")

# ==============================================================================
# STYLE CONFIGURATION
# ==============================================================================

# Nature/Cell style colors
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent1': '#27AE60',      # Green (responders)
    'accent2': '#E74C3C',      # Red (non-responders)
    'accent3': '#9B59B6',      # Purple
    'accent4': '#F39C12',      # Orange
    'light_gray': '#ECF0F1',
    'medium_gray': '#BDC3C7',
    'dark_gray': '#7F8C8D',
}

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': COLORS['dark_gray'],
    'axes.linewidth': 0.8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label (A, B, C, etc.) to subplot"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left', color=COLORS['primary'])

def clean_spines(ax, keep=['bottom', 'left']):
    """Remove unnecessary spines"""
    for spine in ax.spines:
        if spine not in keep:
            ax.spines[spine].set_visible(False)
        else:
            ax.spines[spine].set_color(COLORS['dark_gray'])
            ax.spines[spine].set_linewidth(0.8)

# ==============================================================================
# SECTION 1: LOAD DATA
# ==============================================================================

print("-" * 80)
print("SECTION 1: LOADING DATA")
print("-" * 80)

# Load attention weights
attention_file = os.path.join(RESULTS_DIR, "attention_weights_all.csv")
if os.path.exists(attention_file):
    attention_df = pd.read_csv(attention_file)
    print(f"Loaded attention weights: {len(attention_df):,} sequences")
else:
    print(f"Attention file not found: {attention_file}")
    print("Please run script 08 first.")
    sys.exit(1)

# Load patient stats
patient_stats_file = os.path.join(RESULTS_DIR, "patient_attention_stats.csv")
if os.path.exists(patient_stats_file):
    patient_stats = pd.read_csv(patient_stats_file)
    print(f"Loaded patient statistics: {len(patient_stats)} patients")
else:
    patient_stats = None
    print("Patient statistics not found")

# ==============================================================================
# SECTION 2: FIGURE - ATTENTION DISTRIBUTION
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: ATTENTION DISTRIBUTION FIGURE")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Log-scale histogram
ax = axes[0]
log_attention = np.log10(attention_df['attention_weight'] + 1e-10)

n, bins, patches = ax.hist(log_attention, bins=50, color=COLORS['accent3'],
                            edgecolor='white', linewidth=0.5, alpha=0.8)

# Color gradient
norm = plt.Normalize(min(log_attention), max(log_attention))
for bin_val, patch in zip(bins[:-1], patches):
    color = plt.cm.Purples(norm(bin_val) * 0.6 + 0.3)
    patch.set_facecolor(color)

mean_log = np.mean(log_attention)
ax.axvline(mean_log, color=COLORS['accent2'], linestyle='-', linewidth=2.5,
           label=f'Mean: 10^{mean_log:.2f}')

# Threshold for high attention
high_threshold = np.percentile(log_attention, 99)
ax.axvline(high_threshold, color=COLORS['accent4'], linestyle='--', linewidth=2,
           label=f'Top 1%: 10^{high_threshold:.2f}')

ax.set_xlabel('log10(Attention Weight)', fontsize=11)
ax.set_ylabel('Number of Sequences', fontsize=11)
ax.set_title('Attention Weight Distribution', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
clean_spines(ax)
add_panel_label(ax, 'A')

# Panel B: Cumulative distribution
ax = axes[1]
sorted_att = np.sort(attention_df['attention_weight'])
cumulative = np.arange(1, len(sorted_att) + 1) / len(sorted_att)

ax.fill_between(np.log10(sorted_att + 1e-10), 0, cumulative,
                alpha=0.3, color=COLORS['accent3'])
ax.plot(np.log10(sorted_att + 1e-10), cumulative,
        color=COLORS['accent3'], linewidth=2)

# Add percentile markers
for p in [50, 90, 99]:
    val = np.percentile(attention_df['attention_weight'], p)
    ax.axhline(p/100, color=COLORS['dark_gray'], linestyle=':', alpha=0.5)
    ax.text(np.log10(val + 1e-10), p/100 + 0.02, f'{p}th percentile',
            fontsize=8, color=COLORS['dark_gray'])

ax.set_xlabel('log10(Attention Weight)', fontsize=11)
ax.set_ylabel('Cumulative Proportion', fontsize=11)
ax.set_title('Cumulative Attention Distribution', fontsize=12, fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'B')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'figure_attention_distribution.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 3: FIGURE - ATTENTION BY RESPONSE GROUP
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: ATTENTION BY RESPONSE GROUP")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel A: Violin plot comparison
ax = axes[0]

# Prepare data
responder_att = attention_df[attention_df['response_binary'] == 1]['attention_weight']
nonresponder_att = attention_df[attention_df['response_binary'] == 0]['attention_weight']

plot_data = pd.DataFrame({
    'Response': ['Responders'] * len(responder_att) + ['Non-Responders'] * len(nonresponder_att),
    'Attention': list(responder_att) + list(nonresponder_att)
})

# Use log scale for visualization
plot_data['Log Attention'] = np.log10(plot_data['Attention'] + 1e-10)

palette = {'Responders': COLORS['accent1'], 'Non-Responders': COLORS['accent2']}
sns.violinplot(data=plot_data, x='Response', y='Log Attention', palette=palette,
               inner='box', ax=ax, alpha=0.7)

ax.set_xlabel('')
ax.set_ylabel('log10(Attention Weight)', fontsize=11)
ax.set_title('Attention by Response Group', fontsize=12, fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'A')

# Add stats
r_mean = np.mean(responder_att)
nr_mean = np.mean(nonresponder_att)
stat, pval = stats.mannwhitneyu(responder_att, nonresponder_att)
ax.text(0.02, 0.98, f'p = {pval:.2e}', transform=ax.transAxes,
        fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: High attention proportion
ax = axes[1]

high_att_df = attention_df[attention_df['is_high_attention'] == True]

if 'response_label' in high_att_df.columns:
    counts = high_att_df.groupby('response_label').size()
else:
    counts = pd.Series({
        'Responders': len(high_att_df[high_att_df['response_binary'] == 1]),
        'Non-Responders': len(high_att_df[high_att_df['response_binary'] == 0])
    })

colors_bar = [COLORS['accent1'], COLORS['accent2']]
bars = ax.bar(range(len(counts)), counts.values, color=colors_bar,
              edgecolor='white', linewidth=1)
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index, fontsize=10)
ax.set_ylabel('Number of High-Attention Sequences', fontsize=11)
ax.set_title('High Attention (Top 1%) by Group', fontsize=12, fontweight='bold')

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', fontsize=10, color=COLORS['dark_gray'])

clean_spines(ax)
add_panel_label(ax, 'B')

# Panel C: Mean attention per patient
ax = axes[2]

if patient_stats is not None:
    resp_means = patient_stats[patient_stats['response_binary'] == 1]['mean_attention']
    nonresp_means = patient_stats[patient_stats['response_binary'] == 0]['mean_attention']

    ax.scatter(range(len(resp_means)), sorted(resp_means, reverse=True),
               color=COLORS['accent1'], s=80, alpha=0.7, label='Responders',
               edgecolors='white', linewidths=1)
    ax.scatter(range(len(resp_means), len(resp_means) + len(nonresp_means)),
               sorted(nonresp_means, reverse=True),
               color=COLORS['accent2'], s=80, alpha=0.7, label='Non-Responders',
               edgecolors='white', linewidths=1)

    ax.set_xlabel('Patient (ranked by mean attention)', fontsize=11)
    ax.set_ylabel('Mean Attention Weight', fontsize=11)
    ax.set_title('Patient-Level Mean Attention', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

clean_spines(ax)
add_panel_label(ax, 'C')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'figure_attention_by_response.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 4: FIGURE - ATTENTION HEATMAP
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: ATTENTION HEATMAP")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

if patient_stats is not None:
    # Create patient x feature heatmap
    patient_ids = patient_stats['patient_id'].values
    features = ['mean_attention', 'max_attention', 'std_attention', 'n_high_attention']

    # Normalize each feature
    heatmap_data = patient_stats[features].copy()
    for col in features:
        heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / \
                            (heatmap_data[col].max() - heatmap_data[col].min())

    # Sort by response
    sort_idx = patient_stats['response_binary'].argsort()[::-1]
    heatmap_sorted = heatmap_data.iloc[sort_idx].values

    im = ax.imshow(heatmap_sorted, cmap='YlOrRd', aspect='auto')

    ax.set_yticks(range(len(patient_ids)))
    ax.set_yticklabels(patient_stats['patient_id'].iloc[sort_idx], fontsize=8)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(['Mean\nAttention', 'Max\nAttention', 'Std\nAttention',
                        'N High\nAttention'], fontsize=9)

    ax.set_xlabel('Attention Metrics', fontsize=11)
    ax.set_ylabel('Patient ID (sorted by response)', fontsize=11)
    ax.set_title('Patient Attention Profile Heatmap', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Normalized Value', fontsize=10)

    # Add response annotation
    response_labels = patient_stats['response_binary'].iloc[sort_idx].values
    for i, r in enumerate(response_labels):
        color = COLORS['accent1'] if r == 1 else COLORS['accent2']
        ax.add_patch(plt.Rectangle((-0.6, i-0.5), 0.3, 1, color=color, clip_on=False))

else:
    ax.text(0.5, 0.5, 'Patient statistics not available', ha='center', va='center',
            fontsize=14, transform=ax.transAxes)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'figure_attention_heatmap.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 5: FIGURE - ATTENTION VS CDR3 LENGTH
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: ATTENTION VS CDR3 LENGTH")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Scatter plot
ax = axes[0]

# Sample for visualization (too many points)
sample_size = min(5000, len(attention_df))
sample_df = attention_df.sample(n=sample_size, random_state=42)

colors = [COLORS['accent1'] if r == 1 else COLORS['accent2']
          for r in sample_df['response_binary']]

scatter = ax.scatter(sample_df['cdr3_length'],
                     np.log10(sample_df['attention_weight'] + 1e-10),
                     c=colors, alpha=0.3, s=20, edgecolors='none')

ax.set_xlabel('CDR3 Length (amino acids)', fontsize=11)
ax.set_ylabel('log10(Attention Weight)', fontsize=11)
ax.set_title('Attention vs CDR3 Length', fontsize=12, fontweight='bold')

legend_elements = [mpatches.Patch(facecolor=COLORS['accent1'], label='Responder'),
                   mpatches.Patch(facecolor=COLORS['accent2'], label='Non-Responder')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
clean_spines(ax)
add_panel_label(ax, 'A')

# Panel B: Mean attention by length
ax = axes[1]

length_stats = attention_df.groupby('cdr3_length').agg({
    'attention_weight': ['mean', 'std', 'count']
}).reset_index()
length_stats.columns = ['cdr3_length', 'mean_att', 'std_att', 'count']

# Filter to common lengths
length_stats = length_stats[length_stats['count'] >= 100]

ax.bar(length_stats['cdr3_length'], length_stats['mean_att'],
       color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
ax.errorbar(length_stats['cdr3_length'], length_stats['mean_att'],
            yerr=length_stats['std_att'] / np.sqrt(length_stats['count']),
            fmt='none', color=COLORS['dark_gray'], capsize=2)

ax.set_xlabel('CDR3 Length (amino acids)', fontsize=11)
ax.set_ylabel('Mean Attention Weight', fontsize=11)
ax.set_title('Mean Attention by CDR3 Length', fontsize=12, fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'B')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'figure_attention_vs_length.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 6: SAVE LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: SAVING LOG")
print("-" * 80)

with open(LOG_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ATTENTION VISUALIZATION LOG\n")
    f.write("="*80 + "\n")
    f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("-"*80 + "\n")
    f.write("FIGURES GENERATED\n")
    f.write("-"*80 + "\n")
    f.write(f"1. figure_attention_distribution.png/pdf\n")
    f.write(f"2. figure_attention_by_response.png/pdf\n")
    f.write(f"3. figure_attention_heatmap.png/pdf\n")
    f.write(f"4. figure_attention_vs_length.png/pdf\n")

print(f"Log saved to: {LOG_FILE}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("ATTENTION VISUALIZATION COMPLETE!")
print("="*80)

print(f"\nFIGURES GENERATED:")
print(f"   1. figure_attention_distribution.png/pdf")
print(f"   2. figure_attention_by_response.png/pdf")
print(f"   3. figure_attention_heatmap.png/pdf")
print(f"   4. figure_attention_vs_length.png/pdf")

print(f"\nAll figures saved to: {FIGURES_DIR}")

print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
