#!/usr/bin/env python3
"""
Script 11: Top Predictive Sequences Analysis
=============================================

This script identifies and analyzes the top predictive TCR sequences
based on attention weights from the trained DeepTCR model.

ANALYSIS:
---------
1. Rank sequences by attention weight
2. Extract top 100 highest-attention sequences
3. Analyze V/J gene enrichment in top sequences
4. Calculate fold enrichment vs background
5. Create publication-quality visualizations

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
print("TOP PREDICTIVE SEQUENCES ANALYSIS - SCRIPT 11")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "11_top_sequences.log")

# Style
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent1': '#27AE60',
    'accent2': '#E74C3C',
    'accent3': '#9B59B6',
    'accent4': '#F39C12',
    'dark_gray': '#7F8C8D',
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left', color=COLORS['primary'])

def clean_spines(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# ==============================================================================
# SECTION 1: LOAD DATA
# ==============================================================================

print("-" * 80)
print("SECTION 1: LOADING DATA")
print("-" * 80)

attention_file = os.path.join(RESULTS_DIR, "attention_weights_all.csv")
attention_df = pd.read_csv(attention_file)
print(f"Loaded attention weights: {len(attention_df):,} sequences")

# ==============================================================================
# SECTION 2: IDENTIFY TOP SEQUENCES
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: IDENTIFYING TOP SEQUENCES")
print("-" * 80)

# Sort by attention weight
attention_df_sorted = attention_df.sort_values('attention_weight', ascending=False)

# Top 100 sequences
top_100 = attention_df_sorted.head(100).copy()
top_100['rank'] = range(1, 101)

print(f"\nTop 10 highest-attention sequences:")
print("-" * 60)
for i, row in top_100.head(10).iterrows():
    print(f"{row['rank']:2d}. {row['aminoAcid'][:25]:25s} V={row['vGeneName']:12s} "
          f"att={row['attention_weight']:.6f}")

# Top 1000 for enrichment analysis
top_1000 = attention_df_sorted.head(1000).copy()

# ==============================================================================
# SECTION 3: V-GENE ENRICHMENT ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: V-GENE ENRICHMENT ANALYSIS")
print("-" * 80)

# Background V-gene frequencies
bg_v_freq = attention_df['vGeneName'].value_counts(normalize=True)

# Top sequences V-gene frequencies
top_v_freq = top_1000['vGeneName'].value_counts(normalize=True)

# Calculate enrichment
v_enrichment = []
for gene in bg_v_freq.index:
    bg_freq = bg_v_freq.get(gene, 0)
    top_freq = top_v_freq.get(gene, 0)

    if bg_freq > 0:
        fold_enrichment = top_freq / bg_freq
    else:
        fold_enrichment = 0

    # Fisher's exact test
    top_count = int(top_freq * len(top_1000))
    top_other = len(top_1000) - top_count
    bg_count = int(bg_freq * len(attention_df))
    bg_other = len(attention_df) - bg_count

    try:
        odds_ratio, pval = stats.fisher_exact([[top_count, top_other],
                                               [bg_count, bg_other]])
    except:
        odds_ratio, pval = 1.0, 1.0

    v_enrichment.append({
        'V_Gene': gene,
        'Background_Freq': bg_freq * 100,
        'Top_Freq': top_freq * 100,
        'Fold_Enrichment': fold_enrichment,
        'Odds_Ratio': odds_ratio,
        'P_Value': pval,
        '-log10(p)': -np.log10(pval + 1e-10)
    })

v_enrichment_df = pd.DataFrame(v_enrichment).sort_values('Fold_Enrichment', ascending=False)

print("\nTop enriched V-genes in high-attention sequences:")
print(v_enrichment_df.head(10).to_string(index=False))

# ==============================================================================
# SECTION 4: J-GENE ENRICHMENT ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: J-GENE ENRICHMENT ANALYSIS")
print("-" * 80)

bg_j_freq = attention_df['jGeneName'].value_counts(normalize=True)
top_j_freq = top_1000['jGeneName'].value_counts(normalize=True)

j_enrichment = []
for gene in bg_j_freq.index:
    bg_freq = bg_j_freq.get(gene, 0)
    top_freq = top_j_freq.get(gene, 0)

    if bg_freq > 0:
        fold_enrichment = top_freq / bg_freq
    else:
        fold_enrichment = 0

    j_enrichment.append({
        'J_Gene': gene,
        'Background_Freq': bg_freq * 100,
        'Top_Freq': top_freq * 100,
        'Fold_Enrichment': fold_enrichment
    })

j_enrichment_df = pd.DataFrame(j_enrichment).sort_values('Fold_Enrichment', ascending=False)

print("\nTop enriched J-genes in high-attention sequences:")
print(j_enrichment_df.head(10).to_string(index=False))

# ==============================================================================
# SECTION 5: SEQUENCE CHARACTERISTICS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: TOP SEQUENCE CHARACTERISTICS")
print("-" * 80)

# CDR3 length distribution
top_lengths = top_100['cdr3_length']
all_lengths = attention_df['cdr3_length']

print(f"\nCDR3 Length Comparison:")
print(f"   Top 100: {top_lengths.mean():.2f} +/- {top_lengths.std():.2f}")
print(f"   All: {all_lengths.mean():.2f} +/- {all_lengths.std():.2f}")

# Response distribution in top sequences
top_responders = top_100[top_100['response_binary'] == 1]
top_nonresponders = top_100[top_100['response_binary'] == 0]

print(f"\nResponse distribution in top 100:")
print(f"   Responders: {len(top_responders)} ({len(top_responders)}%)")
print(f"   Non-Responders: {len(top_nonresponders)} ({len(top_nonresponders)}%)")

# ==============================================================================
# SECTION 6: CREATE VISUALIZATIONS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: CREATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Panel A: Top 20 sequences bar chart
ax = fig.add_subplot(gs[0, 0])
top_20 = top_100.head(20)
y_pos = np.arange(len(top_20))
colors = [COLORS['accent1'] if r == 1 else COLORS['accent2']
          for r in top_20['response_binary']]
bars = ax.barh(y_pos, top_20['attention_weight'], color=colors,
               edgecolor='white', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{s[:15]}..." for s in top_20['aminoAcid']], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Attention Weight')
ax.set_title('Top 20 High-Attention Sequences', fontweight='bold')

legend_elements = [mpatches.Patch(facecolor=COLORS['accent1'], label='Responder'),
                   mpatches.Patch(facecolor=COLORS['accent2'], label='Non-Responder')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'A')

# Panel B: V-gene enrichment
ax = fig.add_subplot(gs[0, 1])
top_v = v_enrichment_df.head(10)
colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(top_v)))[::-1]
ax.barh(range(len(top_v)), top_v['Fold_Enrichment'], color=colors,
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top_v)))
ax.set_yticklabels(top_v['V_Gene'], fontsize=9)
ax.invert_yaxis()
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Fold Enrichment')
ax.set_title('V-Gene Enrichment in Top Sequences', fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'B')

# Panel C: J-gene enrichment
ax = fig.add_subplot(gs[0, 2])
top_j = j_enrichment_df.head(10)
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_j)))[::-1]
ax.barh(range(len(top_j)), top_j['Fold_Enrichment'], color=colors,
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top_j)))
ax.set_yticklabels(top_j['J_Gene'], fontsize=9)
ax.invert_yaxis()
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Fold Enrichment')
ax.set_title('J-Gene Enrichment in Top Sequences', fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'C')

# Panel D: CDR3 length comparison
ax = fig.add_subplot(gs[1, 0])
ax.hist(all_lengths, bins=range(5, 30), alpha=0.5, label='All Sequences',
        color=COLORS['dark_gray'], edgecolor='white', density=True)
ax.hist(top_lengths, bins=range(5, 30), alpha=0.7, label='Top 100',
        color=COLORS['accent3'], edgecolor='white', density=True)
ax.axvline(all_lengths.mean(), color=COLORS['dark_gray'], linestyle='--', linewidth=2)
ax.axvline(top_lengths.mean(), color=COLORS['accent3'], linestyle='--', linewidth=2)
ax.set_xlabel('CDR3 Length (amino acids)')
ax.set_ylabel('Density')
ax.set_title('CDR3 Length: Top vs All', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'D')

# Panel E: Attention rank distribution
ax = fig.add_subplot(gs[1, 1])
ax.semilogy(range(1, len(attention_df_sorted) + 1),
            attention_df_sorted['attention_weight'].values,
            color=COLORS['secondary'], linewidth=1)
ax.axhline(top_100['attention_weight'].min(), color=COLORS['accent2'],
           linestyle='--', label='Top 100 threshold')
ax.fill_between(range(1, 101),
                [attention_df_sorted['attention_weight'].max()] * 100,
                [attention_df_sorted['attention_weight'].iloc[99]] * 100,
                alpha=0.3, color=COLORS['accent2'])
ax.set_xlabel('Sequence Rank')
ax.set_ylabel('Attention Weight (log scale)')
ax.set_title('Attention Weight Distribution', fontweight='bold')
ax.set_xlim([1, 10000])
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'E')

# Panel F: Response pie chart for top sequences
ax = fig.add_subplot(gs[1, 2])
sizes = [len(top_responders), len(top_nonresponders)]
colors_pie = [COLORS['accent1'], COLORS['accent2']]
labels = ['Responders', 'Non-Responders']
wedges, texts, autotexts = ax.pie(sizes, colors=colors_pie, autopct='%1.1f%%',
                                   startangle=90, pctdistance=0.75,
                                   wedgeprops=dict(width=0.5, edgecolor='white'))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Response Distribution\nin Top 100 Sequences', fontweight='bold')
ax.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
add_panel_label(ax, 'F')

plt.suptitle('Figure: Top Predictive Sequences Analysis',
             fontsize=14, fontweight='bold', y=1.02)
fig_path = os.path.join(FIGURES_DIR, 'figure_top_sequences.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 7: SAVE RESULTS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: SAVING RESULTS")
print("-" * 80)

# Save top 100 sequences with details
top_100_file = os.path.join(RESULTS_DIR, "top_100_sequences_detailed.csv")
top_100.to_csv(top_100_file, index=False)
print(f"Saved: {top_100_file}")

# Save V-gene enrichment
v_enrichment_file = os.path.join(RESULTS_DIR, "vgene_enrichment.csv")
v_enrichment_df.to_csv(v_enrichment_file, index=False)
print(f"Saved: {v_enrichment_file}")

# Save J-gene enrichment
j_enrichment_file = os.path.join(RESULTS_DIR, "jgene_enrichment.csv")
j_enrichment_df.to_csv(j_enrichment_file, index=False)
print(f"Saved: {j_enrichment_file}")

# Save combined enrichment summary
enrichment_summary = pd.DataFrame({
    'Analysis': ['V-Gene', 'J-Gene'],
    'Top_Enriched': [v_enrichment_df.iloc[0]['V_Gene'],
                     j_enrichment_df.iloc[0]['J_Gene']],
    'Max_Fold_Enrichment': [v_enrichment_df.iloc[0]['Fold_Enrichment'],
                            j_enrichment_df.iloc[0]['Fold_Enrichment']]
})
summary_file = os.path.join(RESULTS_DIR, "enrichment_summary.csv")
enrichment_summary.to_csv(summary_file, index=False)
print(f"Saved: {summary_file}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("TOP PREDICTIVE SEQUENCES ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKEY FINDINGS:")
print(f"   Top V-gene: {v_enrichment_df.iloc[0]['V_Gene']} "
      f"(enrichment: {v_enrichment_df.iloc[0]['Fold_Enrichment']:.2f}x)")
print(f"   Top J-gene: {j_enrichment_df.iloc[0]['J_Gene']} "
      f"(enrichment: {j_enrichment_df.iloc[0]['Fold_Enrichment']:.2f}x)")
print(f"   Top 100 from Responders: {len(top_responders)}%")

print(f"\nOUTPUT FILES:")
print(f"   {top_100_file}")
print(f"   {v_enrichment_file}")
print(f"   {j_enrichment_file}")
print(f"   {fig_path}")

print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
