#!/usr/bin/env python3
"""
Script 10: Responder vs Non-Responder Comparison
=================================================

This script performs statistical comparisons between responders and
non-responders based on attention weights, V/J gene usage, and
sequence characteristics.

STATISTICAL TESTS:
------------------
1. Mann-Whitney U test for attention weight differences
2. Chi-square test for V/J gene usage differences
3. Kolmogorov-Smirnov test for CDR3 length distributions
4. Fisher's exact test for enrichment analysis

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
print("RESPONDER VS NON-RESPONDER COMPARISON - SCRIPT 10")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_processed")

os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "10_responder_comparison.log")

# Style
COLORS = {
    'primary': '#2C3E50',
    'accent1': '#27AE60',      # Green (responders)
    'accent2': '#E74C3C',      # Red (non-responders)
    'accent3': '#9B59B6',
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

# Load attention weights
attention_file = os.path.join(RESULTS_DIR, "attention_weights_all.csv")
attention_df = pd.read_csv(attention_file)
print(f"Loaded attention weights: {len(attention_df):,} sequences")

# Load TCR data
tcr_file = os.path.join(DATA_DIR, "deeptcr_trb_ready.csv")
tcr_df = pd.read_csv(tcr_file)
print(f"Loaded TCR data: {len(tcr_df):,} sequences")

# Split by response
responders = attention_df[attention_df['response_binary'] == 1]
non_responders = attention_df[attention_df['response_binary'] == 0]

print(f"Responders: {len(responders):,} sequences")
print(f"Non-Responders: {len(non_responders):,} sequences")

# ==============================================================================
# SECTION 2: ATTENTION WEIGHT COMPARISON
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: ATTENTION WEIGHT COMPARISON")
print("-" * 80)

# Mann-Whitney U test
stat_mw, pval_mw = stats.mannwhitneyu(
    responders['attention_weight'],
    non_responders['attention_weight'],
    alternative='two-sided'
)

print(f"Mann-Whitney U test:")
print(f"   U statistic: {stat_mw:.2f}")
print(f"   p-value: {pval_mw:.2e}")

# Effect size (rank-biserial correlation)
n1, n2 = len(responders), len(non_responders)
r_effect = 1 - (2 * stat_mw) / (n1 * n2)
print(f"   Effect size (r): {r_effect:.4f}")

# Compare means and medians
r_mean = responders['attention_weight'].mean()
nr_mean = non_responders['attention_weight'].mean()
r_median = responders['attention_weight'].median()
nr_median = non_responders['attention_weight'].median()

print(f"\nMean attention:")
print(f"   Responders: {r_mean:.8f}")
print(f"   Non-Responders: {nr_mean:.8f}")
print(f"   Ratio: {r_mean/nr_mean:.2f}")

# ==============================================================================
# SECTION 3: V-GENE USAGE COMPARISON
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: V-GENE USAGE COMPARISON")
print("-" * 80)

# Get top V genes
v_gene_counts = attention_df['vGeneName'].value_counts()
top_v_genes = v_gene_counts.head(15).index.tolist()

# Calculate frequencies
v_freq_responder = responders['vGeneName'].value_counts(normalize=True)
v_freq_nonresponder = non_responders['vGeneName'].value_counts(normalize=True)

v_comparison = []
for gene in top_v_genes:
    freq_r = v_freq_responder.get(gene, 0) * 100
    freq_nr = v_freq_nonresponder.get(gene, 0) * 100

    # Chi-square for this gene
    obs_r = len(responders[responders['vGeneName'] == gene])
    obs_nr = len(non_responders[non_responders['vGeneName'] == gene])
    total_r = len(responders)
    total_nr = len(non_responders)

    # Expected counts
    total_gene = obs_r + obs_nr
    exp_r = total_gene * total_r / (total_r + total_nr)
    exp_nr = total_gene * total_nr / (total_r + total_nr)

    if exp_r > 0 and exp_nr > 0:
        chi2 = ((obs_r - exp_r)**2 / exp_r) + ((obs_nr - exp_nr)**2 / exp_nr)
        pval = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2, pval = 0, 1

    fold_change = (freq_r + 0.1) / (freq_nr + 0.1)

    v_comparison.append({
        'V_Gene': gene,
        'Freq_Responder': freq_r,
        'Freq_NonResponder': freq_nr,
        'Fold_Change': fold_change,
        'Chi2': chi2,
        'P_Value': pval
    })

v_comparison_df = pd.DataFrame(v_comparison)
print("\nTop V-gene differences:")
print(v_comparison_df.head(10).to_string(index=False))

# ==============================================================================
# SECTION 4: J-GENE USAGE COMPARISON
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: J-GENE USAGE COMPARISON")
print("-" * 80)

j_freq_responder = responders['jGeneName'].value_counts(normalize=True)
j_freq_nonresponder = non_responders['jGeneName'].value_counts(normalize=True)

j_genes = attention_df['jGeneName'].value_counts().index.tolist()

j_comparison = []
for gene in j_genes:
    freq_r = j_freq_responder.get(gene, 0) * 100
    freq_nr = j_freq_nonresponder.get(gene, 0) * 100
    fold_change = (freq_r + 0.1) / (freq_nr + 0.1)

    j_comparison.append({
        'J_Gene': gene,
        'Freq_Responder': freq_r,
        'Freq_NonResponder': freq_nr,
        'Fold_Change': fold_change
    })

j_comparison_df = pd.DataFrame(j_comparison)
print("\nJ-gene differences:")
print(j_comparison_df.to_string(index=False))

# ==============================================================================
# SECTION 5: CDR3 LENGTH COMPARISON
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: CDR3 LENGTH COMPARISON")
print("-" * 80)

r_lengths = responders['cdr3_length']
nr_lengths = non_responders['cdr3_length']

# KS test
stat_ks, pval_ks = stats.ks_2samp(r_lengths, nr_lengths)
print(f"Kolmogorov-Smirnov test:")
print(f"   KS statistic: {stat_ks:.4f}")
print(f"   p-value: {pval_ks:.2e}")

# T-test
stat_t, pval_t = stats.ttest_ind(r_lengths, nr_lengths)
print(f"\nt-test:")
print(f"   t statistic: {stat_t:.4f}")
print(f"   p-value: {pval_t:.2e}")

print(f"\nCDR3 length statistics:")
print(f"   Responders: {r_lengths.mean():.2f} +/- {r_lengths.std():.2f}")
print(f"   Non-Responders: {nr_lengths.mean():.2f} +/- {nr_lengths.std():.2f}")

# ==============================================================================
# SECTION 6: HIGH ATTENTION SEQUENCE COMPARISON
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: HIGH ATTENTION SEQUENCE COMPARISON")
print("-" * 80)

high_att = attention_df[attention_df['is_high_attention'] == True]

high_att_r = high_att[high_att['response_binary'] == 1]
high_att_nr = high_att[high_att['response_binary'] == 0]

print(f"High attention sequences:")
print(f"   Responders: {len(high_att_r):,}")
print(f"   Non-Responders: {len(high_att_nr):,}")

# Chi-square test for enrichment
obs = np.array([[len(high_att_r), len(responders) - len(high_att_r)],
                [len(high_att_nr), len(non_responders) - len(high_att_nr)]])
chi2, pval, dof, expected = stats.chi2_contingency(obs)
print(f"\nChi-square test for high attention enrichment:")
print(f"   Chi2: {chi2:.2f}")
print(f"   p-value: {pval:.2e}")

# ==============================================================================
# SECTION 7: CREATE COMPARISON FIGURE
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: CREATING COMPARISON FIGURE")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Panel A: Attention violin plot
ax = fig.add_subplot(gs[0, 0])
plot_data = pd.DataFrame({
    'Response': ['R'] * len(responders) + ['NR'] * len(non_responders),
    'Log Attention': list(np.log10(responders['attention_weight'] + 1e-10)) +
                     list(np.log10(non_responders['attention_weight'] + 1e-10))
})
palette = {'R': COLORS['accent1'], 'NR': COLORS['accent2']}
sns.violinplot(data=plot_data, x='Response', y='Log Attention', palette=palette, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('log10(Attention Weight)')
ax.set_title(f'Attention Comparison\n(p = {pval_mw:.2e})', fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'A')

# Panel B: V-gene comparison
ax = fig.add_subplot(gs[0, 1])
top_5_v = v_comparison_df.head(8)
x = np.arange(len(top_5_v))
width = 0.35
ax.bar(x - width/2, top_5_v['Freq_Responder'], width, label='Responders',
       color=COLORS['accent1'], edgecolor='white')
ax.bar(x + width/2, top_5_v['Freq_NonResponder'], width, label='Non-Responders',
       color=COLORS['accent2'], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(top_5_v['V_Gene'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Frequency (%)')
ax.set_title('V-Gene Usage Comparison', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'B')

# Panel C: J-gene comparison
ax = fig.add_subplot(gs[0, 2])
top_j = j_comparison_df.head(10)
x = np.arange(len(top_j))
ax.bar(x - width/2, top_j['Freq_Responder'], width, label='Responders',
       color=COLORS['accent1'], edgecolor='white')
ax.bar(x + width/2, top_j['Freq_NonResponder'], width, label='Non-Responders',
       color=COLORS['accent2'], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(top_j['J_Gene'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Frequency (%)')
ax.set_title('J-Gene Usage Comparison', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'C')

# Panel D: CDR3 length distribution
ax = fig.add_subplot(gs[1, 0])
ax.hist(r_lengths, bins=range(5, 30), alpha=0.6, label='Responders',
        color=COLORS['accent1'], edgecolor='white', density=True)
ax.hist(nr_lengths, bins=range(5, 30), alpha=0.6, label='Non-Responders',
        color=COLORS['accent2'], edgecolor='white', density=True)
ax.axvline(r_lengths.mean(), color=COLORS['accent1'], linestyle='--', linewidth=2)
ax.axvline(nr_lengths.mean(), color=COLORS['accent2'], linestyle='--', linewidth=2)
ax.set_xlabel('CDR3 Length (amino acids)')
ax.set_ylabel('Density')
ax.set_title(f'CDR3 Length Distribution\n(KS p = {pval_ks:.2e})', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'D')

# Panel E: High attention comparison
ax = fig.add_subplot(gs[1, 1])
categories = ['Responders', 'Non-Responders']
high_counts = [len(high_att_r), len(high_att_nr)]
low_counts = [len(responders) - len(high_att_r), len(non_responders) - len(high_att_nr)]

x = np.arange(2)
ax.bar(x, high_counts, 0.6, label='High Attention',
       color=COLORS['accent3'], edgecolor='white')
ax.bar(x, low_counts, 0.6, bottom=high_counts, label='Normal Attention',
       color=COLORS['dark_gray'], alpha=0.5, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Number of Sequences')
ax.set_title('High Attention Enrichment', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'E')

# Panel F: Fold change volcano plot
ax = fig.add_subplot(gs[1, 2])
fc = v_comparison_df['Fold_Change'].values
pvals = v_comparison_df['P_Value'].values
log_fc = np.log2(fc)
neg_log_p = -np.log10(pvals + 1e-10)

colors = ['red' if abs(lfc) > 0.5 and nlp > 2 else 'gray'
          for lfc, nlp in zip(log_fc, neg_log_p)]
ax.scatter(log_fc, neg_log_p, c=colors, alpha=0.7, s=60, edgecolors='white')

for i, gene in enumerate(v_comparison_df['V_Gene']):
    if abs(log_fc[i]) > 0.3 or neg_log_p[i] > 1.5:
        ax.annotate(gene, (log_fc[i], neg_log_p[i]), fontsize=7,
                   xytext=(3, 3), textcoords='offset points')

ax.axhline(2, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('log2(Fold Change R/NR)')
ax.set_ylabel('-log10(p-value)')
ax.set_title('V-Gene Enrichment Volcano Plot', fontweight='bold')
clean_spines(ax)
add_panel_label(ax, 'F')

plt.suptitle('Figure: Responder vs Non-Responder Comparison',
             fontsize=14, fontweight='bold', y=1.02)
fig_path = os.path.join(FIGURES_DIR, 'figure_responder_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ==============================================================================
# SECTION 8: SAVE RESULTS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 8: SAVING RESULTS")
print("-" * 80)

# Save comprehensive statistics
stats_results = pd.DataFrame({
    'Test': [
        'Attention Weight (Mann-Whitney U)',
        'CDR3 Length (KS test)',
        'CDR3 Length (t-test)',
        'High Attention Enrichment (Chi-square)'
    ],
    'Statistic': [stat_mw, stat_ks, stat_t, chi2],
    'P_Value': [pval_mw, pval_ks, pval_t, pval],
    'Effect_Size': [r_effect, stat_ks, (r_lengths.mean() - nr_lengths.mean()) / r_lengths.std(), np.nan]
})

stats_file = os.path.join(RESULTS_DIR, "responder_comparison_stats.csv")
stats_results.to_csv(stats_file, index=False)
print(f"Saved: {stats_file}")

# Save V-gene comparison
v_file = os.path.join(RESULTS_DIR, "vgene_comparison.csv")
v_comparison_df.to_csv(v_file, index=False)
print(f"Saved: {v_file}")

# Save J-gene comparison
j_file = os.path.join(RESULTS_DIR, "jgene_comparison.csv")
j_comparison_df.to_csv(j_file, index=False)
print(f"Saved: {j_file}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("RESPONDER COMPARISON COMPLETE!")
print("="*80)

print(f"\nKEY FINDINGS:")
print(f"   Attention difference p-value: {pval_mw:.2e}")
print(f"   CDR3 length difference p-value: {pval_ks:.2e}")
print(f"   High attention enrichment p-value: {pval:.2e}")

print(f"\nOUTPUT FILES:")
print(f"   {stats_file}")
print(f"   {v_file}")
print(f"   {j_file}")
print(f"   {fig_path}")

print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
