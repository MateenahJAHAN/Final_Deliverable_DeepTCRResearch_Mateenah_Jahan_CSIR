#!/usr/bin/env python3
"""
Script 12: Sequence Characteristics Analysis
=============================================

This script analyzes the characteristics of predictive TCR sequences,
including amino acid composition, motif analysis, and structural features.

ANALYSIS:
---------
1. Compare CDR3 lengths of high vs low attention sequences
2. Analyze amino acid composition
3. Analyze V/J gene pairing patterns
4. Identify sequence motifs in high-attention sequences
5. Create comprehensive characteristics table

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
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

print("="*80)
print("SEQUENCE CHARACTERISTICS ANALYSIS - SCRIPT 12")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "paper_final")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "12_sequence_characteristics.log")

# Style
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent1': '#27AE60',
    'accent2': '#E74C3C',
    'accent3': '#9B59B6',
    'accent4': '#F39C12',
    'dark_gray': '#7F8C8D',
    'light_gray': '#ECF0F1',
}

# Amino acid properties
AA_PROPERTIES = {
    # Hydrophobic
    'A': 'hydrophobic', 'V': 'hydrophobic', 'L': 'hydrophobic',
    'I': 'hydrophobic', 'M': 'hydrophobic', 'F': 'hydrophobic',
    'W': 'hydrophobic', 'P': 'hydrophobic',
    # Polar
    'S': 'polar', 'T': 'polar', 'N': 'polar', 'Q': 'polar',
    'Y': 'polar', 'C': 'polar',
    # Charged positive
    'K': 'positive', 'R': 'positive', 'H': 'positive',
    # Charged negative
    'D': 'negative', 'E': 'negative',
    # Special
    'G': 'special'
}

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.family': 'sans-serif',
    'figure.dpi': 300,
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

# Define high and low attention groups
threshold_high = attention_df['attention_weight'].quantile(0.99)
threshold_low = attention_df['attention_weight'].quantile(0.50)

high_attention = attention_df[attention_df['attention_weight'] >= threshold_high]
low_attention = attention_df[attention_df['attention_weight'] <= threshold_low]

print(f"High attention sequences (top 1%): {len(high_attention):,}")
print(f"Low attention sequences (bottom 50%): {len(low_attention):,}")

# ==============================================================================
# SECTION 2: CDR3 LENGTH ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: CDR3 LENGTH ANALYSIS")
print("-" * 80)

high_lengths = high_attention['cdr3_length']
low_lengths = low_attention['cdr3_length']
all_lengths = attention_df['cdr3_length']

length_stats = pd.DataFrame({
    'Group': ['High Attention', 'Low Attention', 'All'],
    'Mean': [high_lengths.mean(), low_lengths.mean(), all_lengths.mean()],
    'Std': [high_lengths.std(), low_lengths.std(), all_lengths.std()],
    'Median': [high_lengths.median(), low_lengths.median(), all_lengths.median()],
    'Min': [high_lengths.min(), low_lengths.min(), all_lengths.min()],
    'Max': [high_lengths.max(), low_lengths.max(), all_lengths.max()]
})

print("\nCDR3 Length Statistics:")
print(length_stats.to_string(index=False))

# ==============================================================================
# SECTION 3: AMINO ACID COMPOSITION
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: AMINO ACID COMPOSITION")
print("-" * 80)

def get_aa_composition(sequences):
    """Calculate amino acid frequencies"""
    all_aa = ''.join(sequences)
    counts = Counter(all_aa)
    total = len(all_aa)
    return {aa: counts.get(aa, 0) / total * 100 for aa in AMINO_ACIDS}

high_aa = get_aa_composition(high_attention['aminoAcid'])
low_aa = get_aa_composition(low_attention['aminoAcid'])
all_aa = get_aa_composition(attention_df['aminoAcid'])

aa_comparison = pd.DataFrame({
    'Amino_Acid': AMINO_ACIDS,
    'High_Attention': [high_aa[aa] for aa in AMINO_ACIDS],
    'Low_Attention': [low_aa[aa] for aa in AMINO_ACIDS],
    'All': [all_aa[aa] for aa in AMINO_ACIDS],
    'Property': [AA_PROPERTIES.get(aa, 'unknown') for aa in AMINO_ACIDS]
})

aa_comparison['Enrichment'] = aa_comparison['High_Attention'] / (aa_comparison['Low_Attention'] + 0.01)

print("\nTop enriched amino acids in high-attention sequences:")
print(aa_comparison.nlargest(5, 'Enrichment').to_string(index=False))

# ==============================================================================
# SECTION 4: PROPERTY-BASED ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: AMINO ACID PROPERTY ANALYSIS")
print("-" * 80)

def get_property_composition(sequences):
    """Calculate amino acid property frequencies"""
    all_aa = ''.join(sequences)
    properties = [AA_PROPERTIES.get(aa, 'unknown') for aa in all_aa]
    counts = Counter(properties)
    total = len(properties)
    return {prop: counts.get(prop, 0) / total * 100 for prop in
            ['hydrophobic', 'polar', 'positive', 'negative', 'special']}

high_props = get_property_composition(high_attention['aminoAcid'])
low_props = get_property_composition(low_attention['aminoAcid'])

property_comparison = pd.DataFrame({
    'Property': list(high_props.keys()),
    'High_Attention': list(high_props.values()),
    'Low_Attention': list(low_props.values())
})
property_comparison['Difference'] = property_comparison['High_Attention'] - property_comparison['Low_Attention']

print("\nAmino acid property comparison:")
print(property_comparison.to_string(index=False))

# ==============================================================================
# SECTION 5: V-J PAIRING ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: V-J GENE PAIRING ANALYSIS")
print("-" * 80)

# V-J pairing in high attention sequences
high_vj = high_attention.groupby(['vGeneName', 'jGeneName']).size().reset_index(name='count')
high_vj = high_vj.sort_values('count', ascending=False)

print("\nTop V-J pairings in high-attention sequences:")
for i, row in high_vj.head(10).iterrows():
    print(f"   {row['vGeneName']} + {row['jGeneName']}: {row['count']}")

# Compare to background
all_vj = attention_df.groupby(['vGeneName', 'jGeneName']).size().reset_index(name='count')
all_vj['freq'] = all_vj['count'] / all_vj['count'].sum()

high_vj['freq_high'] = high_vj['count'] / high_vj['count'].sum()
vj_merged = high_vj.merge(all_vj[['vGeneName', 'jGeneName', 'freq']],
                          on=['vGeneName', 'jGeneName'], how='left')
vj_merged['enrichment'] = vj_merged['freq_high'] / (vj_merged['freq'] + 0.0001)

# ==============================================================================
# SECTION 6: POSITION-SPECIFIC ANALYSIS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: POSITION-SPECIFIC ANALYSIS")
print("-" * 80)

def get_position_composition(sequences, max_len=20):
    """Get amino acid composition at each position"""
    position_counts = {i: Counter() for i in range(max_len)}

    for seq in sequences:
        for i, aa in enumerate(seq[:max_len]):
            position_counts[i][aa] += 1

    # Convert to frequencies
    position_freq = {}
    for pos, counts in position_counts.items():
        total = sum(counts.values())
        if total > 0:
            position_freq[pos] = {aa: counts.get(aa, 0) / total
                                  for aa in AMINO_ACIDS}
        else:
            position_freq[pos] = {aa: 0 for aa in AMINO_ACIDS}

    return position_freq

high_pos = get_position_composition(high_attention['aminoAcid'])
low_pos = get_position_composition(low_attention['aminoAcid'])

print("\nPosition-specific analysis complete (20 positions analyzed)")

# ==============================================================================
# SECTION 7: CREATE VISUALIZATIONS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: CREATING VISUALIZATIONS")
print("-" * 80)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Panel A: CDR3 length comparison
ax = fig.add_subplot(gs[0, 0])
ax.hist(low_lengths, bins=range(5, 30), alpha=0.5, label='Low Attention',
        color=COLORS['dark_gray'], edgecolor='white', density=True)
ax.hist(high_lengths, bins=range(5, 30), alpha=0.7, label='High Attention',
        color=COLORS['accent3'], edgecolor='white', density=True)
ax.axvline(high_lengths.mean(), color=COLORS['accent3'], linestyle='--', linewidth=2)
ax.axvline(low_lengths.mean(), color=COLORS['dark_gray'], linestyle='--', linewidth=2)
ax.set_xlabel('CDR3 Length (amino acids)')
ax.set_ylabel('Density')
ax.set_title('CDR3 Length: High vs Low Attention', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
clean_spines(ax)
add_panel_label(ax, 'A')

# Panel B: Amino acid composition bar chart
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(AMINO_ACIDS))
width = 0.35
ax.bar(x - width/2, aa_comparison['High_Attention'], width,
       label='High Attention', color=COLORS['accent3'], edgecolor='white')
ax.bar(x + width/2, aa_comparison['Low_Attention'], width,
       label='Low Attention', color=COLORS['dark_gray'], alpha=0.6, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(AMINO_ACIDS, fontsize=8)
ax.set_xlabel('Amino Acid')
ax.set_ylabel('Frequency (%)')
ax.set_title('Amino Acid Composition', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'B')

# Panel C: Property comparison
ax = fig.add_subplot(gs[0, 2])
props = property_comparison['Property'].values
x = np.arange(len(props))
ax.bar(x - width/2, property_comparison['High_Attention'], width,
       label='High Attention', color=COLORS['accent3'], edgecolor='white')
ax.bar(x + width/2, property_comparison['Low_Attention'], width,
       label='Low Attention', color=COLORS['dark_gray'], alpha=0.6, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(props, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Frequency (%)')
ax.set_title('Amino Acid Properties', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
clean_spines(ax)
add_panel_label(ax, 'C')

# Panel D: Enrichment heatmap
ax = fig.add_subplot(gs[1, 0])
# Create position-enrichment matrix
positions = list(range(15))
enrichment_matrix = np.zeros((len(AMINO_ACIDS), len(positions)))
for i, pos in enumerate(positions):
    for j, aa in enumerate(AMINO_ACIDS):
        high_val = high_pos.get(pos, {}).get(aa, 0)
        low_val = low_pos.get(pos, {}).get(aa, 0) + 0.001
        enrichment_matrix[j, i] = np.log2(high_val / low_val + 0.1)

im = ax.imshow(enrichment_matrix, cmap='RdBu_r', aspect='auto',
               vmin=-1, vmax=1)
ax.set_xticks(range(len(positions)))
ax.set_xticklabels([str(p+1) for p in positions], fontsize=8)
ax.set_yticks(range(len(AMINO_ACIDS)))
ax.set_yticklabels(AMINO_ACIDS, fontsize=8)
ax.set_xlabel('Position')
ax.set_ylabel('Amino Acid')
ax.set_title('Position-Specific Enrichment\n(log2 High/Low)', fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('log2(Enrichment)', fontsize=8)
add_panel_label(ax, 'D')

# Panel E: V-J pairing heatmap
ax = fig.add_subplot(gs[1, 1])
# Get top V and J genes
top_v = high_attention['vGeneName'].value_counts().head(8).index
top_j = high_attention['jGeneName'].value_counts().head(8).index

vj_matrix = np.zeros((len(top_v), len(top_j)))
for i, v in enumerate(top_v):
    for j, jg in enumerate(top_j):
        count = len(high_attention[(high_attention['vGeneName'] == v) &
                                   (high_attention['jGeneName'] == jg)])
        vj_matrix[i, j] = count

im = ax.imshow(vj_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(top_j)))
ax.set_xticklabels(top_j, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top_v)))
ax.set_yticklabels(top_v, fontsize=8)
ax.set_xlabel('J Gene')
ax.set_ylabel('V Gene')
ax.set_title('V-J Pairing in High Attention', fontweight='bold')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Count', fontsize=8)
add_panel_label(ax, 'E')

# Panel F: Summary table
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

summary_data = [
    ['Characteristic', 'High Attention', 'Low Attention'],
    ['Mean CDR3 Length', f'{high_lengths.mean():.1f}', f'{low_lengths.mean():.1f}'],
    ['Hydrophobic (%)', f'{high_props["hydrophobic"]:.1f}', f'{low_props["hydrophobic"]:.1f}'],
    ['Polar (%)', f'{high_props["polar"]:.1f}', f'{low_props["polar"]:.1f}'],
    ['Positive (%)', f'{high_props["positive"]:.1f}', f'{low_props["positive"]:.1f}'],
    ['Negative (%)', f'{high_props["negative"]:.1f}', f'{low_props["negative"]:.1f}'],
    ['N Sequences', f'{len(high_attention):,}', f'{len(low_attention):,}'],
]

table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                 loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for i in range(3):
    table[(0, i)].set_facecolor(COLORS['secondary'])
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax.set_title('Sequence Characteristics Summary', fontweight='bold', pad=20)
add_panel_label(ax, 'F', x=-0.05)

plt.suptitle('Figure: Sequence Characteristics Analysis',
             fontsize=14, fontweight='bold', y=1.02)
fig_path = os.path.join(FIGURES_DIR, 'figure_sequence_characteristics.png')
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

# Save length statistics
length_file = os.path.join(RESULTS_DIR, "cdr3_length_stats.csv")
length_stats.to_csv(length_file, index=False)
print(f"Saved: {length_file}")

# Save amino acid comparison
aa_file = os.path.join(RESULTS_DIR, "amino_acid_composition.csv")
aa_comparison.to_csv(aa_file, index=False)
print(f"Saved: {aa_file}")

# Save property comparison
prop_file = os.path.join(RESULTS_DIR, "aa_property_comparison.csv")
property_comparison.to_csv(prop_file, index=False)
print(f"Saved: {prop_file}")

# Save V-J pairing
vj_file = os.path.join(RESULTS_DIR, "vj_pairing_high_attention.csv")
high_vj.to_csv(vj_file, index=False)
print(f"Saved: {vj_file}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("SEQUENCE CHARACTERISTICS ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKEY FINDINGS:")
print(f"   High attention mean CDR3 length: {high_lengths.mean():.1f} AA")
print(f"   Low attention mean CDR3 length: {low_lengths.mean():.1f} AA")
print(f"   Hydrophobic enrichment: {high_props['hydrophobic'] - low_props['hydrophobic']:.1f}%")

print(f"\nOUTPUT FILES:")
print(f"   {length_file}")
print(f"   {aa_file}")
print(f"   {prop_file}")
print(f"   {vj_file}")
print(f"   {fig_path}")

print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
