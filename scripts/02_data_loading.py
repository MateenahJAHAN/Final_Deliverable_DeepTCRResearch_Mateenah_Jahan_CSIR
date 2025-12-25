#!/usr/bin/env python3
"""
Script 02: Data Loading and TRB Extraction
==========================================

This script corresponds to Cells 6-16 of the original Untitled1.ipynb notebook.

PURPOSE:
--------
1. Load the complete TCR dataset (both alpha and beta chains)
2. Extract TRB (T-Cell Receptor Beta) chain data only
3. Rename columns to match DeepTCR requirements
4. Add required metadata columns (sequenceStatus, count, frequencyCount)
5. Save processed data for downstream analysis

BIOLOGICAL CONTEXT:
-------------------
T-Cell Receptors (TCRs) are heterodimeric proteins composed of:
- ALPHA chain (TRA) - encoded by TRAV and TRAJ genes
- BETA chain (TRB) - encoded by TRBV, TRBD, and TRBJ genes

Each chain has:
- V gene (Variable) - determines recognition specificity
- D gene (Diversity) - only in beta chain, adds diversity
- J gene (Joining) - completes the binding region
- CDR3 (Complementarity-Determining Region 3) - the hypervariable loop that
  directly contacts antigens (tumor peptides in our case)

WHY TRB ONLY?
-------------
The beta chain (TRB) is often more informative because:
1. Higher diversity due to D gene segment
2. More directly involved in peptide-MHC recognition
3. Standard practice in TCR repertoire analysis
4. Reduces computational complexity while maintaining signal

IMMUNOTHERAPY CONTEXT:
----------------------
We're analyzing TCR sequences from Basal Cell Carcinoma (BCC) patients who
received checkpoint blockade immunotherapy (likely anti-PD-1 or anti-CTLA-4).

Responders: Patients whose tumors shrank (response_binary = 1)
Non-responders: Patients whose tumors didn't shrink (response_binary = 0)

Hypothesis: The TCR repertoire contains clonotypes (specific TCR sequences)
that recognize tumor antigens. These tumor-reactive TCRs may predict response.

DEEPTCR FORMAT REQUIREMENTS:
----------------------------
DeepTCR expects specific column names:
- aminoAcid: CDR3 amino acid sequence
- vGeneName: V gene name (e.g., TRBV20-1)
- jGeneName: J gene name (e.g., TRBJ1-2)
- dGeneName: D gene name (optional for TRB)
- sequenceStatus: "In" for productive, "Out" for non-productive
- count (templates/reads): Read count or UMI count (default 1 if unknown)
- frequencyCount (%): Percentage of repertoire (calculated per patient)

OUTPUTS:
--------
1. deeptcr_trb_only.csv - Extracted TRB data with standardized columns
2. deeptcr_trb_ready.csv - Complete DeepTCR-ready dataset
3. deeptcr_trb_ready.tsv - TSV format (alternative)
4. 02_data_loading.log - Processing log

Author: Converted from Untitled1.ipynb
Date: December 24, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# SETUP: PATHS AND LOGGING
# ==============================================================================

print("="*80)
print("DATA LOADING AND TRB EXTRACTION - SCRIPT 02")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
LOGS = os.path.join(PROJECT_ROOT, "logs")

# Define file paths
INPUT_FILE = os.path.join(DATA_RAW, "deeptcr_complete_dataset (5).csv")
OUTPUT_TRB_ONLY = os.path.join(DATA_PROCESSED, "deeptcr_trb_only.csv")
OUTPUT_TRB_READY = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.csv")
OUTPUT_TRB_TSV = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.tsv")
LOG_FILE = os.path.join(LOGS, "02_data_loading.log")

# ==============================================================================
# SECTION 1: LOAD COMPLETE DATASET (Notebook Cell 8-9)
# ==============================================================================

print("-" * 80)
print("SECTION 1: LOADING COMPLETE DATASET")
print("-" * 80)

print(f"\nReading data from: {INPUT_FILE}")
print("This may take a moment for large files...\n")

try:
    # Load the complete dataset
    df_complete = pd.read_csv(INPUT_FILE)

    print(f"✅ Data loaded successfully!")
    print(f"   Rows: {len(df_complete):,}")
    print(f"   Columns: {len(df_complete.columns)}")

    # Display first few rows
    print(f"\nFirst 3 rows of the dataset:")
    print("-" * 80)
    print(df_complete.head(3).to_string())

    # Display column information
    print(f"\n\nColumn names and data types:")
    print("-" * 80)
    for i, (col, dtype) in enumerate(zip(df_complete.columns, df_complete.dtypes), 1):
        non_null = df_complete[col].notna().sum()
        null_count = df_complete[col].isna().sum()
        print(f"{i:2d}. {col:30s} | {str(dtype):10s} | Non-null: {non_null:7,} | Null: {null_count:7,}")

    # Memory usage
    memory_mb = df_complete.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")

except FileNotFoundError:
    print(f"❌ ERROR: File not found at {INPUT_FILE}")
    print("   Please ensure the data file exists in data_raw/")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    sys.exit(1)

# ==============================================================================
# SECTION 2: EXTRACT TRB CHAIN DATA (Notebook Cell 10)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: EXTRACTING TRB (BETA CHAIN) DATA")
print("-" * 80)

print("\nExtracting TRB-specific columns...")

# Define columns to extract
# Based on notebook Cell 10
columns_to_keep = [
    'barcode',           # Cell barcode (unique identifier from 10x Genomics)
    'TRB_cdr3',         # Beta chain CDR3 amino acid sequence
    'TRB_v_gene',       # Beta chain V gene
    'TRB_j_gene',       # Beta chain J gene
    'patient_id',       # Patient identifier
    'response',         # Response label (text: "Responder"/"Non-responder")
    'response_binary'   # Response label (binary: 1/0)
]

try:
    # Extract TRB columns
    df_trb = df_complete[columns_to_keep].copy()

    print(f"✅ TRB data extracted!")
    print(f"   Rows: {len(df_trb):,}")
    print(f"   Columns: {len(df_trb.columns)}")

    # Check for missing values in critical columns
    print(f"\nMissing value counts:")
    print("-" * 80)
    missing_counts = df_trb.isnull().sum()
    for col, count in missing_counts.items():
        percentage = (count / len(df_trb)) * 100
        status = "⚠️ " if count > 0 else "✅"
        print(f"{status} {col:20s}: {count:7,} ({percentage:5.2f}%)")

    # Display sample of TRB data
    print(f"\nSample TRB data:")
    print("-" * 80)
    print(df_trb.head(5).to_string())

except KeyError as e:
    print(f"❌ ERROR: Missing expected column: {e}")
    print(f"   Available columns: {list(df_complete.columns)}")
    sys.exit(1)

# ==============================================================================
# SECTION 3: RENAME COLUMNS FOR DEEPTCR (Notebook Cell 10 continued)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: RENAMING COLUMNS FOR DEEPTCR COMPATIBILITY")
print("-" * 80)

# Define column name mappings
# DeepTCR expects specific names
column_mapping = {
    'TRB_cdr3': 'aminoAcid',       # CDR3 amino acid sequence
    'TRB_v_gene': 'vGeneName',     # V gene name
    'TRB_j_gene': 'jGeneName',     # J gene name
}

print(f"\nApplying column name mappings:")
for old_name, new_name in column_mapping.items():
    print(f"   {old_name:20s} → {new_name}")

# Rename columns
df_trb.rename(columns=column_mapping, inplace=True)

print(f"\n✅ Columns renamed!")
print(f"   New column names: {list(df_trb.columns)}")

# Save TRB-only dataset
print(f"\nSaving TRB-only dataset to: {OUTPUT_TRB_ONLY}")
df_trb.to_csv(OUTPUT_TRB_ONLY, index=False)
print(f"✅ Saved! ({len(df_trb):,} rows)")

# ==============================================================================
# SECTION 4: ADD DEEPTCR REQUIRED COLUMNS (Notebook Cell 14)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: ADDING DEEPTCR REQUIRED METADATA COLUMNS")
print("-" * 80)

# Create a copy for the ready dataset
df_trb_ready = df_trb.copy()

# Add sequenceStatus column
# "In" = productive (in-frame), "Out" = non-productive (out-of-frame)
# The notebook sets all sequences to "In" (productive)
print("\n1. Adding 'sequenceStatus' column...")
df_trb_ready['sequenceStatus'] = 'In'
print(f"   ✅ All sequences marked as 'In' (productive)")

# Add count column (templates/reads)
# This represents the number of reads or UMI counts for each sequence
# The notebook uses 1 as a placeholder (actual counts may not be available)
print("\n2. Adding 'count (templates/reads)' column...")
df_trb_ready['count (templates/reads)'] = 1
print(f"   ✅ All sequences assigned count = 1 (placeholder)")

# Add frequencyCount column (%)
# This represents what percentage of the patient's repertoire this sequence represents
# Calculated per patient as: (1 / total sequences for that patient) * 100
print("\n3. Adding 'frequencyCount (%)' column...")

# Group by patient and calculate frequency
patient_counts = df_trb_ready.groupby('patient_id').size()
print(f"   Calculating frequencies for {len(patient_counts)} patients...")

# Calculate frequency for each sequence
df_trb_ready['frequencyCount (%)'] = df_trb_ready.apply(
    lambda row: (1 / patient_counts[row['patient_id']]) * 100,
    axis=1
)

print(f"   ✅ Frequencies calculated!")
print(f"\n   Frequency statistics:")
print(f"   - Mean: {df_trb_ready['frequencyCount (%)'].mean():.6f}%")
print(f"   - Median: {df_trb_ready['frequencyCount (%)'].median():.6f}%")
print(f"   - Min: {df_trb_ready['frequencyCount (%)'].min():.6f}%")
print(f"   - Max: {df_trb_ready['frequencyCount (%)'].max():.6f}%")

# ==============================================================================
# SECTION 5: FINAL DATASET SUMMARY
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: FINAL DEEPTCR-READY DATASET SUMMARY")
print("-" * 80)

print(f"\nFinal dataset shape: {df_trb_ready.shape}")
print(f"   Rows (sequences): {len(df_trb_ready):,}")
print(f"   Columns: {len(df_trb_ready.columns)}")

print(f"\nColumn list:")
for i, col in enumerate(df_trb_ready.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\nSample of final dataset:")
print("-" * 80)
print(df_trb_ready.head(10).to_string())

# Data quality checks
print(f"\n\nData Quality Checks:")
print("-" * 80)

# Check for null values
null_counts = df_trb_ready.isnull().sum()
has_nulls = null_counts[null_counts > 0]
if len(has_nulls) == 0:
    print("✅ No null values detected")
else:
    print("⚠️  Null values detected:")
    for col, count in has_nulls.items():
        print(f"   {col}: {count:,}")

# Check for empty CDR3 sequences
empty_cdr3 = (df_trb_ready['aminoAcid'].str.len() == 0).sum()
if empty_cdr3 == 0:
    print("✅ No empty CDR3 sequences")
else:
    print(f"⚠️  {empty_cdr3:,} empty CDR3 sequences detected")

# Check patient distribution
patient_dist = df_trb_ready.groupby('patient_id').size()
print(f"\n✅ Patient distribution:")
print(f"   Total patients: {len(patient_dist)}")
print(f"   Sequences per patient - Mean: {patient_dist.mean():.0f}, Median: {patient_dist.median():.0f}")
print(f"   Min: {patient_dist.min()}, Max: {patient_dist.max()}")

# Check response distribution
response_dist = df_trb_ready['response_binary'].value_counts()
print(f"\n✅ Response distribution:")
for label, count in response_dist.items():
    label_name = "Responder" if label == 1 else "Non-responder"
    percentage = (count / len(df_trb_ready)) * 100
    print(f"   {label_name} ({label}): {count:,} sequences ({percentage:.2f}%)")

# Patient-level response distribution
patient_response = df_trb_ready.groupby('patient_id')['response_binary'].first().value_counts()
print(f"\n✅ Patient-level response distribution:")
for label, count in patient_response.items():
    label_name = "Responder" if label == 1 else "Non-responder"
    percentage = (count / len(patient_response)) * 100
    print(f"   {label_name} ({label}): {count} patients ({percentage:.2f}%)")

# ==============================================================================
# SECTION 6: SAVE PROCESSED DATA
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: SAVING PROCESSED DATA")
print("-" * 80)

# Save as CSV
print(f"\nSaving DeepTCR-ready dataset...")
print(f"   Format: CSV")
print(f"   File: {OUTPUT_TRB_READY}")
df_trb_ready.to_csv(OUTPUT_TRB_READY, index=False)
file_size_mb = os.path.getsize(OUTPUT_TRB_READY) / (1024 * 1024)
print(f"   ✅ Saved! Size: {file_size_mb:.2f} MB")

# Save as TSV (alternative format)
print(f"\n   Format: TSV")
print(f"   File: {OUTPUT_TRB_TSV}")
df_trb_ready.to_csv(OUTPUT_TRB_TSV, sep='\t', index=False)
file_size_mb = os.path.getsize(OUTPUT_TRB_TSV) / (1024 * 1024)
print(f"   ✅ Saved! Size: {file_size_mb:.2f} MB")

# ==============================================================================
# SECTION 7: SAVE LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: SAVING PROCESSING LOG")
print("-" * 80)

try:
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATA LOADING AND TRB EXTRACTION LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("INPUT DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"File: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df_complete):,}\n")
        f.write(f"Columns: {len(df_complete.columns)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PROCESSING STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Loaded complete dataset (TRA + TRB)\n")
        f.write("2. Extracted TRB chain data only\n")
        f.write("3. Renamed columns for DeepTCR compatibility\n")
        f.write("4. Added sequenceStatus = 'In'\n")
        f.write("5. Added count (templates/reads) = 1\n")
        f.write("6. Calculated frequencyCount (%) per patient\n\n")

        f.write("-" * 80 + "\n")
        f.write("OUTPUT DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"File: {OUTPUT_TRB_READY}\n")
        f.write(f"Rows: {len(df_trb_ready):,}\n")
        f.write(f"Columns: {len(df_trb_ready.columns)}\n")
        f.write(f"Patients: {df_trb_ready['patient_id'].nunique()}\n")
        f.write(f"Responders: {patient_response[1]} patients\n")
        f.write(f"Non-responders: {patient_response[0]} patients\n\n")

        f.write("-" * 80 + "\n")
        f.write("COLUMN NAMES\n")
        f.write("-" * 80 + "\n")
        for i, col in enumerate(df_trb_ready.columns, 1):
            f.write(f"{i:2d}. {col}\n")

    print(f"\n✅ Log saved to: {LOG_FILE}")

except Exception as e:
    print(f"\n⚠️  Could not save log file: {e}")

# ==============================================================================
# SUMMARY AND NEXT STEPS
# ==============================================================================

print("\n" + "="*80)
print("DATA LOADING COMPLETE")
print("="*80)

print("\n📋 SUMMARY:")
print(f"   ✅ Loaded {len(df_complete):,} sequences from complete dataset")
print(f"   ✅ Extracted {len(df_trb_ready):,} TRB sequences")
print(f"   ✅ Processed {df_trb_ready['patient_id'].nunique()} patients")
print(f"   ✅ Added DeepTCR-required metadata columns")
print(f"   ✅ Saved processed data to: data_processed/")

print("\n📊 NEXT STEPS:")
print("   → Run Script 03: Exploratory Data Analysis")
print("   → Command: python scripts/03_exploratory_analysis.py")

print("\n" + "="*80)
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
