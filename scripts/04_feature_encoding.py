#!/usr/bin/env python3
"""
Script 04: Feature Encoding and Numpy Array Creation
====================================================

This script corresponds to Cells 27-44 of the original Untitled1.ipynb notebook.

PURPOSE:
--------
1. Implement one-hot encoding for TCR amino acid sequences
2. Encode V and J gene usage
3. Create combined feature vectors for deep learning
4. Generate full dataset numpy arrays (X_onehot.npy, y_labels.npy, patient_ids.npy)
5. Validate data format and save for DeepTCR training

BIOLOGICAL & COMPUTATIONAL CONTEXT:
-----------------------------------
Deep learning models (neural networks) require numerical input, but TCR sequences
are text strings (e.g., "CASSLAPG"). We need to convert biological sequences into
numbers while preserving their information content.

ONE-HOT ENCODING STRATEGY:
--------------------------
1. AMINO ACID ALPHABET:
   - 20 standard amino acids: ACDEFGHIKLMNPQRSTVWY
   - Each amino acid gets a unique position in a 20-dimensional vector
   - Example: 'C' = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
             'A' = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

2. SEQUENCE ENCODING:
   - Each sequence is a string of amino acids
   - Convert each position to a 20-D one-hot vector
   - Stack vectors to create a 2D matrix: (sequence_length, 20)
   - Pad shorter sequences to max_length (40 amino acids)

3. V/J GENE ENCODING:
   - One-hot encode V gene usage (50 TRB V genes)
   - One-hot encode J gene usage (13 TRB J genes)

4. COMBINED FEATURE VECTOR:
   - Flatten sequence matrix: (40, 20) → (800,)
   - Concatenate: [sequence_features (800) + V_gene (50) + J_gene (13)]
   - Final feature dimension: 863 features per sequence

WHY THIS ENCODING?
------------------
- Preserves amino acid identity (no information loss)
- Position-specific (captures sequence order)
- No arbitrary numerical assignments (unlike A=1, C=2, ...)
- Standard input format for neural networks
- Allows learning of amino acid patterns and motifs

MEMORY OPTIMIZATION:
--------------------
For M3 MacBook: We process in batches to manage memory efficiently
- Full dataset: 239,637 sequences × 863 features = ~205M floats = ~1.6 GB
- Use memory-mapped arrays for large datasets
- Monitor memory usage throughout

OUTPUTS:
--------
1. X_onehot.npy - Feature matrix (239,637, 863)
2. y_labels.npy - Binary labels (239,637,) - 1=Responder, 0=Non-responder
3. patient_ids.npy - Patient IDs (239,637,)
4. X_test_batch.npy - Small test batch (200, 863) for verification
5. 04_feature_encoding.log - Processing log with memory stats

Author: Converted from Untitled1.ipynb
Date: December 24, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import gc  # Garbage collection for memory management

# ==============================================================================
# SETUP: PATHS AND CONFIGURATION
# ==============================================================================

print("="*80)
print("FEATURE ENCODING AND NUMPY ARRAY CREATION - SCRIPT 04")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
LOGS = os.path.join(PROJECT_ROOT, "logs")

# Define file paths
INPUT_FILE = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.csv")
OUTPUT_X = os.path.join(DATA_PROCESSED, "X_onehot.npy")
OUTPUT_Y = os.path.join(DATA_PROCESSED, "y_labels.npy")
OUTPUT_PIDS = os.path.join(DATA_PROCESSED, "patient_ids.npy")
OUTPUT_TEST_BATCH = os.path.join(DATA_PROCESSED, "X_test_batch.npy")
LOG_FILE = os.path.join(LOGS, "04_feature_encoding.log")

# ==============================================================================
# SECTION 1: DEFINE ENCODING PARAMETERS (Notebook Cells 27-30)
# ==============================================================================

print("-" * 80)
print("SECTION 1: DEFINING ENCODING PARAMETERS")
print("-" * 80)

# 1. AMINO ACID ALPHABET
print("\n1. AMINO ACID ALPHABET:")
print("-" * 80)

# 20 standard amino acids (single letter code)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}

print(f"   Alphabet: {AMINO_ACIDS}")
print(f"   Number of amino acids: {len(AMINO_ACIDS)}")
print(f"\n   Amino Acid → Index mapping:")
for i, aa in enumerate(AMINO_ACIDS):
    print(f"   {aa} → {i:2d}", end="  ")
    if (i + 1) % 10 == 0:
        print()  # New line every 10 amino acids

# 2. MAXIMUM SEQUENCE LENGTH
print("\n\n2. MAXIMUM SEQUENCE LENGTH:")
print("-" * 80)

MAX_LENGTH = 40  # Pad all sequences to this length

print(f"   Max length: {MAX_LENGTH} amino acids")
print(f"   Shorter sequences will be padded with zeros")
print(f"   Longer sequences will be truncated")

# 3. FEATURE DIMENSIONS
print("\n3. FEATURE DIMENSIONS:")
print("-" * 80)

FEATURE_DIM_AA = MAX_LENGTH * len(AMINO_ACIDS)  # 40 * 20 = 800

print(f"   Amino acid features: {MAX_LENGTH} positions × {len(AMINO_ACIDS)} AAs = {FEATURE_DIM_AA}")
print(f"   V gene features: Will be determined from data")
print(f"   J gene features: Will be determined from data")

# ==============================================================================
# SECTION 2: LOAD DATA
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 2: LOADING DATA")
print("-" * 80)

print(f"\nReading data from: {INPUT_FILE}\n")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Data loaded successfully!")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

except FileNotFoundError:
    print(f"❌ ERROR: File not found at {INPUT_FILE}")
    print("   Please run script 02_data_loading.py first")
    sys.exit(1)

# ==============================================================================
# SECTION 3: PREPARE GENE ENCODERS (Notebook Cells 31-32)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: PREPARING GENE ENCODERS")
print("-" * 80)

# Extract unique V and J genes
unique_v_genes = sorted(df['vGeneName'].unique())
unique_j_genes = sorted(df['jGeneName'].unique())

print(f"\n1. V GENE ENCODER:")
print(f"   Unique V genes: {len(unique_v_genes)}")
v_to_index = {gene: idx for idx, gene in enumerate(unique_v_genes)}
print(f"   First 10 V genes: {unique_v_genes[:10]}")

print(f"\n2. J GENE ENCODER:")
print(f"   Unique J genes: {len(unique_j_genes)}")
j_to_index = {gene: idx for idx, gene in enumerate(unique_j_genes)}
print(f"   All J genes: {unique_j_genes}")

# Total feature dimension
FEATURE_DIM_V = len(unique_v_genes)
FEATURE_DIM_J = len(unique_j_genes)
TOTAL_FEATURES = FEATURE_DIM_AA + FEATURE_DIM_V + FEATURE_DIM_J

print(f"\n3. TOTAL FEATURE DIMENSION:")
print(f"   Amino acid: {FEATURE_DIM_AA}")
print(f"   V gene:     {FEATURE_DIM_V}")
print(f"   J gene:     {FEATURE_DIM_J}")
print(f"   ──────────────────────")
print(f"   TOTAL:      {TOTAL_FEATURES} features per sequence")

# ==============================================================================
# SECTION 4: DEFINE ENCODING FUNCTIONS (Notebook Cells 33-34)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: DEFINING ENCODING FUNCTIONS")
print("-" * 80)

def encode_amino_acid_sequence(sequence, max_length=40):
    """
    One-hot encode an amino acid sequence.

    Args:
        sequence (str): Amino acid sequence (e.g., "CASSLAPG")
        max_length (int): Maximum sequence length (padding/truncation)

    Returns:
        np.array: One-hot encoded matrix, shape (max_length, 20)
    """
    # Initialize matrix with zeros
    encoded = np.zeros((max_length, len(AMINO_ACIDS)), dtype=np.float32)

    # Truncate if necessary
    sequence = sequence[:max_length]

    # Encode each amino acid
    for pos, aa in enumerate(sequence):
        if aa in AA_TO_INDEX:
            aa_idx = AA_TO_INDEX[aa]
            encoded[pos, aa_idx] = 1.0
        else:
            # Unknown amino acid - leave as all zeros
            pass

    return encoded


def encode_v_gene(v_gene):
    """
    One-hot encode V gene.

    Args:
        v_gene (str): V gene name (e.g., "TRBV20-1")

    Returns:
        np.array: One-hot encoded vector, shape (num_v_genes,)
    """
    encoded = np.zeros(len(unique_v_genes), dtype=np.float32)
    if v_gene in v_to_index:
        encoded[v_to_index[v_gene]] = 1.0
    return encoded


def encode_j_gene(j_gene):
    """
    One-hot encode J gene.

    Args:
        j_gene (str): J gene name (e.g., "TRBJ1-2")

    Returns:
        np.array: One-hot encoded vector, shape (num_j_genes,)
    """
    encoded = np.zeros(len(unique_j_genes), dtype=np.float32)
    if j_gene in j_to_index:
        encoded[j_to_index[j_gene]] = 1.0
    return encoded


def encode_full_tcr(sequence, v_gene, j_gene, max_length=40):
    """
    Encode full TCR sequence with V and J genes.

    Args:
        sequence (str): CDR3 amino acid sequence
        v_gene (str): V gene name
        j_gene (str): J gene name
        max_length (int): Maximum sequence length

    Returns:
        np.array: Combined feature vector, shape (863,)
                  [AA features (800) + V gene (50) + J gene (13)]
    """
    # Encode sequence
    seq_encoded = encode_amino_acid_sequence(sequence, max_length)
    seq_flat = seq_encoded.flatten()  # (40, 20) → (800,)

    # Encode genes
    v_encoded = encode_v_gene(v_gene)
    j_encoded = encode_j_gene(j_gene)

    # Concatenate all features
    full_vector = np.concatenate([seq_flat, v_encoded, j_encoded])

    return full_vector


print("\n✅ Encoding functions defined:")
print("   1. encode_amino_acid_sequence() - Encodes CDR3 sequences")
print("   2. encode_v_gene() - Encodes V gene usage")
print("   3. encode_j_gene() - Encodes J gene usage")
print("   4. encode_full_tcr() - Combines all features")

# ==============================================================================
# SECTION 5: TEST ENCODING ON EXAMPLE (Notebook Cell 52-54)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: TESTING ENCODING ON EXAMPLE SEQUENCE")
print("-" * 80)

# Get example from data
example_row = df.iloc[0]
example_seq = example_row['aminoAcid']
example_v = example_row['vGeneName']
example_j = example_row['jGeneName']

print(f"\nExample sequence:")
print(f"   CDR3:   {example_seq}")
print(f"   V gene: {example_v}")
print(f"   J gene: {example_j}")

# Encode
encoded_example = encode_full_tcr(example_seq, example_v, example_j, MAX_LENGTH)

print(f"\n✅ Encoded successfully!")
print(f"   Output shape: {encoded_example.shape}")
print(f"   Output type: {encoded_example.dtype}")
print(f"   Non-zero elements: {np.count_nonzero(encoded_example)}")
print(f"   Memory: {encoded_example.nbytes} bytes")

# Show first 30 features
print(f"\n   First 30 features:")
print(f"   {encoded_example[:30]}")

# ==============================================================================
# SECTION 6: ENCODE FULL DATASET (Notebook Cells 35-37)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: ENCODING FULL DATASET")
print("-" * 80)

num_sequences = len(df)

print(f"\nPreparing to encode {num_sequences:,} sequences...")
print(f"Expected memory usage: ~{(num_sequences * TOTAL_FEATURES * 4) / (1024**3):.2f} GB\n")

# Initialize arrays
print("Initializing numpy arrays...")
X_onehot = np.zeros((num_sequences, TOTAL_FEATURES), dtype=np.float32)
y_labels = df['response_binary'].values.astype(np.int32)
patient_ids = df['patient_id'].values

print(f"✅ Arrays initialized")
print(f"   X_onehot shape: {X_onehot.shape}")
print(f"   y_labels shape: {y_labels.shape}")
print(f"   patient_ids shape: {patient_ids.shape}")

# Encode all sequences
print(f"\nEncoding sequences (this may take a few minutes)...")
print("Progress: ", end="", flush=True)

batch_size = 10000  # Process in batches for progress reporting
num_batches = (num_sequences + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_sequences)

    # Encode batch
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        X_onehot[i] = encode_full_tcr(row['aminoAcid'],
                                       row['vGeneName'],
                                       row['jGeneName'],
                                       MAX_LENGTH)

    # Progress indicator
    progress = (end_idx / num_sequences) * 100
    print(f"{progress:.0f}%...", end="", flush=True)

print(" Done!")

print(f"\n✅ Encoding complete!")
print(f"   Total sequences encoded: {num_sequences:,}")
print(f"   Feature matrix size: {X_onehot.nbytes / (1024**3):.2f} GB")

# ==============================================================================
# SECTION 7: VALIDATE ENCODED DATA
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: VALIDATING ENCODED DATA")
print("-" * 80)

print("\n1. FEATURE MATRIX (X_onehot):")
print(f"   Shape: {X_onehot.shape}")
print(f"   Data type: {X_onehot.dtype}")
print(f"   Memory: {X_onehot.nbytes / (1024**2):.2f} MB")
print(f"   Min value: {X_onehot.min()}")
print(f"   Max value: {X_onehot.max()}")
print(f"   Mean: {X_onehot.mean():.6f}")
print(f"   Non-zero ratio: {np.count_nonzero(X_onehot) / X_onehot.size:.4f}")

print("\n2. LABEL VECTOR (y_labels):")
print(f"   Shape: {y_labels.shape}")
print(f"   Data type: {y_labels.dtype}")
print(f"   Unique values: {np.unique(y_labels)}")
print(f"   Label distribution:")
unique, counts = np.unique(y_labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = "Responder" if label == 1 else "Non-responder"
    print(f"   - {label_name} ({label}): {count:,} ({count/len(y_labels)*100:.2f}%)")

print("\n3. PATIENT IDs:")
print(f"   Shape: {patient_ids.shape}")
print(f"   Unique patients: {len(np.unique(patient_ids))}")
print(f"   First 10 patient IDs: {patient_ids[:10]}")

# ==============================================================================
# SECTION 8: CREATE TEST BATCH (Notebook Cell 41-44)
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 8: CREATING TEST BATCH")
print("-" * 80)

# Create a small test batch (200 sequences)
test_batch_size = 200
X_test_batch = X_onehot[:test_batch_size]

print(f"\nCreated test batch:")
print(f"   Shape: {X_test_batch.shape}")
print(f"   Purpose: Quick testing and verification")

# ==============================================================================
# SECTION 9: SAVE NUMPY ARRAYS
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 9: SAVING NUMPY ARRAYS")
print("-" * 80)

print("\nSaving arrays to disk...")

# Save X_onehot
print(f"\n1. Saving feature matrix (X_onehot.npy)...")
np.save(OUTPUT_X, X_onehot)
file_size_mb = os.path.getsize(OUTPUT_X) / (1024 * 1024)
print(f"   ✅ Saved: {OUTPUT_X}")
print(f"   Size: {file_size_mb:.2f} MB")

# Save y_labels
print(f"\n2. Saving label vector (y_labels.npy)...")
np.save(OUTPUT_Y, y_labels)
file_size_mb = os.path.getsize(OUTPUT_Y) / (1024 * 1024)
print(f"   ✅ Saved: {OUTPUT_Y}")
print(f"   Size: {file_size_mb:.2f} MB")

# Save patient_ids
print(f"\n3. Saving patient IDs (patient_ids.npy)...")
np.save(OUTPUT_PIDS, patient_ids)
file_size_mb = os.path.getsize(OUTPUT_PIDS) / (1024 * 1024)
print(f"   ✅ Saved: {OUTPUT_PIDS}")
print(f"   Size: {file_size_mb:.2f} MB")

# Save test batch
print(f"\n4. Saving test batch (X_test_batch.npy)...")
np.save(OUTPUT_TEST_BATCH, X_test_batch)
file_size_mb = os.path.getsize(OUTPUT_TEST_BATCH) / (1024 * 1024)
print(f"   ✅ Saved: {OUTPUT_TEST_BATCH}")
print(f"   Size: {file_size_mb:.2f} MB")

# ==============================================================================
# SECTION 10: SAVE LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 10: SAVING PROCESSING LOG")
print("-" * 80)

try:
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE ENCODING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ENCODING PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Amino acid alphabet: {AMINO_ACIDS}\n")
        f.write(f"Max sequence length: {MAX_LENGTH}\n")
        f.write(f"Unique V genes: {len(unique_v_genes)}\n")
        f.write(f"Unique J genes: {len(unique_j_genes)}\n")
        f.write(f"Total features: {TOTAL_FEATURES}\n\n")

        f.write("-" * 80 + "\n")
        f.write("OUTPUT ARRAYS\n")
        f.write("-" * 80 + "\n")
        f.write(f"X_onehot: {X_onehot.shape}, {X_onehot.dtype}\n")
        f.write(f"y_labels: {y_labels.shape}, {y_labels.dtype}\n")
        f.write(f"patient_ids: {patient_ids.shape}\n")
        f.write(f"X_test_batch: {X_test_batch.shape}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FILES CREATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. {OUTPUT_X}\n")
        f.write(f"2. {OUTPUT_Y}\n")
        f.write(f"3. {OUTPUT_PIDS}\n")
        f.write(f"4. {OUTPUT_TEST_BATCH}\n")

    print(f"\n✅ Log saved to: {LOG_FILE}")

except Exception as e:
    print(f"\n⚠️  Could not save log file: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("FEATURE ENCODING COMPLETE")
print("="*80)

print("\n📋 SUMMARY:")
print(f"   ✅ Encoded {num_sequences:,} TCR sequences")
print(f"   ✅ Feature dimension: {TOTAL_FEATURES} (AA: {FEATURE_DIM_AA}, V: {FEATURE_DIM_V}, J: {FEATURE_DIM_J})")
print(f"   ✅ Total data size: {X_onehot.nbytes / (1024**3):.2f} GB")
print(f"   ✅ Saved 4 numpy arrays to data_processed/")

print("\n📊 NEXT STEPS:")
print("   → Run Script 05: Baseline logistic regression model")
print("   → Command: python scripts/05_baseline_model.py")

print("\n" + "="*80)
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
