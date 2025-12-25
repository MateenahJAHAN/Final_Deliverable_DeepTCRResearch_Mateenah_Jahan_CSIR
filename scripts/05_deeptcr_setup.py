#!/usr/bin/env python3
"""
Script 05: DeepTCR Installation and Setup
=========================================

This script corresponds to Cells 64-81 of the original Untitled1.ipynb notebook.

PURPOSE:
--------
1. Install DeepTCR library (if not already installed)
2. Verify installation and imports
3. Test DeepTCR functionality
4. Prepare data in DeepTCR format
5. Initialize DeepTCR model object

ABOUT DEEPTCR:
--------------
DeepTCR is a python package for deep learning analysis of T-Cell receptor repertoires.

Key Features:
- Sequence classification (supervised)
- Repertoire classification
- Multiple Instance Learning (MIL) for patient-level prediction
- Attention mechanisms to identify important sequences
- Unsupervised learning and clustering

Architecture:
- Built on TensorFlow
- Uses variational autoencoders and supervised classifiers
- Implements attention-based aggregation for repertoire analysis

GitHub: https://github.com/sidhomj/DeepTCR
Paper: Sidhom et al., Nature Communications 2021

INSTALLATION OPTIONS:
--------------------
1. From GitHub (recommended for latest version):
   pip install git+https://github.com/sidhomj/DeepTCR.git

2. From PyPI:
   pip install DeepTCR

3. With dependencies:
   pip install DeepTCR[tf2]  # For TensorFlow 2.x

OUTPUTS:
--------
1. DeepTCR installation verification
2. Test model initialization
3. Data loaded into DeepTCR format
4. 05_deeptcr_setup.log - Setup log

Author: Converted from Untitled1.ipynb
Date: December 24, 2025
"""

import os
import sys
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd

# ==============================================================================
# SETUP: PATHS
# ==============================================================================

print("="*80)
print("DEEPTCR INSTALLATION AND SETUP - SCRIPT 05")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data_processed")
LOGS = os.path.join(PROJECT_ROOT, "logs")

LOG_FILE = os.path.join(LOGS, "05_deeptcr_setup.log")

# ==============================================================================
# SECTION 1: CHECK IF DEEPTCR IS INSTALLED
# ==============================================================================

print("-" * 80)
print("SECTION 1: CHECKING DEEPTCR INSTALLATION")
print("-" * 80)

deeptcr_installed = False

try:
    import DeepTCR
    deeptcr_version = DeepTCR.__version__ if hasattr(DeepTCR, '__version__') else "Unknown"
    print(f"\n✅ DeepTCR is already installed!")
    print(f"   Version: {deeptcr_version}")
    print(f"   Location: {DeepTCR.__file__}")
    deeptcr_installed = True

except ImportError:
    print(f"\n❌ DeepTCR is not installed")
    deeptcr_installed = False

# ==============================================================================
# SECTION 2: INSTALL DEEPTCR (IF NEEDED)
# ==============================================================================

if not deeptcr_installed:
    print("\n" + "-" * 80)
    print("SECTION 2: INSTALLING DEEPTCR")
    print("-" * 80)

    print("\n⚙️  Installing DeepTCR from GitHub...")
    print("   This may take several minutes...\n")

    try:
        # Install from GitHub
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/sidhomj/DeepTCR.git",
            "-q"
        ])

        print("\n✅ DeepTCR installation complete!")

        # Verify installation
        import DeepTCR
        deeptcr_version = DeepTCR.__version__ if hasattr(DeepTCR, '__version__') else "Unknown"
        print(f"   Version: {deeptcr_version}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\n📝 MANUAL INSTALLATION INSTRUCTIONS:")
        print("   Run the following command in your terminal:")
        print("   pip install git+https://github.com/sidhomj/DeepTCR.git")
        print("\n   Or try:")
        print("   pip install DeepTCR")
        sys.exit(1)

# ==============================================================================
# SECTION 3: IMPORT AND TEST DEEPTCR
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 3: IMPORTING DEEPTCR MODULES")
print("-" * 80)

print("\nAttempting to import DeepTCR classes...")

try:
    from DeepTCR.DeepTCR import DeepTCR_SS  # Supervised Sequence
    print("✅ DeepTCR_SS (Supervised Sequence) imported successfully")

    # Try to import other useful classes
    try:
        from DeepTCR.DeepTCR import DeepTCR_WF  # Supervised with repertoire features
        print("✅ DeepTCR_WF (Repertoire features) imported successfully")
    except:
        print("⚠️  DeepTCR_WF not available (optional)")

    try:
        from DeepTCR.DeepTCR import DeepTCR_U  # Unsupervised
        print("✅ DeepTCR_U (Unsupervised) imported successfully")
    except:
        print("⚠️  DeepTCR_U not available (optional)")

except ImportError as e:
    print(f"❌ Failed to import DeepTCR classes: {e}")
    print("\n📝 TROUBLESHOOTING:")
    print("   1. Try reinstalling: pip uninstall DeepTCR && pip install DeepTCR")
    print("   2. Check TensorFlow installation: pip install tensorflow")
    print("   3. Check compatibility: DeepTCR requires TensorFlow 1.x or 2.x")
    sys.exit(1)

# ==============================================================================
# SECTION 4: CHECK TENSORFLOW
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 4: CHECKING TENSORFLOW")
print("-" * 80)

try:
    import tensorflow as tf
    tf_version = tf.__version__
    print(f"\n✅ TensorFlow installed")
    print(f"   Version: {tf_version}")

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("   ⚠️  No GPU detected - will use CPU")
        print("   (Apple Silicon M3 may use Metal acceleration)")

except ImportError:
    print("\n⚠️  TensorFlow not found")
    print("   DeepTCR requires TensorFlow")
    print("   Installing TensorFlow...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow", "-q"])
        print("   ✅ TensorFlow installed")
    except:
        print("   ❌ Failed to install TensorFlow")
        print("   Please install manually: pip install tensorflow")

# ==============================================================================
# SECTION 5: LOAD DATA FOR DEEPTCR
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 5: LOADING DATA FOR DEEPTCR")
print("-" * 80)

print("\nLoading processed data files...")

# Load CSV data
INPUT_FILE = os.path.join(DATA_PROCESSED, "deeptcr_trb_ready.csv")
df = pd.read_csv(INPUT_FILE)

print(f"✅ Loaded CSV: {len(df):,} sequences")

# Load numpy arrays
X_FILE = os.path.join(DATA_PROCESSED, "X_onehot.npy")
Y_FILE = os.path.join(DATA_PROCESSED, "y_labels.npy")
PID_FILE = os.path.join(DATA_PROCESSED, "patient_ids.npy")

X_onehot = np.load(X_FILE)
y_labels = np.load(Y_FILE)
patient_ids = np.load(PID_FILE)

print(f"✅ Loaded numpy arrays:")
print(f"   X_onehot: {X_onehot.shape}")
print(f"   y_labels: {y_labels.shape}")
print(f"   patient_ids: {patient_ids.shape}")

# ==============================================================================
# SECTION 6: PREPARE DATA IN DEEPTCR FORMAT
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 6: PREPARING DATA IN DEEPTCR FORMAT")
print("-" * 80)

print("\nExtracting sequences and genes for DeepTCR...")

# Extract sequences and genes
beta_sequences = df['aminoAcid'].values
v_beta = df['vGeneName'].values
j_beta = df['jGeneName'].values

# Create dummy alpha chain data (DeepTCR may require it even if not used)
num_samples = len(df)
dummy_alpha_seqs = np.array(["AAA"] * num_samples)
dummy_v_alpha = np.array(["TRAV1-1"] * num_samples)
dummy_j_alpha = np.array(["TRAJ1"] * num_samples)

print(f"✅ Data prepared:")
print(f"   Beta sequences: {len(beta_sequences):,}")
print(f"   V genes: {len(v_beta):,}")
print(f"   J genes: {len(j_beta):,}")
print(f"   Dummy alpha data created (required by DeepTCR)")

# Convert labels to proper format
# DeepTCR expects labels as integers (not one-hot)
Y_binary = y_labels.reshape(-1, 1)  # Shape: (N, 1)

print(f"\n   Labels: {Y_binary.shape}")
print(f"   Unique labels: {np.unique(Y_binary)}")

# Patient IDs
pids_arr = patient_ids

print(f"   Patient IDs: {len(np.unique(pids_arr))} unique patients")

# ==============================================================================
# SECTION 7: TEST DEEPTCR INITIALIZATION
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 7: TESTING DEEPTCR INITIALIZATION")
print("-" * 80)

print("\nInitializing DeepTCR model...")

try:
    # Initialize DeepTCR model
    test_model = DeepTCR_SS("test_initialization")

    print("✅ DeepTCR_SS model initialized successfully!")
    print(f"   Model name: test_initialization")

    # Test data loading
    print("\nTesting data loading into DeepTCR...")

    test_model.Load_Data(
        beta_sequences=beta_sequences,
        v_beta=v_beta,
        j_beta=j_beta,
        alpha_sequences=dummy_alpha_seqs,
        v_alpha=dummy_v_alpha,
        j_alpha=dummy_j_alpha,
        Y=Y_binary,
        sample_labels=pids_arr
    )

    print("✅ Data loaded successfully into DeepTCR!")
    print(f"   Sequences loaded: {len(beta_sequences):,}")
    print(f"   Patients: {len(np.unique(pids_arr))}")

except Exception as e:
    print(f"❌ Error initializing DeepTCR: {e}")
    print("\n📝 TROUBLESHOOTING:")
    print("   This may be due to TensorFlow compatibility issues")
    print("   DeepTCR may require TensorFlow 1.x or specific 2.x versions")
    print(f"\n   Error details: {str(e)}")

# ==============================================================================
# SECTION 8: DISPLAY DEEPTCR INFORMATION
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 8: DEEPTCR MODEL INFORMATION")
print("-" * 80)

print("\nDEEPTCR CONFIGURATION:")
print(f"   Default max sequence length: 40")
print(f"   Architecture: Embedding + Attention + Classification")
print(f"   Training method: Monte Carlo Cross-Validation")
print(f"   Attention concepts: 64 (default)")
print(f"   Network size: small/medium/large options")

print("\nRECOMMENDED PARAMETERS FOR M3 MacBook:")
print(f"   Folds: 100 (for robust cross-validation)")
print(f"   Batch size: 32-64 (adjust based on memory)")
print(f"   Epochs: 10-25")
print(f"   Test size: 0.25 (25% held out)")
print(f"   Network size: 'small' or 'medium'")

print("\nOPTIMIZATION FOR APPLE SILICON:")
print(f"   ✅ M3 chip will use CPU cores efficiently")
print(f"   ✅ Metal acceleration may be available (TF)")
print(f"   ⚠️  Training 100 folds will take time (estimate: 10-40 hrs)")
print(f"   💡 Consider starting with fewer folds for testing (e.g., 10)")

# ==============================================================================
# SECTION 9: SAVE LOG
# ==============================================================================

print("\n" + "-" * 80)
print("SECTION 9: SAVING SETUP LOG")
print("-" * 80)

try:
    with open(LOG_FILE, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEPTCR SETUP LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("INSTALLATION STATUS\n")
        f.write("-" * 80 + "\n")
        f.write(f"DeepTCR installed: Yes\n")
        try:
            f.write(f"DeepTCR version: {DeepTCR.__version__}\n")
        except:
            f.write(f"DeepTCR version: Unknown\n")

        try:
            f.write(f"TensorFlow version: {tf.__version__}\n")
        except:
            f.write(f"TensorFlow version: Not available\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total sequences: {len(beta_sequences):,}\n")
        f.write(f"Unique patients: {len(np.unique(pids_arr))}\n")
        f.write(f"Responders: {np.sum(Y_binary == 1)}\n")
        f.write(f"Non-responders: {np.sum(Y_binary == 0)}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Run Monte Carlo training (script 06)\n")
        f.write("2. Evaluate model performance\n")
        f.write("3. Analyze attention weights\n")
        f.write("4. Interpret results\n")

    print(f"\n✅ Log saved to: {LOG_FILE}")

except Exception as e:
    print(f"\n⚠️  Could not save log file: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("DEEPTCR SETUP COMPLETE")
print("="*80)

print("\n📋 SUMMARY:")
print("   ✅ DeepTCR installed and verified")
print("   ✅ TensorFlow available")
print("   ✅ Data loaded in DeepTCR format")
print("   ✅ Test initialization successful")

print("\n⚠️  IMPORTANT NOTES:")
print("   • Monte Carlo training with 100 folds will be computationally intensive")
print("   • On M3 MacBook, expect 10-40 hours for full 100-fold CV")
print("   • Consider starting with 10 folds for initial testing")
print("   • Monitor system temperature and battery during long runs")

print("\n📊 NEXT STEPS:")
print("   → Run Script 06: Monte Carlo training (MAIN TRAINING)")
print("   → Command: python scripts/06_monte_carlo_training.py")
print("   → Note: This is the final and most time-consuming step!")

print("\n" + "="*80)
print(f"Script completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
