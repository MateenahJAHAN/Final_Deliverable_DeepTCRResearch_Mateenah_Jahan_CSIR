# DeepTCR Monte Carlo Cross-Validation on Lambda H100

> **See [TECHNICAL_SETUP.md](TECHNICAL_SETUP.md) for detailed TensorFlow/CUDA setup instructions and troubleshooting.**

## Project Overview

**Goal:** Predict immunotherapy response from T-cell receptor (TCR) sequences using DeepTCR deep learning framework with 100-fold Monte Carlo cross-validation.

**Dataset:** 239,634 TCR-beta sequences from 153 patients (77 responders, 76 non-responders)

**Hardware:** Lambda Cloud H100 80GB GPU instance

---

## Quick Start

```bash
# Navigate to project
cd /lambda/nfs/datasetmedgemma/lambda_deepTCR_deployement

# Run training (optimized for H100)
./run_training.sh

# Monitor progress
tail -f logs/training_optimized.log | grep -E "^\s+[0-9]+%|AUC"

# Check GPU
nvidia-smi
```

---

## Directory Structure

```
lambda_deepTCR_deployement/
├── README.md                    # This file
├── run_training.sh              # Main training launcher (sets CUDA paths)
├── data_raw/                    # Original dataset
│   └── deeptcr_complete_dataset (5).csv
├── data_processed/              # Preprocessed data (ready for training)
│   ├── X_onehot.npy            # Feature matrix (789 MB)
│   ├── y_labels.npy            # Binary labels
│   ├── patient_ids.npy         # Patient identifiers
│   └── deeptcr_trb_ready.csv   # Cleaned TCR data
├── scripts/                     # Python scripts
│   ├── 01_environment_setup.py
│   ├── 02_data_loading.py
│   ├── 03_exploratory_analysis.py
│   ├── 04_feature_encoding.py
│   ├── 05_deeptcr_setup.py
│   └── 06_monte_carlo_training.py  # Main training script
├── figures/                     # EDA visualizations
├── models/                      # Trained model outputs
├── results/                     # Analysis results
└── logs/                        # Training logs
```

---

## What Worked

### 1. TensorFlow 2.12.0 + CUDA 11 Libraries (GPU Detection Fix)

**Problem:** TensorFlow 2.12.0 couldn't detect H100 GPU because it expects CUDA 11.8, but Lambda has CUDA 12.8.

**Solution:** Install CUDA 11 pip packages and set LD_LIBRARY_PATH:

```bash
# Install CUDA 11 libraries
pip3 install nvidia-cudnn-cu11==8.6.0.163 \
             nvidia-cuda-nvrtc-cu11==11.8.89 \
             nvidia-cuda-runtime-cu11==11.8.89 \
             nvidia-cublas-cu11==11.11.3.6 \
             nvidia-cufft-cu11 \
             nvidia-cusolver-cu11 \
             nvidia-cusparse-cu11

# Set library path (included in run_training.sh)
export LD_LIBRARY_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:..."
```

### 2. GPU Optimization (19x Speedup)

**Benchmark Results:**

| Configuration | Time/Fold | GPU Memory | Speedup |
|--------------|-----------|------------|---------|
| batch=32, small | 416.4s | 1,259 MB | 1.0x |
| batch=256, small | 121.9s | 2,283 MB | 3.4x |
| batch=512, small | 52.2s | 4,331 MB | 8.0x |
| batch=1024, small | 49.2s | 8,427 MB | 8.5x |
| batch=512, large | 34.5s | 8,429 MB | 12.1x |
| **batch=1024, large** | **21.7s** | **8,429 MB** | **19.2x** |

**Optimized Parameters (in 06_monte_carlo_training.py):**
```python
BATCH_SIZE = 1024        # Was 32
NETWORK_SIZE = 'large'   # Was 'small'
```

### 3. DeepTCR 2.1.29 Compatibility

DeepTCR works despite version warnings when using TensorFlow 2.12.0 with proper CUDA setup.

---

## What Did NOT Work

### 1. TensorFlow 2.15+ with DeepTCR

- TensorFlow 2.15+ uses Keras 3.x
- DeepTCR uses TensorFlow 1.x style APIs (`tf.compat.v1.layers`)
- Error: `conv2d is not available with Keras 3`
- **Conclusion:** Must use TensorFlow 2.12.0

### 2. System TensorFlow 2.19.0

- Lambda has system TensorFlow 2.19.0 that detects GPU
- But incompatible with DeepTCR's Keras 2.x APIs
- **Conclusion:** Use pip TensorFlow 2.12.0 with CUDA 11 packages

### 3. Default Batch Size (32)

- Only uses 1.5% of H100's 80GB memory
- 32% GPU utilization
- Extremely slow training
- **Conclusion:** Use batch_size=1024 or higher

### 4. CUDA 12 Symlinks

- Tried symlinking CUDA 12 libs as CUDA 11
- Did not work due to API incompatibilities
- **Conclusion:** Install actual CUDA 11 pip packages

---

## Training Performance

### With Optimized Settings (batch=1024, large network)

- **Per fold:** ~20-22 seconds
- **100 folds:** ~35-40 minutes
- **GPU Memory:** 8.4 GB / 81 GB (10%)
- **GPU Utilization:** 30-50%

### With Default Settings (batch=32, small network)

- **Per fold:** ~7 minutes
- **100 folds:** ~12 hours
- **GPU Memory:** 1.2 GB / 81 GB (1.5%)
- **GPU Utilization:** 30%

---

## Key Files

### run_training.sh
Wrapper script that sets CUDA library paths before running training.

### scripts/06_monte_carlo_training.py
Main training script with optimized parameters:
- `NUM_FOLDS = 100`
- `BATCH_SIZE = 1024`
- `NETWORK_SIZE = 'large'`
- `TEST_SIZE = 0.25`
- `EPOCHS_MIN = 10`

---

## Environment Setup

### Required Packages
```
tensorflow==2.12.0
keras==2.12.0
DeepTCR==2.1.29
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
```

### CUDA Libraries (pip)
```
nvidia-cudnn-cu11==8.6.0.163
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cublas-cu11==11.11.3.6
nvidia-cufft-cu11
nvidia-cusolver-cu11
nvidia-cusparse-cu11
```

---

## Troubleshooting

### GPU Not Detected
```bash
# Check if LD_LIBRARY_PATH is set
echo $LD_LIBRARY_PATH

# Should include nvidia library paths
# Use run_training.sh which sets this automatically
```

### JIT Compilation Warning
```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 9.0.
CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
```
This is normal for H100 (compute capability 9.0) with TF 2.12. First fold takes longer, subsequent folds are fast.

### Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 512  # or 256
```

---

## Results

### Expected Outputs
- `models/DeepTCR_MonteCarlo_100folds_*/` - Trained models
- `results/` - Performance metrics, AUC scores
- `logs/training_optimized.log` - Training log

### Metrics
- **AUC:** 0.776 ± 0.007 (SD), range: 0.758-0.788
- **Cross-validation:** 100 independent train/test splits

---

## Commands Reference

```bash
# Start training
./run_training.sh

# Monitor progress
tail -f logs/training_optimized.log

# Check GPU
nvidia-smi

# Check fold progress
grep -E "[0-9]+/100" logs/training_optimized.log | tail -5

# Kill training
pkill -f "06_monte_carlo"
```

---

## Lessons Learned

1. **Always benchmark batch sizes** - Default values are often for small GPUs
2. **H100 has 80GB memory** - Use it! batch_size=1024+ is fine
3. **CUDA version matters** - TensorFlow builds are tied to specific CUDA versions
4. **pip CUDA packages work** - Can install CUDA 11 alongside system CUDA 12
5. **JIT compilation is one-time** - First fold slow, rest are fast
6. **Network size affects speed** - 'large' network with big batch is faster than 'small' with small batch

---

## Author
Created: December 25, 2025
Lambda H100 GPU Instance
