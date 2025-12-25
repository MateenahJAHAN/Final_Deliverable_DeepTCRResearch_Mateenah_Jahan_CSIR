# Environment Setup Guide

This guide covers setting up the DeepTCR analysis environment on different systems.

---

## Quick Start (Any System)

```bash
# Clone repository
git clone https://github.com/MateenahJAHAN/Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR.git
cd Final_Deliverable_DeepTCRResearch_Mateenah_Jahan_CSIR

# Install Git LFS and pull large files
git lfs install
git lfs pull

# Install Python dependencies
pip install -r requirements.txt

# Run the pipeline
python scripts/01_environment_setup.py
python scripts/02_data_loading.py
python scripts/03_exploratory_analysis.py
python scripts/04_feature_encoding.py
python scripts/05_deeptcr_setup.py
./run_training.sh  # or: python scripts/06_monte_carlo_training.py
```

---

## Lambda Labs H100 GPU Setup

Lambda Labs instances come with CUDA 12.x, but DeepTCR/TensorFlow 2.12 requires CUDA 11.x libraries.

### Step 1: Install CUDA 11 pip packages

```bash
pip install nvidia-cudnn-cu11==8.6.0.163
pip install nvidia-cuda-nvrtc-cu11==11.8.89
pip install nvidia-cuda-runtime-cu11==11.8.89
pip install nvidia-cublas-cu11==11.11.3.6
pip install nvidia-cufft-cu11
pip install nvidia-cusolver-cu11
pip install nvidia-cusparse-cu11
```

### Step 2: Set Library Path

Add to your `~/.bashrc` or run before training:

```bash
export LD_LIBRARY_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cublas/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cufft/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cusolver/lib:\
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cusparse/lib:\
$LD_LIBRARY_PATH"
```

### Step 3: Verify GPU Detection

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should output: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### Step 4: Run Training

```bash
./run_training.sh
```

Or directly:
```bash
python scripts/06_monte_carlo_training.py
```

---

## Expected Performance

| Hardware | Time per Fold | 100 Folds Total |
|----------|---------------|-----------------|
| NVIDIA H100 80GB | ~21 seconds | ~35 minutes |
| NVIDIA A100 40GB | ~45 seconds | ~75 minutes |
| CPU only | ~7 minutes | ~12 hours |

---

## Troubleshooting

### "No GPU detected"
- Ensure LD_LIBRARY_PATH is set correctly
- Run `nvidia-smi` to verify GPU is available
- Use `run_training.sh` which sets paths automatically

### "CUDA out of memory"
- Reduce batch size in `scripts/06_monte_carlo_training.py`:
  ```python
  BATCH_SIZE = 512  # or 256
  ```

### "JIT compilation warning"
```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 9.0
```
This is normal for H100. First fold is slow (~30s), subsequent folds are fast (~20s).

---

## File Sizes

Large files stored via Git LFS:
- `data_processed/X_onehot.npy` - 789 MB
- `DeepTCR_MonteCarlo_*/alpha_features.pkl` - 120 MB
- `DeepTCR_MonteCarlo_*/beta_features.pkl` - 120 MB

Total repository size: ~1.1 GB

---

*Last updated: December 2025*
