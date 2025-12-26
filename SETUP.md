# Environment Setup Guide

This guide covers setting up the DeepTCR analysis environment and running the complete pipeline.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Lambda Labs H100 Setup](#lambda-labs-h100-gpu-setup)
3. [Running the Pipeline](#running-the-pipeline)
4. [Post-Training Analysis](#post-training-analysis-scripts-07-13)
5. [Expected Performance](#expected-performance)
6. [Troubleshooting](#troubleshooting)
7. [File Sizes](#file-sizes)

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

# Run the complete pipeline
# Phase 1: Training
python scripts/01_environment_setup.py
python scripts/02_data_loading.py
python scripts/03_exploratory_analysis.py
python scripts/04_feature_encoding.py
python scripts/05_deeptcr_setup.py
./run_training.sh  # or: python scripts/06_monte_carlo_training.py

# Phase 2: Post-Training Analysis
python scripts/07_post_training_analysis.py
python scripts/08_attention_weight_extraction.py
python scripts/09_attention_visualization.py
python scripts/10_responder_comparison.py
python scripts/11_top_predictive_sequences.py
python scripts/12_sequence_characteristics.py
python scripts/13_generate_presentation.py
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

## Running the Pipeline

### Phase 1: Training Pipeline (Scripts 01-06)

| Script | Description | Time |
|--------|-------------|------|
| `01_environment_setup.py` | Verify Python, TensorFlow, CUDA | ~10 sec |
| `02_data_loading.py` | Load and preprocess TCR data | ~30 sec |
| `03_exploratory_analysis.py` | Generate EDA visualizations | ~1 min |
| `04_feature_encoding.py` | One-hot encode sequences | ~2 min |
| `05_deeptcr_setup.py` | Install and test DeepTCR | ~1 min |
| `06_monte_carlo_training.py` | 100-fold Monte Carlo CV | ~35 min |

**Total Phase 1:** ~40 minutes on H100 GPU

### Phase 2: Post-Training Analysis (Scripts 07-13)

| Script | Description | Time |
|--------|-------------|------|
| `07_post_training_analysis.py` | Extract AUC, bootstrap CI (1000 iterations) | ~35 min |
| `08_attention_weight_extraction.py` | Extract attention weights from models | ~2 min |
| `09_attention_visualization.py` | Generate attention visualizations | ~30 sec |
| `10_responder_comparison.py` | Statistical comparison R vs NR | ~30 sec |
| `11_top_predictive_sequences.py` | Identify top 100 predictive sequences | ~30 sec |
| `12_sequence_characteristics.py` | Analyze amino acid composition | ~30 sec |
| `13_generate_presentation.py` | Generate PowerPoint presentation | ~10 sec |

**Total Phase 2:** ~40 minutes (mostly script 07 re-training for AUC)

---

## Post-Training Analysis (Scripts 07-13)

### Script 07: Post-Training Analysis

Extracts AUC values and performs bootstrap validation:

```bash
python scripts/07_post_training_analysis.py
```

**Outputs:**
- `results/auc_values.npy` - Array of 100 AUC scores
- `results/bootstrap_results.csv` - Mean, SD, 95% CI, 99% CI
- `results/per_fold_auc.csv` - Per-fold breakdown

### Script 08: Attention Weight Extraction

Extracts attention weights from trained models:

```bash
python scripts/08_attention_weight_extraction.py
```

**Outputs:**
- `results/attention_weights_all.csv` - All 239,634 sequences with attention
- `results/attention_weights_summary.csv` - Summary statistics
- `results/high_attention_sequences.csv` - Top 1% sequences
- `results/top_100_sequences.csv` - Top 100 by attention

### Script 09: Attention Visualization

Generates publication-quality attention figures:

```bash
python scripts/09_attention_visualization.py
```

**Outputs:**
- `figures/paper_final/figureS5_attention_distribution.png`
- `figures/paper_final/figureS6_attention_heatmap.png`
- Additional attention visualizations

### Script 10: Responder Comparison

Statistical comparison of responders vs non-responders:

```bash
python scripts/10_responder_comparison.py
```

**Outputs:**
- `results/responder_comparison_stats.csv` - Mann-Whitney, KS test
- `results/vgene_comparison.csv` - V-gene usage by group
- `results/jgene_comparison.csv` - J-gene usage by group
- `figures/paper_final/figureS8_responder_comparison.png`

### Script 11: Top Predictive Sequences

Identifies and analyzes top predictive sequences:

```bash
python scripts/11_top_predictive_sequences.py
```

**Outputs:**
- `results/top_100_sequences_detailed.csv` - Full details
- `results/vgene_enrichment.csv` - V-gene fold enrichment
- `results/jgene_enrichment.csv` - J-gene fold enrichment
- `figures/paper_final/figureS7_top_sequences.png`

### Script 12: Sequence Characteristics

Analyzes amino acid composition and CDR3 features:

```bash
python scripts/12_sequence_characteristics.py
```

**Outputs:**
- `results/cdr3_length_stats.csv` - Length statistics
- `results/amino_acid_composition.csv` - AA enrichment
- `results/vj_pairing_high_attention.csv` - V-J pairings
- `figures/paper_final/figureS9_sequence_characteristics.png`

### Script 13: Generate Presentation

Creates a 14-slide PowerPoint presentation:

```bash
python scripts/13_generate_presentation.py
```

**Outputs:**
- `results/DeepTCR_Results_Presentation.pptx`

---

## Expected Performance

| Hardware | Time per Fold | 100 Folds Total |
|----------|---------------|-----------------|
| NVIDIA H100 80GB | ~21 seconds | ~35 minutes |
| NVIDIA A100 40GB | ~45 seconds | ~75 minutes |
| CPU only | ~7 minutes | ~12 hours |

### Key Results

| Metric | Value |
|--------|-------|
| Mean AUC | 0.754 ± 0.035 |
| 95% CI | 0.747 - 0.761 |
| High-attention sequences | 2,379 (top 1%) |
| Top 100 from responders | 72% |

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

### "ModuleNotFoundError: No module named 'DeepTCR'"
```bash
pip install DeepTCR==2.1.29
```

### "python-pptx not found"
```bash
pip install python-pptx>=0.6.21
```

### LaTeX compilation errors
Install LaTeX for paper compilation:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## File Sizes

Large files stored via Git LFS:

| File | Size |
|------|------|
| `data_processed/X_onehot.npy` | 789 MB |
| `DeepTCR_MonteCarlo_*/alpha_features.pkl` | 120 MB |
| `DeepTCR_MonteCarlo_*/beta_features.pkl` | 120 MB |
| `results/attention_weights_all.csv` | 27 MB |

**Total repository size:** ~1.2 GB

---

## Dependencies

### Python Packages

```
tensorflow==2.12.0
keras==2.12.0
DeepTCR==2.1.29
numpy==1.23.5
pandas>=1.5.3
scikit-learn>=1.2.2
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
python-pptx>=0.6.21
```

### CUDA Libraries (for GPU)

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

*Last updated: December 26, 2025*
