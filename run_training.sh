#!/bin/bash
# Run DeepTCR training with proper CUDA library paths

export LD_LIBRARY_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/nccl/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cufft/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/curand/lib:/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

cd /lambda/nfs/datasetmedgemma/lambda_deepTCR_deployement

echo "Starting DeepTCR training at $(date)"
echo "LD_LIBRARY_PATH is set for CUDA 11 libraries"
echo ""

python3 scripts/06_monte_carlo_training.py 2>&1 | tee logs/training_gpu.log
