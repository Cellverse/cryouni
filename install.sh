#!/bin/bash
set -e

env_name="${1:-cryouni}"

echo "[1/5] Creating conda env '$env_name'..."
conda create -y -n "$env_name" python=3.11

echo "[2/5] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_name"

echo "[3/5] Installing RAPIDS..."
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.10.*" "dask-cudf-cu12==25.10.*" "cuml-cu12==25.10.*" \
    "cugraph-cu12==25.10.*" "nx-cugraph-cu12==25.10.*" "cuxfilter-cu12==25.10.*" \
    "cucim-cu12==25.10.*" "pylibraft-cu12==25.10.*" "raft-dask-cu12==25.10.*" \
    "cuvs-cu12==25.10.*" "nx-cugraph-cu12==25.10.*"

echo "[4/5] Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

echo "[5/5] Installing project requirements..."
pip install -r requirements.txt

echo "[✓] Setup complete."
