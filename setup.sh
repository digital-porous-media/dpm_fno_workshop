#!/bin/bash

# ------------------------------
# 🚀 Setup Script for Workshop
# ------------------------------

module load python3

echo ""
echo "Creating virtual environment..."
python3 -m venv .fno_venv
source $WORK/.fno_venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# ------------------------------
# 🧠 Choose your torch version
# ------------------------------

# Uncomment the one you want:

# CPU-only version
# TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"


# GPU (CUDA 12.1)
TORCH_INSTALL="pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"

echo ""
echo "📦 Installing PyTorch..."
eval $TORCH_INSTALL

echo ""
echo "📦 Installing other Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Downloading Mean Curvature Flow Data"
gdown https://drive.google.com/uc?id=1n9nNCTWsaF2BMElcHpNkDupIO1f_lCOX -O "./2_mean_curvature_flow/mc_flow_data.h5"
echo ""
echo "✅ Setup complete!"
echo "💡 Run 'source .fno_venv/bin/activate' to enter the environment."
