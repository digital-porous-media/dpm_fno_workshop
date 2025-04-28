#!/bin/bash

# ---- Detect cluster ----
HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == *"ls6"* ]]; then
    CLUSTER="ls6"
elif [[ "$HOSTNAME" == *"vista"* ]]; then
    CLUSTER="vista"
elif [[ -n "$COLAB_GPU" ]] || [[ "$HOME" == "/root" && -d "/content" ]]; then
    CLUSTER="Colab"
else
    echo "Unknown cluster: $HOSTNAME"
    exit 1
fi

echo "Setting up on $CLUSTER"

# ------------------------------
# 🚀 Setup Script for Workshop
# ------------------------------
if [[ "$CLUSTER" == "vista" ]]; then
    module load gcc cuda
    module load python3_mpi
elif [[ "$CLUSTER" == "ls6" ]]; then
    module load python3
fi

if [[ "$CLUSTER" == "ls6" ]] || [[ "$CLUSTER" == "vista" ]]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv $WORK/.fno_venv
    source $WORK/.fno_venv/bin/activate
    echo ""
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip

    # ---- Install PyTorch on LS6 ----
    if [[ "$CLUSTER" == "ls6" ]]; then
        echo ""
        echo "Installing PyTorch..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi

    # ---- Get Mean Curvature Flow data ----
    echo ""
    echo "Getting Mean Curvature Flow Data"
    cp '/work/06898/bchang/mc_flow_data.h5' ./2_mean_curvature_flow/mc_flow_data.h5
fi

if [[ "$CLUSTER" == "Colab" ]]; then
    echo ""
    echo "Getting Mean Curvature Flow Data"
    gdown https://drive.google.com/uc?id=1n9nNCTWsaF2BMElcHpNkDupIO1f_lCOX -O "./2_mean_curvature_flow/mc_flow_data.h5"
fi

echo ""
echo "📦 Installing other Python dependencies..."
python3 -m pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "💡 Run 'source $WORK/.fno_venv/bin/activate' to enter the environment."
