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

if [[ "$CLUSTER" == "ls6" ]] || [[ "$CLUSTER" == "vista" ]]; then
    module load python3

    echo ""
    echo "Creating virtual environment..."
    python3 -m venv $WORK/.fno_venv
    source $WORK/.fno_venv/bin/activate
    echo ""
    echo "Upgrading pip..."
    pip3 install --upgrade pip3

    # ---- Install PyTorch on LS6 ----
    if [[ "$CLUSTER" == "ls6" ]]; then
        echo ""
        echo "Installing PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi

    # ---- Get Mean Curvature Flow data ----
    echo ""
    echo "Getting Mean Curvature Flow Data"
    cp '/scratch/08780/cedar996/lbfoam/level_set/mc_flow_data.h5' ./2_mean_curvature_flow/mc_flow_data.h5
fi

if [[ "$CLUSTER"=="Colab" ]]; then
    echo ""
    echo "Getting Mean Curvature Flow Data"
    gdown https://drive.google.com/uc?id=1n9nNCTWsaF2BMElcHpNkDupIO1f_lCOX -O "./2_mean_curvature_flow/mc_flow_data.h5"
fi

echo ""
echo "📦 Installing other Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "💡 Run 'source .fno_venv/bin/activate' to enter the environment."
