#!/bin/bash
# REAL (Robust Extreme Agility Learning) - Installation Script

conda create -n real python=3.8
conda activate real

# PyTorch (adjust CUDA version as needed)
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Isaac Gym - download binaries from https://developer.nvidia.com/isaac-gym
# cd isaacgym/python && pip install -e .

# Install REAL packages
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .

# Additional dependencies
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python flask pymeshlab
