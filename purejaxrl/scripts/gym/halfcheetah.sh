#!/bin/bash
# 

source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
conda activate brax

module load cuDNN/8.6.0.163-CUDA-11.8.0

python3 ours.py --env_name=halfcheetah

wait
