#!/bin/bash

# SET CUDA_HOME
# export CUDA_HOME=/mnt/wuxuaner/workspace/miniconda3/envs/torch231
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Prompt for model path
read -p "Enter the model path (default: RewardModels/allenai/tulu-v2.5-dpo-13b-hh-rlhf-60k): " MODEL_PATH
MODEL_PATH=${MODEL_PATH:}

# Prompt for ref model path
# read -p "Enter the ref model path: (default: None, Llama-3-Base-8B-SFT)" ref_model_path
# ref_model_path=${}

# Prompt for CUDA device
read -p "Enter the CUDA device (default: 2,3): " CUDA_DEVICE
CUDA_DEVICE=${CUDA_DEVICE:-2,3}

# Add this line at the top of your run_dpo.sh script
export PYTHONPATH=$PYTHONPATH:pwd
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

chat_template=llama-3
python scripts/run_dpo.py \
    --model $MODEL_PATH \
    --chat_template $chat_template \
    --datapath data/total_dataset.json \
    --batch_size 8 \
    --ref_free_type avg \
    --trust_remote_code
