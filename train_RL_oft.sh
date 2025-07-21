#!/bin/bash

# 设置工作目录
cd SimplerEnv

# 设置环境变量
export CUDA_VISIBLE_DEVICES="5,6"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 运行程序
python simpler_env/train_ms3_ppo.py \
    --name=ppo-pc25m_v3-warmup \
    --env_id=PutOnPlateInScene25Main-v3 \
    --vla_path=Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
    --vla_unnorm_key=bridge_orig \
    --vla_load_path=../openvla_oft/checkpoints/warmup/Openvla-oft-SFT-libero-spatial-traj1+warmup+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--2000_chkpt \
    --seed=0