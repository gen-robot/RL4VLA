#!/bin/bash

source /nvme_data/zhiyuanma/miniconda3/bin/activate rlvla
# 切换到工作目录
cd SimplerEnv

# 定义变量
unnorm_key="bridge_orig"
vla_load_path="../SimplerEnv/wandb/run-20250614_062442-cert5non/glob/steps_389"

# 遍历种子和环境 ID
for seed in 0 1 2; do
    for env_id in \
        "PutOnPlateInScene25VisionImage-v1" \
        "PutOnPlateInScene25VisionTexture03-v1" \
        "PutOnPlateInScene25VisionTexture05-v1" \
        "PutOnPlateInScene25VisionWhole03-v1" \
        "PutOnPlateInScene25VisionWhole05-v1" \
        "PutOnPlateInScene25Carrot-v1" \
        "PutOnPlateInScene25Plate-v1" \
        "PutOnPlateInScene25Instruct-v1" \
        "PutOnPlateInScene25MultiCarrot-v1" \
        "PutOnPlateInScene25MultiPlate-v1" \
        "PutOnPlateInScene25Position-v1" \
        "PutOnPlateInScene25EEPose-v1" \
        "PutOnPlateInScene25PositionChangeTo-v1"; do
    
        ckpt_path="../openvla/checkpoints/warmup/steps_2000/merged_002000"
        CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python simpler_env/train_ms3_ppo.py \
            --vla_path="${ckpt_path}" \
            --vla_unnorm_key="${unnorm_key}" \
            --vla_load_path="${vla_load_path}" \
            --env_id="${env_id}" \
            --seed=${seed} \
            --buffer_inferbatch=64 \
            --no_wandb \
            --only_render
    done
done