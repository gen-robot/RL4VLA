cd openvla_oft

cuda="1,2,3,4"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda$ \
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 2000 \
  --eval_steps 50 \
  --save_steps "0,500,1000,1500,2000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --unnorm_key "bridge_orig" \
  --wandb_project "RLVLA_oft_sft" \
#   --use_l1_regression False \
#   --use_diffusion False \