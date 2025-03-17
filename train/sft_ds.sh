# Reference Running: bash train/sft.sh
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
epochs=5
weight_decay=1e-4
micro_batch_size=1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=true
output_dir="ckpts/m1-${uid}"

# Make sure the output directory exists
mkdir -p ${output_dir}

# Set environment variables that might help NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft.py \
    --block_size=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --num_train_epochs=${epochs} \
    --train_file_path="starlife/m1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --deepspeed="train/ds_config.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${output_dir}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True