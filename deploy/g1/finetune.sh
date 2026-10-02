#!/usr/bin/env bash
# Reference x86 CUDA training environment: upstream `uv sync`. No training on the robot.
set -euo pipefail
cd "$(dirname "$0")/../.."
model_dir=$(realpath "${1:?Usage: finetune.sh MODEL_DIR DATA_CONFIG RUN_DIR [full|lora]}")
data_config=$(realpath "${2:?Supply a generated data config}")
run_dir=$(realpath -m "${3:?Supply a new run directory}")
method=${4:-lora}
max_steps=${MAX_TRAIN_STEPS:-200}
warmup_steps=${WARMUP_STEPS:-20}
[[ "$max_steps" =~ ^[1-9][0-9]*$ && "$warmup_steps" =~ ^[0-9]+$ ]] || {
  echo "MAX_TRAIN_STEPS and WARMUP_STEPS must be integer counts" >&2; exit 2;
}
case "$method" in
  full) training_config=unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml ;;
  lora) training_config=unifolm_wla/config/training/mmdit_lora_frozen_vlm.yaml ;;
  *) echo "Expected full or lora" >&2; exit 2 ;;
esac
[[ ! -e "$run_dir" ]] || { echo "Run directory already exists: $run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
export WANDB_MODE=disabled
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Upstream launchers reference a missing deepspeed_zero2.yaml. Supply the actual config explicitly.
.venv/bin/accelerate launch --use_deepspeed --num_processes "${NUM_PROCESSES:-1}" \
  --num_machines 1 --machine_rank 0 --mixed_precision bf16 \
  --deepspeed_config_file unifolm_wla/config/deepseeds/ds_config.yaml \
  unifolm_wla/training/train_unifolm_wla.py --config_yaml "$training_config" \
  --framework.qwenvl.base_vlm "$model_dir/tokenizer" \
  --trainer.pretrained_checkpoint "$model_dir/checkpoints/model.safetensors" \
  --trainer.max_train_steps "$max_steps" --trainer.num_warmup_steps "$warmup_steps" \
  --datasets.vla_data.data_config_path "$data_config" \
  --run_root_dir "$(dirname "$run_dir")" --run_id "$(basename "$run_dir")"
# The projector is nested inside qwen_vl_interface and is frozen by the current upstream config.
# LoRA output needs merging into a base model before this deployment loader can use it.
