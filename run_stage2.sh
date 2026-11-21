#!/usr/bin/env bash
# 文件作用：启动早期小规模标签复核与特征对照实验。
# 主要职责：设置原500题数据及模型参数，调用stage2_teacher_table.py执行指定阶段。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
python -u stage2_teacher_table.py "${1:-plan}" \
  --screen_folder /root/autodl-tmp/router_data/gsm8k_screen_500 \
  --oracle_folder /root/autodl-tmp/router_data/gsm8k_joint_selected_50 \
  --output_folder /root/autodl-tmp/router_stage2/v1 \
  --model_name /root/autodl-tmp/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --cache_dir /root/autodl-tmp/hf_cache --local_files_only \
  --prompt sequential --latent_steps 20 --max_new_tokens 2048 \
  --pca_rank 8 --sink_size 4 --seed 42 --device cuda "${@:2}"
