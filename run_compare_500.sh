#!/usr/bin/env bash
# 文件作用：启动原500道题的固定预算与 Oracle 比较流程。
# 主要职责：设置服务器数据、模型和运行参数，调用compare_fixed_oracle_500.py。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
# Run in the original obf_author_clean_py310 environment and screen session.
# Defaults match the historical commands; check those commands before run mode.
python -u compare_fixed_oracle_500.py "${1:-plan}" \
  --screen_folder /root/autodl-tmp/router_data/gsm8k_screen_500 \
  --oracle_folder /root/autodl-tmp/router_data/gsm8k_joint_selected_50 \
  --output_folder /root/autodl-tmp/router_data/gsm8k_fixed32_comparison_500 \
  --model_name /root/autodl-tmp/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --cache_dir /root/autodl-tmp/hf_cache --local_files_only \
  --prompt sequential --latent_steps 20 --max_new_tokens 2048 \
  --pca_rank 8 --sink_size 4 --seed 42 --device cuda "${@:2}"
