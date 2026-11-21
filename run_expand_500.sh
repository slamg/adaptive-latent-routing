#!/usr/bin/env bash
# 文件作用：启动新增500道题的数据扩充流程。
# 主要职责：设置离线缓存与采样参数，调用expand_router_data.py组织选题、初筛及完整搜索。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
python -u expand_router_data.py "${1:-all}" \
  --previous_screen /root/autodl-tmp/router_data/gsm8k_screen_500 \
  --output_root /root/autodl-tmp/router_data/gsm8k_expansion_500_v1 \
  --count 500 --sample_seed 20261108 --easy_full_count 20 \
  --model_name /root/autodl-tmp/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
