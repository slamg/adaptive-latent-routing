#!/usr/bin/env bash
# 文件作用：启动扩充数据上的 Router 训练或完整解题评估。
# 主要职责：固定合并数据、特征及模型路径，将指定任务和参数传给run_router_v2.py。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
python -u run_router_v2.py "$@" \
  --data /root/autodl-tmp/router_data/gsm8k_merged_v2 \
  --features /root/autodl-tmp/router_diagnostics/features_v2 \
  --output /root/autodl-tmp/router_training/router_v2 \
  --model_name /root/autodl-tmp/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --local_files_only
