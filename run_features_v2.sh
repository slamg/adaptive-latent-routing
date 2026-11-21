#!/usr/bin/env bash
# 文件作用：启动扩充数据的特征补采和历史状态对照。
# 主要职责：配置离线模型与合并数据路径，调用expanded_feature_comparison.py执行指定阶段。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
python -u expanded_feature_comparison.py "${1:-plan}" \
  --merged_folder /root/autodl-tmp/router_data/gsm8k_merged_v2 \
  --output_folder /root/autodl-tmp/router_diagnostics/features_v2 \
  --model_name /root/autodl-tmp/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --local_files_only "${@:2}"
