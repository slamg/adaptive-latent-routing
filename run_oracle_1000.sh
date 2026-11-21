#!/usr/bin/env bash
# 文件作用：启动1000道题的固定预算与 Oracle 对照。
# 主要职责：设置服务器目录和离线环境，将命令行参数传给complete_oracle_1000.py。

set -euo pipefail
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE
python -u complete_oracle_1000.py "$@"
