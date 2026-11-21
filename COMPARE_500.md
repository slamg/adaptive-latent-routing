# 同一批500题的固定B32补跑与Oracle对比

新增独立脚本，不修改模型、压缩器、Router训练和原数据。只补固定B32，不扩B16/B64。

## 运行前

在原Conda环境运行：`conda activate /root/autodl-tmp/conda_envs/obf_author_clean_py310`。
将更新包中的文件解压到 `/root/autodl-tmp/When-Less-Latent-adaptive-router`。
旧数据缺少完整推理参数，先确认与原实验一致：Qwen3-4B相同snapshot、sequential、latent_steps=20、max_new_tokens=2048、pca_rank=8、sink_size=4、seed=42，未开启think和latent_space_realign。若历史命令不同，先调整shell脚本参数，不能强行混合运行。

## 三步运行

```bash
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
# 1. 不加载模型，只核对名单、题目身份、完整搜索与B8重叠结果
bash run_compare_500.sh plan
# 预期：questions=500 reused_b32=76 missing_b32=424 oracle_success=483

# 2. 原配置核对一致后，先补1题检查推理能否正常运行
bash run_compare_500.sh run --confirm_settings --limit 1

# 3. screen内继续运行；自动跳过刚才成功完成的1题及已有76题
mkdir -p /root/autodl-tmp/router_logs
set -o pipefail
bash run_compare_500.sh run --confirm_settings \
  2>&1 | tee -a /root/autodl-tmp/router_logs/fixed32_comparison_500.log
```

补跑逐题保存为JSON；中断后重复第3条命令继续。写文件采用临时文件完成后替换，未完成题重跑。全部齐全后自动生成 `comparison.md` 和 `comparison.json`。不推理而单独重建表格用 `bash run_compare_500.sh report`。

结果目录：`/root/autodl-tmp/router_data/gsm8k_fixed32_comparison_500`。
新样本在其`samples`子目录；不要将原Oracle样本复制进去。

## 校验与结果口径

原初筛 `selection.json` 是唯一500题名单；原筛选分片须齐全；全部B8失败题须有64组合搜索。Oracle中名单外样本、重复题、题目或gold不一致、B8两次运行答案或KV不一致均停止，不静默跳过。补跑前再核对当前GSM8K相同索引的题目与gold。

主表报告三种策略的答对数、正确率、各自答对题上的平均总预算及KV。另报告B32与Oracle共同答对题上的KV节省。所有失败题没有Oracle成功成本；不把缺失Oracle成本当作0。各行成功题平均值分母不同，不直接相除当节省比例。

该脚本不测Router时延，不训练Router，不执行下一步标签复核或新旧特征对照。重叠B8校验只检出部分不一致，不能替代完整标签复核。
