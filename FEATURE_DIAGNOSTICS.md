# Router Feature Separability Diagnostics

该诊断只读取现有的 screening/Oracle `.pt` 数据，不重新生成标签、不修改
HOBF，也不更新正式 Router 权重。

## 运行命令

```bash
cd /root/autodl-tmp/When-Less-Latent-adaptive-router

python -u diagnose_router_features.py \
  --screen_folder /root/autodl-tmp/router_data/gsm8k_screen_500 \
  --exhaustive_folder /root/autodl-tmp/router_data/gsm8k_joint_selected_50 \
  --output_folder /root/autodl-tmp/router_diagnostics/feature_sep_v1 \
  --view primary_reserve \
  --budget_choices 8 16 32 64 \
  --validation_fraction 0.2 \
  --split_seed 20261001 \
  --linear_epochs 100 \
  --device cuda \
  2>&1 | tee /root/autodl-tmp/router_logs/feature_sep_v1.log
```

如果实际 exhaustive 文件夹名称不同，只替换 `--exhaustive_folder`。这里必须使用
第一版训练时同一个 screening 文件夹、exhaustive 文件夹和 `split_seed`，才能与已有
Router 指标公平比较。

## 输出文件

- `separability_report.json`：全部详细指标；
- `probe_summary.csv`：各探针的验证集汇总；
- `centroid_distances_*.csv`：类别中心、类内半径与归一化分离度；
- `pca_*.csv`：PCA 坐标；
- `pca_*.png`：如果环境中安装了 matplotlib，则额外输出散点图。

诊断分别运行：

1. 四分类 `B=8/16/32/64`；
2. 二分类“B=8 足够 / 必须 B>8”；
3. overall；
4. handoff 0、1、2。

探针包括：

- nearest centroid；
- plain kNN；
- 类别先验校正后的 balanced kNN；
- 带类别平衡的线性 probe。

## 优先看什么

首先看 `binary_b8_sufficient`，因为它直接回答两阶段 Router 的第一阶段是否可行。
不要只看普通 accuracy；当前多数类基线很高，应重点看：

- `macro_recall` / `balanced_accuracy`；
- `larger_budget_recall`；
- B>8 precision；
- confusion matrix；
- 多种简单 probe 的结果是否一致；
- overall 与各 handoff 是否存在明显差异。

结果判断原则：

- 若多种简单探针在未见过的 source-level validation 上都明显优于 50% 的二分类
  balanced accuracy，并能稳定召回 B>8，说明现有特征含有可用信号，可以继续设计
  两阶段 Router；
- 若训练指标很高、验证指标接近随机，说明特征容易记忆训练题，但缺乏跨题泛化；
- 若训练和验证均接近随机，说明当前 `last_hidden` 特征很可能不够，需要增加新的
  uncertainty、attention 或 latent statistics 特征；
- 若只有某个 handoff 可分，应考虑为不同 handoff 使用独立头，而不是假定三者共享
  完全相同的决策边界。

PCA 只用于直观观察，不能单凭二维图重叠就断言高维特征不可分；最终以 source-level
验证集上的简单 probe 指标为主要依据。
