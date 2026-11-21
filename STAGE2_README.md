# 第二张表 小规模标签复核与新旧特征对照

仅新增独立脚本，不覆盖原代码、原数据和已有权重。全部计算仍以「答对且总预算最低」为目标。

## 实验规模

- 32道独立题，训练24道（easy/hard各12），验证8道（各4）。沿用第一轮的来源题归属，不将旧训练题移动到验证侧。
- easy/hard按原B8是否答对区分，hard只包含完整搜索可救回、有标签的题。easy优先选已有完整搜索的题，便于复核。
- 从上述集合四个分层各取2道，共8道，重新完整运行64种预算组合。比较64次正确性、全局最低预算、完整并列最优轨迹和全部前缀状态标签。任一标签检查失败，停止下一阶段，先分析原因。
- 32题做特征补采：保留原有效标签，只重放预算前缀到目标通信前，停止后续推理，不运行Judger。叶前缀也产生父前缀，避免重复补采。
- 新旧最后一步hidden必须逐状态相差不超过1e-3，重复前缀的新特征也检查一致性；不一致停止，不将旧标签强行用于不同状态。

## 三种输入

1. `last_hidden`：最后latent步hidden，原输入。
2. `mean_last5`：最后5个latent步骤hidden的均值，维度与原输入相同。
3. `statistics`：18维，包含hidden范数、相邻步骤变化和余弦相似度统计，以及最后一层各KV head在prompt区域的attention熵和top8/16/32/64质量。覆盖整个有效prompt（包括sink），不是HOBF实际保留内容的精确模拟。所有量均在当前预算压缩前获得。

固定同一线性层训练设置和类别权重，做二分类/四分类，三个初始化种子取均值；补充balanced kNN5。高维均值与低维统计分别比较，后者差异包含降维效果。没有根据验证集挑最佳epoch或自动调参。这个小验证集合仍来自开发数据，不是正式test；尤其按题数量很少，不能以状态多认定置信度高。

## 使用

先激活 `/root/autodl-tmp/conda_envs/obf_author_clean_py310`，上传并解压更新包至主项目。

```bash
cd /root/autodl-tmp/When-Less-Latent-adaptive-router
# 1. 只建立固定题目名单，不加载模型
bash run_stage2.sh plan
# 查看 /root/autodl-tmp/router_stage2/v1/plan.json
# 核对原模型、prompt、latent步数、生成长度等参数与脚本一致，再运行后续命令。

# 2. screen内运行8题完整重放，重启相同命令自动续跑完整题
set -o pipefail
bash run_stage2.sh audit --confirm_settings \
  2>&1 | tee -a /root/autodl-tmp/router_logs/stage2_audit.log
# 查看 label_audit.json：passed=true 才进入下一步

# 3. 先补采1题，检查hook和特征复现
bash run_stage2.sh collect --confirm_settings --limit 1

# 4. 继续采完剩下题，已完成题自动跳过
bash run_stage2.sh collect --confirm_settings \
  2>&1 | tee -a /root/autodl-tmp/router_logs/stage2_collect.log

# 5. 训练简单探针并生成第二张表，不更新正式Router
bash run_stage2.sh report \
  2>&1 | tee -a /root/autodl-tmp/router_logs/stage2_report.log
```

结果在 `/root/autodl-tmp/router_stage2/v1/`：

- `plan.json`：固定独立题目与数据指纹；
- `label_audit.json`：8题复核明细；
- `features/`：逐题新增特征；
- `second_table.md`、`second_table.json`：老师要求的对照表和详细诊断。

若plan缺少足够分层来源、复核未通过或补采特征不一致，请保留输出并检查，不手动改passed标记、不混用数据。若审计变化，可以先只汇报标签复核结果；不以更换标签掩盖复现问题。

原数据没有保存完整配置，脚本参数必须对照旧运行命令核对。新输入在复核题上准确并不能保证其他题标签稳定；抽样通过只是有限证据。
