# 第一批新增500道独立训练题

本补丁只新增两个运行文件，不覆盖模型、Router、数据生成或训练代码。

流程：从旧selection.json未选中的GSM8K train题随机抽500题，冻结名单；
复用原screen脚本跑(8,8,8)，对全部失败题和随机20道成功题复用原collect脚本做64组合搜索。
正确简单题保留主轨迹标签，完整搜索题保留原有主轨迹和21前缀记录；失败状态仍由原逻辑排除。
不会自动合并旧数据或改动旧训练/验证划分，不采集新的18维特征，不训练Router。

上传解压到项目根目录后，激活原conda环境。在screen会话内运行：

```bash
bash run_expand_500.sh plan
set -o pipefail
mkdir -p /root/autodl-tmp/router_logs
bash run_expand_500.sh all 2>&1 | tee -a /root/autodl-tmp/router_logs/gsm8k_expansion_500_v1.log
```

plan不加载模型、不消耗GPU；all自动依次运行筛选和完整搜索。
也可分别使用screen、full子命令。重启后用同一个all命令续跑。
复用原脚本按题断点：完整保存的题跳过，未完成题从头运行。
原脚本并非原子写入pt；若断电导致某个pt无法读取，保留错误信息，确认具体文件后单独处理，不要删除整批数据。
不要同时启动两份相同输出目录的任务。

输出目录：/root/autodl-tmp/router_data/gsm8k_expansion_500_v1

- expansion_plan.json：新增题目、排除题目、参数、代码摘要。
- screen/summary.json：筛选情况。
- screen/samples/：筛选原始记录。
- exhaustive/samples/：完整搜索原始记录。

后续批次必须把此前所有screen目录传给--previous_screen，并使用新的output_root。
当前从旧selection.json取得数据集规模并复用原缓存；应保持相同GSM8K版本。
首次运行参数固定为latent_steps=20、max_new_tokens=2048、sequential、seed=42、pca_rank=8、sink_size=4。
采集不根据某个模型预测错误挑选验证题；新增来源后续在特征比较前按题冻结划分，旧来源归属保留。

500道筛选题不等于500道有效困难题。全部预算都失败的题不能生成成功最优标签。
本地已测试排除抽样和参数冻结/续跑调度；真实模型推理仍需AutoDL验证。
