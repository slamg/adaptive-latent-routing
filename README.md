# 动态通信预算 Router：代码审查阅读指南

> **本仓库用于向老师展示当前实验实现，方便阅读、追踪数据流和查找问题。**
> 当前对象是本地整合的 **debug 版**，保留实验迭代形成的文件组织和实现方式。本次首页整理只改文档，没有移动源码、重构函数或修改算法。
>
> **debug 不表示已经排除全部问题。** 此前已经修正部分评分、预算约定和选模问题，但没有据此重新完成真实 Qwen 模型的数据生成、训练和 193 题完整解题实验。历史结果不能直接视为当前 debug 版的验证结果。

更新日期：2026-11-21。本文在此前《新增代码文件说明与阅读指南》的基础上，结合当前文件和 debug 修正记录更新。整理前的 README 原文保存在 [README_历史说明.md](README_历史说明.md)，其中同时包含原项目介绍和早期追加的 Router 实验说明；旧说明中的阶段性要求不应直接当成当前实验配置。

## 1. 这个实验在做什么，目前想请老师检查什么

基础项目研究多 Agent 之间的 KV 缓存传递与压缩。我们在其 sequential LatentMAS + HOBF 路径上，增加一个小型 Router 网络：在当前 Agent 完成 latent 推理、压缩器开始工作之前，读取状态特征，选择本次通信预算 **B ∈ {8, 16, 32, 64}**，再把 B 传给原压缩器。

这里的 **B 是压缩器中当前提示词 KV 的保留预算**，不是压缩比例，也不等于传给下一 Agent 的全部 KV 长度。历史缓存、sink 和 latent 状态等也会影响最终缓存及通信量。我们沿用原有 Agent 推理与 HOBF 选择/补偿的主要计算过程，增加特征返回、预算选择、参数传递和记录。

此前主要观察到：最后一步 hidden 的 Router 在训练状态上表现较好，在验证状态上的非 B8 识别较弱；接入完整解题后，也没有体现出超过固定 B8 的可靠收益。统计特征 Router 则出现明显偏向 B8 的现象，不能笼统认为所有网络都已经在训练集学好。

围绕这一问题，已经做过扩充题目、固定来源划分、标签抽查、新旧特征对照、重新训练及完整解题对照。仍在讨论的原因包括：非 B8 状态少、类别不平衡、非 B8 独立来源题少、输入特征判别信息不足，以及训练/选模/网络设计或代码实现问题。**这些是待区分的假设，不是已经确定的唯一根因。** 希望老师重点检查“标签是否合理、训练与选模是否相容、文件之间是否正确衔接、在线评估是否可信”。

## 2. 先从哪里读：三块主线

建议先看下面这张地图，再按感兴趣的部分进入文件。每个文件顶部已有简体中文用途说明；相对原作者基础版本新增的函数/方法已有“功能、输入、输出”注释。注释用于帮助理解当前实现，不能替代对代码正确性的检查。

| 主线 | 需要回答的问题 | 推荐阅读顺序 |
|---|---|---|
| **一：制作数据集** | 哪些题被运行？预算标签怎样产生？状态怎样整理、划分？ | [筛选](screen_router_candidates.py) → [完整搜索](collect_router_data.py) → [Oracle 标签函数](budget_router.py) → [训练记录整理](router_training_data.py) → [新旧数据合并](prepare_expanded_dataset.py) |
| **二：搭建、训练和验证 Router** | 网络输入输出是什么？损失、平衡权重、选模怎样计算？ | [BudgetRouter 类](budget_router.py) → [训练程序](train_budget_router.py) → [扩充实验入口的 train](run_router_v2.py) |
| **三：把动态 B 接入完整推理** | Router 在哪里决定 B？预算是否真正影响压缩？最终指标怎样统计？ | [LatentMAS 流程](methods/latent_mas.py) → [HOBF 压缩器](compression_methods/HOBF.py) → [模型特征返回](models.py) → [完整题目评估](run_router_v2.py)；通用命令行入口另看 [run.py](run.py) |

```text
题目抽样 → 最小预算初筛 → 对选中题完整搜索预算组合
                                  ↓
                   Oracle：由最终正确性和预算成本构建标签
                                  ↓
          状态记录整理、去重 → 按独立题目固定训练/验证划分
                                  ↓
          状态特征 + 通信阶段 → 训练 Router → 保存最佳权重
                                  ↓
  完整解题：Agent latent 推理 → 当前状态特征 → Router 选择 B
                                  ↓
                     HOBF 压缩 → 下一 Agent → 最终答案
                                  ↓
               同题比较正确率、KV 张量量和端到端时延
```

**制作标签时，预算由搜索脚本指定；在线解题时，预算由训练好的 Router 预测。** 在线 Router 不读取该题的 Oracle 标签来作决定，也不会在每次通信时重新执行 64 组合搜索。

### 一：数据生成、标签与整理

| 文件 | 当前职责 | 建议重点看什么 |
|---|---|---|
| [router_data_selection.py](router_data_selection.py) | 题目抽样、索引文件读写、完整搜索选题 | 索引指原数据集位置；失败题与成功抽查题怎样选取 |
| [screen_router_candidates.py](screen_router_candidates.py) | 用候选中的最小预算初筛；成功题生成主轨迹标签；输出完整搜索名单 | `make_primary_examples()`、`main()`；默认最小轨迹为 `(8,8,8)` |
| [collect_router_data.py](collect_router_data.py) | 执行选中题的全部预算组合，保存答案、正确性、通信量、状态特征和标签 | `main()`、`collect_state_features()`；预算前缀如何与特征对应 |
| [budget_router.py](budget_router.py) 中的 Oracle 函数 | 枚举组合；选择最低成功总预算；保存并列最优；构建主轨迹与备用状态标签 | `select_global_optimal_schedules()`、`build_primary_examples()`、`build_reserve_state_examples()` |
| [expand_router_data.py](expand_router_data.py) | 排除旧题、冻结新题名单，调度初筛和完整搜索 | `choose_new()`、`main()`；本文件复用前面的生成脚本 |
| [router_training_data.py](router_training_data.py) | 统一记录字段、选择主/备用视图、去重、过滤无成功后继状态；封装 Dataset 和来源划分 | `build_router_examples()`、`_normalise_example()`、`split_source_ids()`、`RouterFeatureDataset` |
| [prepare_expanded_dataset.py](prepare_expanded_dataset.py) | 检查新旧题重复和搜索覆盖，保留旧划分，为新题分配归属并合并 | `read_complete_batch()`、`assign_new_sources()`、`main()`；输出 `merged_dataset.pt` 和 `source_split.json` |
| [audit_router_sources.py](audit_router_sources.py) | 统计独立题目、历史状态、非 B8 来源和状态集中程度 | `summarize()`；数据盘点不能自行证明训练数据已经充分 |

标签口径需要区分：

- 四个预算、三次选择构成 **4³=64 种完整轨迹**。只有进入完整搜索的题才执行这一流程；初筛成功题可以凭最小预算轨迹得到主轨迹标签。哪些失败题被送入完整搜索由选题参数决定，不能把任意一次筛选都理解成“所有失败题一定搜索过”。
- **整题 Oracle**：从最终答对的轨迹中，选择 `B0+B1+B2` 最小的集合；保留并列最优，并选定一条代表轨迹生成三个主状态样本。
- **备用状态 Oracle**：固定已经走过的预算前缀，再考虑从当前状态继续答对所需的最低未来成本；完整搜索共有 `1+4+16=21` 个决策前前缀。没有成功后继的状态不提供成功预算标签。
- **非 B8 状态**：最优动作集合不包含 B8。它不等同于“整道题用 `(8,8,8)` 答错”。
- 数据划分单位是**独立题目**；一题产生的全部历史状态应在训练或验证的同一侧。扩充实验沿用冻结的 `source_split.json`，不能再按状态随机拆分。

历史扩充实验处理了原 500 题和新增 500 题，其中 966 道题有成功方案，按来源分成 773 道训练题和 193 道开发验证题。对应有效历史状态为训练 3515 条、验证 951 条，共 4466 条。题目数和状态数是两种统计单位；这些数量描述历史数据版本，评分修正后应重新核对受影响的标签和记录。

### 二：网络定义、训练与历史状态验证

| 文件/位置 | 当前职责 | 主要输入与输出 |
|---|---|---|
| [budget_router.py](budget_router.py)：`BudgetRouter` | 对状态特征做 LayerNorm，与通信阶段 Embedding 拼接，再经过小型分类网络 | 输入 `[批量,特征维度]` 和阶段编号；输出每个候选预算的 logits，`select_budget()` 返回分数、类别索引和实际预算 |
| [train_budget_router.py](train_budget_router.py)：`multi_positive_losses()` | 接受并列最优动作集合中的任意动作 | 输出 `-log(最优集合内预测概率之和)`，不是只针对代表标签的单标签交叉熵 |
| [train_budget_router.py](train_budget_router.py)：权重及主训练循环 | 根据训练记录计算类别/阶段权重，反向传播，验证，早停和保存 | 输入已整理的状态记录；输出 `best_router.pt`、`history.csv`、`setup.json`、`metrics.json` |
| [train_budget_router.py](train_budget_router.py)：`evaluate()`、`selection_score()` | 在已保存的历史状态上预测预算并选模 | 当前 debug 版优先采用按最低预算可用/不可用分组的集合兼容准确率；具体修正见第 5 节 |
| [run_router_v2.py](run_router_v2.py)：`load_data()`、`train()` | 将这次扩充数据、固定划分和所选输入交给训练程序 | 读取合并数据及补采特征，训练后为权重追加数据、特征计划和划分信息 |

`budget_router.py` 同时放了网络定义和标签函数，这是当前真实的文件组织。读网络时先看 `BudgetRouter`；读数据制作时再看该文件后面的 Oracle 函数。

**当前扩充训练入口的实现保留原样：** `run_router_v2.train()` 会替换 `train_budget_router` 模块的数据读取/划分函数，并修改 `sys.argv` 后调用其 `main()`；这些进程内替换没有在返回时恢复。独立进程退出后状态结束，同一进程继续调用时需要关注影响。它还固定检查 773/193 道来源题和 4466 条状态，是本轮实验入口，不能直接当作任意规模通用接口。

只读基本的最后一步 hidden 训练逻辑，可以先看 `budget_router.py`、`router_training_data.py`、`train_budget_router.py`。但**当前 v2 入口即使选择 `last_hidden`，仍会读取补采特征计划和逐题特征文件**，并核对旧 hidden 的一致性；仓库没有为了首页展示而去掉这些依赖。

### 三：动态预算接入与完整题目评估

| 文件 | 当前职责 | 建议重点看什么 |
|---|---|---|
| [models.py](models.py) | 封装基础模型推理；增加本地加载参数及最后一步 hidden 返回 | `generate_latent_batch()` 的返回值，尤其新增的 Router 特征；不是在此训练 Router |
| [methods/latent_mas.py](methods/latent_mas.py) | 组织 Agent 流程；在 latent 推理后、压缩前选择预算 | fixed/random/router 和指定预算轨迹的选择；传给压缩器的是实际 B，不是类别下标 |
| [compression_methods/HOBF.py](compression_methods/HOBF.py) | 执行 HOBF 压缩，接受运行时 `kv_budget` | `compress()` 中运行时预算怎样参与实际保留数量；其余 KV 部分如何保留 |
| [compression_methods/Headwise.py](compression_methods/Headwise.py) | 同步支持运行时预算接口 | 属于兼容的另一种压缩器；当前主要实验使用 HOBF |
| [run.py](run.py) | 通用推理入口，解析策略参数、加载 Router 权重、传参和输出记录 | `--budget_policy`、`--router_checkpoint` 与 `LatentMASMethod` 初始化；未给权重的 Router 是未训练网络 |
| [run_router_v2.py](run_router_v2.py)：`install_live_features()`、`evaluate()`、`summary()` | 在同一批验证题上运行固定 B8、固定 B32、hidden Router、统计特征 Router | 在线特征获取、权重恢复、逐题结果保存与续跑、最终正确率/KV量/时延统计 |

完整题目评估与历史状态验证分开理解：

| 维度 | 输入与执行过程 | 回答的问题 |
|---|---|---|
| **完整题目** | 每次根据当前实际状态选 B，压缩后的 KV 会改变后续 Agent 的输入和状态 | 最终答对多少题、传了多少 KV、整题花了多长时间 |
| **历史状态** | 读取已保存的固定状态，对照该状态的预算标签；预测不接回后续解题 | 网络在这些已知状态上能否判断合适预算 |

当前开发验证题来自 GSM8K `train` 内部划分，且用于模型选择，不是官方 `test` 成绩。报告的 KV 是传递张量大小的统计，不是实测网络流量；时延要结合硬件、同步、热身和统计范围解释。Oracle 的事后最优收益也不能等同于训练后 Router 的实际收益。

## 3. 辅助代码：按需要继续读

### 标签复核、特征补采与对照

| 文件 | 用途 | 与主线的关系 |
|---|---|---|
| [stage2_features.py](stage2_features.py) | 计算最后一步 hidden、最近5步均值、7维 hidden 统计与11维注意力统计；提供目标状态采集包装 | 统计特征共18维；部分函数也用于在线推理 |
| [stage2_teacher_table.py](stage2_teacher_table.py) | 早期小规模标签复核、特征采集和对照 | 不只是排版表格；其 `read`、`digest` 等函数被后续脚本导入 |
| [capture_expanded_impl.py](capture_expanded_impl.py) | 按已有预算前缀到达目标状态，补采扩充数据的特征 | 沿用标签并核对 hidden；不等于重新完成64组合标签复核 |
| [diagnose_router_features.py](diagnose_router_features.py) | 线性分类器、kNN、类中心等历史状态诊断 | 用来检查特征的可判别性，不执行完整 Agent 解题 |
| [expanded_feature_comparison.py](expanded_feature_comparison.py) | 组织固定划分下的补采计划、采集及三种特征对照 | 保存预测与按独立来源题重采样的波动区间 |

最近5步 hidden 均值参与了历史状态特征对照；当前 `run_router_v2.py` 的训练选择只有 `last_hidden` 和 `statistics`，不能据此描述成“三种特征都已训练并完成在线 Router 对照”。

### Oracle 收益分析

| 文件 | 用途 |
|---|---|
| [summarize_oracle_gain.py](summarize_oracle_gain.py) | 从已有完整搜索记录统计固定预算与 Oracle 的收益 |
| [compare_fixed_oracle_500.py](compare_fixed_oracle_500.py) | 原500题固定 B32 缺失结果补跑及 Oracle 比较 |
| [complete_oracle_1000.py](complete_oracle_1000.py) | 合并新旧1000题的可复用结果，组织缺失 B32 补跑与报告 |
| [oracle_comparison_core.py](oracle_comparison_core.py) | 结果读取/核对、固定预算补跑和统计的公共实现；与500题脚本存在功能重叠，当前保留 |

### 启动脚本与测试

启动脚本主要保存路径、参数和调用命令，不定义新的网络算法。部分路径固定为本次 AutoDL 环境，下载仓库后不能假定它们无需配置即可运行。

| 文件 | 对应任务 |
|---|---|
| [run.sh](run.sh) | 基础项目推理启动方式 |
| [run_expand_500.sh](run_expand_500.sh) | 新题规划、初筛、完整搜索 |
| [run_features_v2.sh](run_features_v2.sh) | 扩充数据的特征补采和对照 |
| [run_router_v2.sh](run_router_v2.sh) | 扩充后的训练与完整题目评估 |
| [run_stage2.sh](run_stage2.sh) | 早期小规模标签复核和特征诊断 |
| [run_compare_500.sh](run_compare_500.sh)、[run_oracle_1000.sh](run_oracle_1000.sh) | 对应题目范围的固定预算与 Oracle 比较 |

| 测试文件 | 主要核查内容 |
|---|---|
| [test_budget_router.py](tests/test_budget_router.py) | 网络输出形状、预算候选规则 |
| [test_dynamic_hobf.py](tests/test_dynamic_hobf.py) | 运行时 B 是否影响实际缓存长度 |
| [test_local_loading.py](tests/test_local_loading.py) | 本地/离线加载参数 |
| [test_oracle_labels.py](tests/test_oracle_labels.py) | 整题最优、并列最优、主轨迹及备用状态标签 |
| [test_router_data_selection.py](tests/test_router_data_selection.py) | 题目抽样、完整搜索选题与索引规则 |
| [test_router_training.py](tests/test_router_training.py) | 去重、来源划分和集合损失 |
| [test_numeric_scoring.py](tests/test_numeric_scoring.py) | debug新增：数值答案提取与等价判分 |
| [test_budget_contracts.py](tests/test_budget_contracts.py) | debug新增：最小预算执行及类别/预算约定 |
| [test_set_selection.py](tests/test_set_selection.py) | debug新增：集合兼容选模、早停与权重保存/恢复 |

这些测试主要使用构造数据或受控接口，不能替代真实数据标签复核和大模型端到端实验。已有测试也应作为老师检查实现意图的材料。

## 4. 哪些是原有代码，哪些是我们修改或新增的

比较基准是最初提供的 **author-clean 源码包**及其本地保留副本，不等同于已经核验过的上游 Git 提交；当前对象是逐次补丁整合后的本地 debug 副本，也没有在本次文档整理中与服务器目录逐文件对齐。

基础研究题目为 *When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration*。原项目介绍、引用和早期运行说明保留在 [整理前 README](README_历史说明.md)。本仓库新增研究是动态预算 Router，不能把原作者 OBF 方法和原论文结果描述成我们的新增成果。

| 分类 | 文件/范围 | 变化说明 |
|---|---|---|
| **原有文件，增加动态接入** | [run.py](run.py)、[models.py](models.py)、[methods/latent_mas.py](methods/latent_mas.py)、[HOBF.py](compression_methods/HOBF.py)、[Headwise.py](compression_methods/Headwise.py) | 策略与权重入口、最后一步 hidden 返回、本地加载、运行时预算及记录 |
| **原有文件，debug评分修正** | [utils.py](utils.py) | 修正数值答案提取与归一化；这会影响部分正确性判定，详见下一节 |
| **新增功能文件，共19个Python文件** | 第2节的数据生成/整理、网络/训练/v2入口，以及第3节的特征诊断和Oracle分析文件 | 按职责查阅对应表格；其中 `budget_router.py` 跨网络定义与标签构建两部分 |
| **新增启动脚本，共6个** | 第3节除 `run.sh` 外的 Shell 脚本 | 保存各轮实验配置与入口 |
| **新增测试，共9个Python文件** | 上述测试表 | 包括原先6个测试文件和debug修正新增的3个 |
| **沿用的其余基础Python文件** | 如 [data.py](data.py)、[prompts.py](prompts.py)、[methods/baseline.py](methods/baseline.py)、[methods/text_mas.py](methods/text_mas.py) 及其余压缩器等 | 本次与本地基准比较，这些文件的 Python 语法树一致；部分增加了中文文件头，不表示原始字节完全相同 |

当前共52个Python文件：28个相对基准新增；24个来自基础项目，其中上述6个有语法树变化。另有7个Shell脚本，其中6个新增。文件计数仅统计源码，不包括数据、日志和说明。

此前已为59个Python/Shell文件添加用途说明，并为189个新增函数/方法添加中文功能、输入、输出注释；原有函数中的新增代码块也有说明。注释版本已检查原有代码行及语法树保持不变。**这说明注释没有改变计算逻辑，不说明计算逻辑已经没有问题。**

## 5. debug 已经改了什么，哪些问题仍需要老师看

2026-11-19 的修正发生在此次 README 整理之前。这里列明它们，是为了区分当前代码与之前跑出实验结果的版本。

| 审查编号 | 文件 | debug版已完成的改动 | 对结果的边界 |
|---|---|---|---|
| R-005 | [utils.py](utils.py) | 正确处理合法千分位和普通十进制等价形式，用 Decimal 避免浮点近似误判 | 已保存的 `correct` 和标签不会自动更新；应按新规则重新核对原始回答和受影响标签 |
| R-006 | [screen_router_candidates.py](screen_router_candidates.py) | 初筛执行值和成功标签使用 `min(budget_choices)`，类别索引仍按候选表顺序解释 | 修正非升序候选情形；不能因此断言历史默认升序实验的执行预算出错 |
| R-023 | [run_router_v2.py](run_router_v2.py) | v2入口明确接受固定 `(8,16,32,64)`，检查标签关系并核对权重词表 | 避免类别编号与实际 B 解释错位；不代表所有配置及数据约束都已审计完 |
| R-020，设计调整 | [train_budget_router.py](train_budget_router.py) | 按“最优集合允许最低预算/不允许最低预算”两组的集合兼容准确率等权选模；同分依次比较总体兼容率、较低损失；早停使用同一规则 | 不是更换网络结构或训练损失；必须重新训练/选模才会影响选出的权重，旧权重不会自动更新 |

上述修正曾完成7组共39项相关测试，以及两种输入的合成数据 CPU 训练、保存和恢复检查。**这些是局部逻辑与接口证据，不是实际 GSM8K 成绩。** 本次 README 整理没有重新执行测试或训练；此前修正也没有完成真实 Qwen/193题GPU重跑。

以下内容仍保留当前实现，建议老师审查时关注：

- **标签与状态的对应**：预算前缀是否唯一确定当前状态，重复到达时如何检查特征；主轨迹代表标签与完整最优集合如何衔接。
- **训练目标与泛化**：类别/阶段权重来自哪些样本，少数类独立来源是否充分，新的选模准则是否适合研究目标。不能用已经加权来证明不平衡问题已解决。
- **训练与在线设置的一致性**：v2 权重记录了特征计划摘要，但评估中的核对并不覆盖全部记录；实际推理有固定参数，报告元数据与真实执行设置需要一起检查（既有 R-021）。
- **续跑及结果复用**：已有逐题文件怎样验收、不同数据/代码/参数是否可能混用；汇总的完整性判定与字段验证仍有审查事项（如 R-001、R-022）。这些未因本次首页整理而修复。
- **跨文件依赖与使用边界**：v2训练替换模块函数和 `sys.argv`；训练/采集入口从 `stage2_teacher_table.py` 导入公共工具；统计在线特征依赖 `stage2_features.py`。文件名像“做表格”不意味着可以从运行路径直接移除。

多个实验计划会保存并严格比较**源码文件 SHA256**。既往加注释或修正代码也会改变摘要；读取旧产物时应尊重版本核对，不能删除旧清单或跳过检查来把新旧结果混在一起。当前数据规模、模型快照路径及输出路径也保留实验时的配置，没有为展示统一改写。

## 6. 上传范围、运行前提与历史资料

这里说明准备上传的内容边界；**本次只写 README 并保留旧 README，没有执行 Git 提交/上传、删除文件、修改 `.gitignore` 或重新配置环境。**

| 内容 | 本次展示建议 | 说明 |
|---|---|---|
| Python源码、Shell脚本、测试、README | 随仓库保留，文件位置和内容按当前版本展示 | 便于老师追踪完整调用链，也保留当前实现中的问题 |
| [requirements.txt](requirements.txt)、[requirements-freeze.txt](requirements-freeze.txt) | 保留并注明角色 | 前者列直接依赖但未固定版本；后者是环境快照，不是已经验证适用于所有机器的一键安装配置 |
| 原项目图片及说明材料 | 与相应文档一起保留 | 保留原作者背景与来源，避免混淆原工作和新增工作 |
| 数据划分/配置说明、少量必要结果摘要 | 有对应版本信息时，可作为审查材料提供 | 单独注明题目范围、评分版本、代码版本；不把修复前成绩标为debug版复现结果 |
| `logs/`、`results/`、`failed_runs/`、缓存、权重、完整训练数据 | 原始产物在本地或实验存储中保留；不要求全部进入代码仓库 | 当前本地目录仍有历史日志和结果，本文没有替用户移除或排除它们 |

**当前上传前还有一个明确事项：** [`.gitignore`](.gitignore) 含有 `*`，会忽略全部未跟踪文件。此次按“只改文档”的范围保留原样；正式上传前需单独处理跟踪范围，否则不能假定源码会自动被加入提交。

若老师只做代码审查，不需要先下载模型或运行实验。若需要运行，应先核对入口要求的模型、生成参数、数据目录和环境；现有Shell脚本包含 `/root/autodl-tmp` 及特定模型快照。合并数据、补采特征、固定划分和训练权重是相应阶段的外部输入，不能仅凭源码目录存在就认为产物齐全。

历史说明入口：

- [整理前README：原项目介绍、引用与早期Router说明](README_历史说明.md)
- [新增500题的阶段说明](README_expansion.md)
- [早期标签复核/特征诊断说明](STAGE2_README.md)
- [特征诊断说明](FEATURE_DIAGNOSTICS.md)
- [原500题固定预算与Oracle比较说明](COMPARE_500.md)

这些阶段文档保留原文。审查当前实现时，以本 README 的版本边界和实际源码为准；旧文档中的补丁数量、运行路径、选模规则或阶段计划可能不再覆盖当前 debug 版。
