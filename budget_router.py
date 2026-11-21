# 文件作用：定义动态预算 Router 网络及 Oracle 预算标签构建规则。
# 主要职责：将状态特征和通信阶段映射为预算分数，并从搜索结果构建主轨迹与备用状态标签。

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import nn


class BudgetRouter(nn.Module):
    """A small classifier that maps a latent feature to a relay KV budget.

    The backbone LLM remains frozen.  The router predicts one logit for each
    entry in ``budget_choices`` and may optionally condition on the handoff
    (planner, critic, or refiner).
    """

    # 功能：构建将状态特征与通信阶段共同映射为候选预算分数的 Router 分类网络。
    # 输入：hidden_dim 是状态特征维度 D；budget_choices 是按类别顺序排列的候选预算，num_handoffs 是通信阶段数。
    # 输入：router_hidden_dim 控制中间层宽度，handoff_embedding_dim 控制阶段嵌入维度，dropout 控制丢弃概率。
    # 输出：无返回值（None）；在 self 上保存配置并注册 LayerNorm、阶段 Embedding 和两层全连接网络。
    # 流程：归一化状态特征后拼接阶段嵌入，输出维度为候选预算数量；候选顺序决定类别索引含义。
    def __init__(
        self,
        hidden_dim: int,
        *,
        budget_choices: Sequence[int] = (8, 16, 32, 64),
        num_handoffs: int = 3,
        router_hidden_dim: int = 256,
        handoff_embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        choices = tuple(int(value) for value in budget_choices)
        if not choices:
            raise ValueError("budget_choices must contain at least one value")
        if any(value <= 0 for value in choices):
            raise ValueError(f"All budget choices must be positive, got {choices}")
        if len(set(choices)) != len(choices):
            raise ValueError(f"budget_choices must be unique, got {choices}")

        self.hidden_dim = int(hidden_dim)
        self.budget_choices: Tuple[int, ...] = choices
        self.num_handoffs = int(num_handoffs)

        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.handoff_embedding = nn.Embedding(
            self.num_handoffs,
            int(handoff_embedding_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(
                self.hidden_dim + int(handoff_embedding_dim),
                int(router_hidden_dim),
            ),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(router_hidden_dim), len(self.budget_choices)),
        )

    # 功能：执行一次 Router 前向计算，为每个状态给出各候选预算的未归一化分数。
    # 输入：router_feature 为 [N, D] 特征张量；handoff_ids 为 [N] 阶段编号，或可扩展到整批的标量编号。
    # 输出：形状为 [N, K] 的 logits；N 是批量大小，K 是候选预算数，尚未转换为概率或具体预算。
    # 流程：检查形状，将阶段编号转为 long，特征转为网络参数精度，再拼接归一化特征与阶段嵌入。
    # 注意：输入特征与网络参数应位于同一设备；阶段编号对应从 0 开始的 Embedding 行号。
    def forward(
        self,
        router_feature: torch.Tensor,
        handoff_ids: torch.Tensor,
    ) -> torch.Tensor:
        if router_feature.dim() != 2:
            raise ValueError(
                "router_feature must have shape (batch, hidden_dim), "
                f"got {tuple(router_feature.shape)}"
            )
        if router_feature.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got {router_feature.shape[-1]}"
            )

        handoff_ids = handoff_ids.to(
            device=router_feature.device,
            dtype=torch.long,
        )
        if handoff_ids.dim() == 0:
            handoff_ids = handoff_ids.expand(router_feature.shape[0])
        if handoff_ids.shape != (router_feature.shape[0],):
            raise ValueError(
                "handoff_ids must have shape (batch,), "
                f"got {tuple(handoff_ids.shape)}"
            )

        parameter_dtype = self.feature_norm.weight.dtype
        feature = router_feature.to(dtype=parameter_dtype)
        feature = self.feature_norm(feature)
        handoff_feature = self.handoff_embedding(handoff_ids)
        return self.network(torch.cat([feature, handoff_feature], dim=-1))

    # 功能：依据 Router 分数选择当前通信节点的预算。
    # 输入：router_feature 为 [N, D] 特征，handoff_ids 为阶段编号，要求与 forward 相同。
    # 输出：依次返回 logits [N, K]、所选类别索引 [N]、映射后的实际预算值 [N]。
    # 流程：先前向计算，再沿候选维取 argmax，最后用 budget_choices 将类别索引转换为 B。
    # 注意：此方法不自行关闭梯度或切换 eval 模式；训练和推理模式由调用方管理。
    def select_budget(
        self,
        router_feature: torch.Tensor,
        handoff_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, selected class indices, and selected budgets."""

        logits = self(router_feature, handoff_ids)
        class_indices = logits.argmax(dim=-1)
        choices = torch.as_tensor(
            self.budget_choices,
            device=class_indices.device,
            dtype=torch.long,
        )
        budgets = choices[class_indices]
        return logits, class_indices, budgets


# 功能：统一候选预算的数据类型并检查候选集合是否可用。
# 输入：values 是可迭代的预算值序列，各值会转换为整数。
# 输出：保持输入顺序的整数元组；空集合、非正预算或重复预算会触发 ValueError。
# 注意：这里不排序；后续类别索引始终对应这个元组中的位置。
def parse_budget_choices(values: Iterable[int]) -> Tuple[int, ...]:
    choices = tuple(int(value) for value in values)
    if not choices:
        raise ValueError("At least one budget choice is required")
    if any(value <= 0 for value in choices):
        raise ValueError(f"Budget choices must be positive, got {choices}")
    if len(set(choices)) != len(choices):
        raise ValueError(f"Budget choices must be unique, got {choices}")
    return choices


# 功能：在给定单个预算的候选结果中，选择能够答对的最小预算。
# 输入：budget_results 以预算为键且包含 correct 字段；budget_choices 指定候选顺序。
# 输出：返回（目标预算、对应类别索引、状态）；全部失败时为（None, None, all_incorrect）。
# 注意：全部候选成功标为 all_correct，否则成功结果标为 valid；此函数不枚举三次通信的联合轨迹。
def choose_oracle_budget(
    budget_results: Dict[int, Dict],
    budget_choices: Tuple[int, ...],
) -> tuple[int | None, int | None, str]:
    """Choose the lowest-cost budget that produces a correct final answer.

    All-incorrect counterfactual sets deliberately receive no hard label.  A
    later training pipeline may add a continuous loss or teacher score for
    those examples without silently teaching the router that the smallest
    budget was correct.
    """

    correct_budgets = [
        budget
        for budget in budget_choices
        if bool(budget_results[budget]["correct"])
    ]
    if not correct_budgets:
        return None, None, "all_incorrect"

    target_budget = min(correct_budgets)
    target_class = budget_choices.index(target_budget)
    status = (
        "all_correct"
        if len(correct_budgets) == len(budget_choices)
        else "valid"
    )
    return target_budget, target_class, status


# 功能：生成所有通信节点预算选择的笛卡尔积，供完整轨迹搜索使用。
# 输入：budget_choices 为候选预算序列，num_handoffs 为正整数通信次数。
# 输出：轨迹元组组成的元组；每条轨迹长度为 num_handoffs，总数为候选数的 num_handoffs 次方。
# 示例：4 个预算、3 次通信对应 64 条轨迹；枚举顺序沿用候选预算顺序。
def enumerate_budget_schedules(
    budget_choices: Sequence[int],
    num_handoffs: int = 3,
) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate every joint budget schedule in deterministic order."""

    choices = parse_budget_choices(budget_choices)
    if num_handoffs <= 0:
        raise ValueError("num_handoffs must be positive")
    return tuple(product(choices, repeat=int(num_handoffs)))


# 功能：从整题搜索结果中找出总预算最小的所有成功轨迹。
# 输入：schedule_results 以预算轨迹元组为键，结果中的 correct 表示最终答案是否正确。
# 输出：返回（最低总预算、排序后的全部并列最优轨迹、unique_optimum 或 tied_optimum 状态）。
# 注意：全部失败时返回（None, 空元组, all_incorrect）；优化成本是 sum(schedule)，不是实测 KV 字节数。
def select_global_optimal_schedules(
    schedule_results: Mapping[Tuple[int, ...], Mapping],
) -> tuple[int | None, Tuple[Tuple[int, ...], ...], str]:
    """Return all correct schedules with minimum total nominal KV budget."""

    correct_schedules = [
        tuple(int(value) for value in schedule)
        for schedule, result in schedule_results.items()
        if bool(result.get("correct", False))
    ]
    if not correct_schedules:
        return None, tuple(), "all_incorrect"

    minimum_cost = min(sum(schedule) for schedule in correct_schedules)
    optimal_schedules = tuple(
        sorted(
            schedule
            for schedule in correct_schedules
            if sum(schedule) == minimum_cost
        )
    )
    status = "unique_optimum" if len(optimal_schedules) == 1 else "tied_optimum"
    return minimum_cost, optimal_schedules, status


# 功能：从并列最优轨迹中选出一个稳定的代表，用于构造主轨迹样本。
# 输入：optimal_schedules 为最优预算轨迹序列，各轨迹会转换为整数元组。
# 输出：按字典序最小的轨迹元组；没有成功轨迹时返回 None。
# 注意：代表轨迹只是固定取样依据，不能据此认为原问题不存在其他并列最优方案。
def select_canonical_schedule(
    optimal_schedules: Sequence[Sequence[int]],
) -> Tuple[int, ...] | None:
    """Choose a reproducible hard-label trajectory from tied global optima."""

    normalized = [tuple(int(value) for value in schedule) for schedule in optimal_schedules]
    return min(normalized) if normalized else None


# 功能：把代表性全局最优轨迹上的每个决策前状态整理成一个监督样本。
# 输入：canonical_schedule 是代表轨迹或 None；state_records 用已执行预算前缀定位特征；budget_choices 决定类别索引。
# 输出：主轨迹样本列表，含 handoff_index、prefix、router_feature、target_budget、target_class 及标签状态。
# 流程：第 t 次决策使用轨迹前 t 项作为状态键，以轨迹第 t 项作为标签；无成功轨迹时返回空列表。
def build_primary_examples(
    canonical_schedule: Sequence[int] | None,
    state_records: Mapping[Tuple[int, ...], Mapping],
    budget_choices: Sequence[int],
) -> list[Dict]:
    """Build the three hard-label examples on the canonical optimal path."""

    if canonical_schedule is None:
        return []

    choices = parse_budget_choices(budget_choices)
    schedule = tuple(int(value) for value in canonical_schedule)
    examples: list[Dict] = []
    for handoff_index, target_budget in enumerate(schedule):
        prefix = schedule[:handoff_index]
        if prefix not in state_records:
            raise KeyError(f"Missing router state for prefix {prefix}")
        state = state_records[prefix]
        examples.append(
            {
                "handoff_index": handoff_index,
                "prefix": prefix,
                "agent_name": state.get("agent_name"),
                "agent_role": state.get("agent_role"),
                "router_feature": state["router_feature"],
                "target_budget": target_budget,
                "target_class": choices.index(target_budget),
                "label_status": "canonical_global_optimum",
            }
        )
    return examples


# 功能：用完整轨迹结果反向计算每个预算前缀状态的最优后续动作，保留并列最优预算。
# 输入：schedule_results 为完整组合结果，state_records 为前缀状态特征，budget_choices 与 num_handoffs 定义搜索树。
# 输出：各前缀状态的样本列表，包含特征、代表标签、optimal_budget_mask、最低后续成本和状态标记。
# 流程：从最后一次通信向前递推，动作成本为当前 B 加成功后继的最低成本；最低后续成本不含已花费的前缀预算。
# 注意：无成功后继的状态仍输出，但标签为 None、掩码全 False；默认 4 个预算和 3 次通信共有 21 个前缀状态。
def build_reserve_state_examples(
    schedule_results: Mapping[Tuple[int, ...], Mapping],
    state_records: Mapping[Tuple[int, ...], Mapping],
    budget_choices: Sequence[int],
    num_handoffs: int = 3,
) -> list[Dict]:
    """Build Oracle labels for every reachable budget-prefix state.

    The value of a prefix is computed backwards from correct complete
    schedules.  This produces 1 + K + ... + K**(H-1) state records for K
    choices and H handoffs.  States with no correct descendant are retained
    for analysis but deliberately receive no hard label.
    """

    choices = parse_budget_choices(budget_choices)
    expected_schedules = enumerate_budget_schedules(choices, num_handoffs)
    missing = [schedule for schedule in expected_schedules if schedule not in schedule_results]
    if missing:
        raise ValueError(
            "schedule_results must contain the complete Cartesian product; "
            f"missing {len(missing)} schedules"
        )

    # minimum_future_cost[prefix] excludes budgets already present in prefix.
    minimum_future_cost: Dict[Tuple[int, ...], int | None] = {}
    records_by_prefix: Dict[Tuple[int, ...], Dict] = {}

    for handoff_index in range(num_handoffs - 1, -1, -1):
        for prefix in product(choices, repeat=handoff_index):
            prefix = tuple(prefix)
            action_costs: Dict[int, int | None] = {}
            for budget in choices:
                child = prefix + (budget,)
                if len(child) == num_handoffs:
                    feasible = bool(schedule_results[child].get("correct", False))
                    future_cost = 0 if feasible else None
                else:
                    future_cost = minimum_future_cost.get(child)
                action_costs[budget] = (
                    budget + future_cost if future_cost is not None else None
                )

            feasible_costs = [cost for cost in action_costs.values() if cost is not None]
            if feasible_costs:
                best_cost = min(feasible_costs)
                optimal_budgets = tuple(
                    budget
                    for budget in choices
                    if action_costs[budget] == best_cost
                )
                target_budget = optimal_budgets[0]
                target_class = choices.index(target_budget)
                label_status = (
                    "unique_optimum" if len(optimal_budgets) == 1 else "tied_optimum"
                )
                minimum_future_cost[prefix] = best_cost
            else:
                best_cost = None
                optimal_budgets = tuple()
                target_budget = None
                target_class = None
                label_status = "no_successful_descendant"
                minimum_future_cost[prefix] = None

            if prefix not in state_records:
                raise KeyError(f"Missing router state for prefix {prefix}")
            state = state_records[prefix]
            records_by_prefix[prefix] = {
                "handoff_index": handoff_index,
                "prefix": prefix,
                "agent_name": state.get("agent_name"),
                "agent_role": state.get("agent_role"),
                "router_feature": state["router_feature"],
                "action_costs": action_costs,
                "minimum_future_cost": best_cost,
                "optimal_budgets": optimal_budgets,
                "optimal_budget_mask": tuple(
                    budget in optimal_budgets for budget in choices
                ),
                "target_budget": target_budget,
                "target_class": target_class,
                "label_status": label_status,
                "feature_observations": int(state.get("feature_observations", 1)),
                "feature_max_abs_diff": float(state.get("feature_max_abs_diff", 0.0)),
            }

    return [
        records_by_prefix[prefix]
        for handoff_index in range(num_handoffs)
        for prefix in product(choices, repeat=handoff_index)
    ]
