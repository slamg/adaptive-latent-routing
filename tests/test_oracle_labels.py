# 文件作用：验证完整预算搜索结果对应的 Oracle 标签逻辑。
# 主要职责：检查最优轨迹、并列最优、主轨迹标签及不同预算前缀的备用状态标签。

import unittest
from itertools import product

from budget_router import (
    build_primary_examples,
    build_reserve_state_examples,
    choose_oracle_budget,
    enumerate_budget_schedules,
    select_canonical_schedule,
    select_global_optimal_schedules,
)


class OracleLabelTest(unittest.TestCase):
    choices = (8, 16, 32, 64)

    # 功能：为所有三次预算组合构造正确性记录，供 Oracle 单元测试使用。
    # 输入：self.choices 为候选预算；correct_schedules 指定哪些完整组合答对，默认没有成功组合。
    # 输出：以预算组合元组为键的字典，每项为 {correct:布尔值}；默认四预算下共有64项。
    def make_results(self, correct_schedules=()):
        correct_schedules = {tuple(schedule) for schedule in correct_schedules}
        return {
            schedule: {"correct": schedule in correct_schedules}
            for schedule in enumerate_budget_schedules(self.choices)
        }

    # 功能：为所有决策前预算前缀构造状态占位记录。
    # 输入：self.choices 为四个候选预算；无其他参数。
    # 输出：以长度0、1、2的前缀为键的21项字典，值包含特征字符串及 Agent 名称/角色。
    # 说明：这些特征是标签逻辑测试的占位符，并非可以训练的真实 hidden 张量。
    def make_states(self):
        return {
            tuple(prefix): {
                "router_feature": f"feature-{tuple(prefix)}",
                "agent_name": f"agent-{handoff_index}",
                "agent_role": f"role-{handoff_index}",
            }
            for handoff_index in range(3)
            for prefix in product(self.choices, repeat=handoff_index)
        }

    # 功能：检查单步 Oracle 从成功预算中选择数值最小者。
    # 输入：self 为 unittest 测试实例；内部设 B8 错、B16/B32/B64 对，预期返回 (16,1,"valid")。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_smallest_correct_budget_is_selected(self):
        choices = (8, 16, 32, 64)
        results = {
            8: {"correct": False},
            16: {"correct": True},
            32: {"correct": True},
            64: {"correct": True},
        }
        self.assertEqual(
            choose_oracle_budget(results, choices),
            (16, 1, "valid"),
        )

    # 功能：检查单步预算全部失败时不伪造最小预算标签。
    # 输入：self 为 unittest 测试实例；内部将四个候选预算均标记为失败，预期预算与类别均为 None。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_all_incorrect_is_not_given_a_false_label(self):
        choices = (8, 16, 32, 64)
        results = {budget: {"correct": False} for budget in choices}
        self.assertEqual(
            choose_oracle_budget(results, choices),
            (None, None, "all_incorrect"),
        )

    # 功能：检查四预算、三次通信的笛卡尔积完整且无重复。
    # 输入：self 为 unittest 测试实例；使用 self.choices=(8,16,32,64)，预期64种组合，首尾为 (8,8,8)/(64,64,64)。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_joint_enumeration_has_64_unique_schedules(self):
        schedules = enumerate_budget_schedules(self.choices)
        self.assertEqual(len(schedules), 64)
        self.assertEqual(len(set(schedules)), 64)
        self.assertEqual(schedules[0], (8, 8, 8))
        self.assertEqual(schedules[-1], (64, 64, 64))

    # 功能：检查完整题目 Oracle 按三次预算之和选最优，而非只看第一个 B。
    # 输入：self 为 unittest 测试实例；内部仅设 (8,8,64) 和 (16,16,16) 成功，预期后者以总预算48胜出。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_global_oracle_uses_total_cost_not_local_smallest_budget(self):
        results = self.make_results(
            correct_schedules=[
                (8, 8, 64),   # cost 80
                (16, 16, 16), # cost 48: global optimum
            ]
        )
        minimum_cost, schedules, status = select_global_optimal_schedules(results)
        self.assertEqual(minimum_cost, 48)
        self.assertEqual(schedules, ((16, 16, 16),))
        self.assertEqual(status, "unique_optimum")

    # 功能：检查并列全局最优全部保留，并能确定性选取代表轨迹。
    # 输入：self 为 unittest 测试实例；内部设置两个总预算56的成功轨迹及一个高成本成功轨迹。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：代表轨迹按元组字典序选择，其他并列最优不应从最优集合丢失。
    def test_tied_global_optima_are_preserved_and_canonicalized(self):
        results = self.make_results(
            correct_schedules=[
                (16, 8, 32),
                (8, 16, 32),
                (64, 64, 64),
            ]
        )
        minimum_cost, schedules, status = select_global_optimal_schedules(results)
        self.assertEqual(minimum_cost, 56)
        self.assertEqual(schedules, ((8, 16, 32), (16, 8, 32)))
        self.assertEqual(status, "tied_optimum")
        self.assertEqual(select_canonical_schedule(schedules), (8, 16, 32))

    # 功能：检查完整搜索全部失败时没有最低成功成本或代表成功轨迹。
    # 输入：self 为 unittest 测试实例；内部生成64条全失败记录，预期成本 None、最优集合空、代表轨迹 None。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_all_incorrect_joint_search_has_no_canonical_path(self):
        results = self.make_results()
        minimum_cost, schedules, status = select_global_optimal_schedules(results)
        self.assertIsNone(minimum_cost)
        self.assertEqual(schedules, ())
        self.assertEqual(status, "all_incorrect")
        self.assertIsNone(select_canonical_schedule(schedules))

    # 功能：检查主轨迹样本的三个前缀与当前预算标签正确对应。
    # 输入：self 为 unittest 测试实例；内部指定轨迹 (16,8,32)，预期前缀依次为 ()、(16,)、(16,8)。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_primary_examples_contain_only_three_canonical_path_states(self):
        states = self.make_states()
        examples = build_primary_examples(
            (16, 8, 32),
            states,
            self.choices,
        )
        self.assertEqual(len(examples), 3)
        self.assertEqual(
            [(record["prefix"], record["target_budget"]) for record in examples],
            [((), 16), ((16,), 8), ((16, 8), 32)],
        )

    # 功能：检查备用状态视图覆盖全部21个决策前前缀。
    # 输入：self 为 unittest 测试实例；内部设64种组合全答对，预期每个前缀的最优当前预算均为8。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_reserve_view_contains_21_states(self):
        results = {
            schedule: {"correct": True}
            for schedule in enumerate_budget_schedules(self.choices)
        }
        records = build_reserve_state_examples(
            results,
            self.make_states(),
            self.choices,
        )
        self.assertEqual(len(records), 21)
        self.assertTrue(all(record["target_budget"] == 8 for record in records))

    # 功能：检查状态标签考虑后续最优延续，不能仅贪心选当前最小预算。
    # 输入：self 为 unittest 测试实例；内部仅设 (8,64,8) 和 (16,8,8) 成功，预期根状态选16，总未来成本32。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_reserve_root_action_accounts_for_best_future_continuation(self):
        results = self.make_results(
            correct_schedules=[
                (8, 64, 8),  # choosing 8 first forces total cost 80
                (16, 8, 8),  # choosing 16 first gives global cost 32
            ]
        )
        records = build_reserve_state_examples(
            results,
            self.make_states(),
            self.choices,
        )
        root = next(record for record in records if record["prefix"] == ())
        self.assertEqual(root["target_budget"], 16)
        self.assertEqual(root["minimum_future_cost"], 32)
        self.assertEqual(root["action_costs"][8], 80)
        self.assertEqual(root["action_costs"][16], 32)

    # 功能：检查已走入无成功后继的前缀时保留无标签状态。
    # 输入：self 为 unittest 测试实例；内部仅设 (8,8,8) 成功；前缀 (16,) 应为 no_successful_descendant，预算为 None。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_state_without_successful_descendant_has_no_false_label(self):
        results = self.make_results(correct_schedules=[(8, 8, 8)])
        records = build_reserve_state_examples(
            results,
            self.make_states(),
            self.choices,
        )
        by_prefix = {record["prefix"]: record for record in records}
        self.assertEqual(by_prefix[()]["target_budget"], 8)
        self.assertEqual(by_prefix[(8,)]["target_budget"], 8)
        self.assertIsNone(by_prefix[(16,)]["target_budget"])
        self.assertEqual(
            by_prefix[(16,)]["label_status"],
            "no_successful_descendant",
        )


if __name__ == "__main__":
    unittest.main()
