# 文件作用：验证预算 Router 的基本输出和候选预算处理行为。
# 主要职责：使用构造张量检查网络接口与相关预算规则。

import unittest

import torch

from budget_router import BudgetRouter, parse_budget_choices


class BudgetRouterTest(unittest.TestCase):
    # 功能：检查 Router 批量输出形状和所选预算的取值范围。
    # 输入：self 为 unittest 测试实例；内部构造 (2,16) 状态特征、阶段编号 [0,2] 和四预算的小网络。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：分数应为 (2,4)，类别与预算均为 (2,)；所选预算应属于候选集合。
    def test_output_and_selected_budget_shapes(self):
        torch.manual_seed(7)
        router = BudgetRouter(
            hidden_dim=16,
            budget_choices=(8, 16, 32, 64),
            router_hidden_dim=12,
            dropout=0.0,
        ).eval()
        feature = torch.randn(2, 16)
        handoff_ids = torch.tensor([0, 2])

        logits, class_indices, budgets = router.select_budget(feature, handoff_ids)

        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(class_indices.shape), (2,))
        self.assertEqual(tuple(budgets.shape), (2,))
        self.assertTrue(set(budgets.tolist()).issubset({8, 16, 32, 64}))

    # 功能：检查候选预算解析保留合法值并拒绝重复或非正预算。
    # 输入：self 为 unittest 测试实例；内部传入 [8,16]、[8,8] 和 [0,8] 三组候选。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_budget_validation(self):
        self.assertEqual(parse_budget_choices([8, 16]), (8, 16))
        with self.assertRaises(ValueError):
            parse_budget_choices([8, 8])
        with self.assertRaises(ValueError):
            parse_budget_choices([0, 8])


if __name__ == "__main__":
    unittest.main()
