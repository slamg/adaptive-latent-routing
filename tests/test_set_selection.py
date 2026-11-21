# 文件作用：回归验证集合兼容分组选模与早停规则。
# 主要职责：检查并列标签、代表标签变化、类别不平衡和空组情形，并核对训练权重保存与恢复。

"""Regression tests for set-compatible, minority-aware checkpoint selection."""
import contextlib
import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import train_budget_router as training
from budget_router import BudgetRouter


class LogitRouter(torch.nn.Module):
    # 功能：将输入原样当作分类分数，隔离评估逻辑与真实网络学习过程。
    # 输入：self 为测试模型；features 为 (批量,类别数) 的预设 logits；handoffs 仅为兼容 Router 接口而接收。
    # 输出：原 features 张量，不使用阶段编号、不修改输入。
    def forward(self, features, handoffs):
        return features


# 功能：把指定的预测类别转成 logits，调用真实评估函数计算指标。
# 输入：masks 为逐样本最优类别掩码；targets 为代表类别索引；predictions 为预测类别索引；choices 为类别对应预算。
# 输出：training.evaluate 返回的指标字典，含集合兼容准确率、分组指标及诊断指标。
# 说明：内部在 CPU 上以 batch_size=2 评估构造样本，不训练网络。
def evaluate_predictions(masks, targets, predictions, choices=(8, 16, 32, 64)):
    records = []
    for mask, target, prediction in zip(masks, targets, predictions):
        logits = torch.full((len(choices),), -5.0)
        logits[prediction] = 5.0
        records.append({
            "router_feature": logits, "handoff_index": 0,
            "target_class": target,
            "optimal_budget_mask": torch.tensor(mask, dtype=torch.bool),
            "source_kind": "synthetic",
        })
    # A small batch size checks that group counts accumulate across batches.
    return training.evaluate(LogitRouter(), DataLoader(records, batch_size=2), torch.device("cpu"), choices)


# 功能：构造代表标签指标与集合标签评价会给出不同排序的反例。
# 输入：无参数；内部固定5条掩码、代表标签及两组预测。
# 输出：(masks,targets,a,b)：原始掩码、代表标签，以及模型A/B预测对应的评估指标字典。
def counterexample():
    masks = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    targets = [0, 1, 1, 2, 3]
    a = evaluate_predictions(masks, targets, [1, 1, 1, 2, 3])
    b = evaluate_predictions(masks, targets, [0, 1, 0, 2, 3])
    return masks, targets, a, b


class SetSelectionTest(unittest.TestCase):
    # 功能：检查新选模规则优先选择全部落在最优集合内的预测。
    # 输入：self 为 unittest 测试实例；使用 counterexample()；A的合法并列预测使旧 macro_recall 较低，但集合兼容表现好于B。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_legal_tie_no_longer_loses_to_wrong_budget(self):
        _, _, a, b = counterexample()
        self.assertAlmostEqual(a["compatible_accuracy"], 1.0)
        self.assertAlmostEqual(b["compatible_accuracy"], 0.8)
        self.assertAlmostEqual(a["macro_recall"], 0.75)
        self.assertAlmostEqual(b["macro_recall"], 0.875)
        self.assertGreater(training.selection_score(a), training.selection_score(b))
        self.assertEqual(a["balanced_compatible_accuracy"], 1.0)
        self.assertEqual(b["balanced_compatible_accuracy"], 0.875)

    # 功能：检查只改变并列集合内的代表标签不会改变新选模得分。
    # 输入：self 为 unittest 测试实例；保留 masks 和预测不变，将反例第一条代表类别从0换成1。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_representative_change_does_not_change_selection(self):
        masks, targets, a, _ = counterexample()
        targets[0] = 1  # Same optimal set, different legal representative.
        changed = evaluate_predictions(masks, targets, [1, 1, 1, 2, 3])
        self.assertNotEqual(a["macro_recall"], changed["macro_recall"])
        self.assertEqual(training.selection_score(a), training.selection_score(changed))
        self.assertEqual(a["per_budget_group"], changed["per_budget_group"])

    # 功能：检查分组等权评价不会被大量 B8 状态掩盖少数非B8状态的失败。
    # 输入：self 为 unittest 测试实例；内部构造9条仅允许B8及1条仅允许B16的状态，比较总选B8与能识别少数组的预测。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_minority_group_not_swamped_by_b8_majority(self):
        masks = [[1, 0, 0, 0]] * 9 + [[0, 1, 0, 0]]
        targets = [0] * 9 + [1]
        all_b8 = evaluate_predictions(masks, targets, [0] * 10)
        balanced = evaluate_predictions(masks, targets, [1, 1] + [0] * 7 + [1])
        self.assertGreater(all_b8["compatible_accuracy"], balanced["compatible_accuracy"])
        self.assertEqual(all_b8["balanced_compatible_accuracy"], 0.5)
        self.assertGreater(training.selection_score(balanced), training.selection_score(all_b8))

    # 功能：检查组别依据真实最优掩码是否允许最小预算，而非模型选出的预算。
    # 输入：self 为 unittest 测试实例；内部构造 {8,16} 与 {32,64} 两个最优集合，分别预测16、64。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_groups_use_mask_not_predicted_budget(self):
        # Selecting B16 is fully correct when the state allows both B8 and B16.
        result = evaluate_predictions([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 2], [1, 3])
        self.assertEqual(result["balanced_compatible_accuracy"], 1.0)
        self.assertEqual(result["per_budget_group"]["minimum_budget_allowed"]["support"], 1)
        self.assertEqual(result["per_budget_group"]["non_minimum_required"]["support"], 1)

    # 功能：检查某组没有样本时，不把该组错误计为零准确率纳入平均。
    # 输入：self 为 unittest 测试实例；内部分别构造仅允许最小预算组、仅需非最小预算组的数据，检查空组 support=0、准确率=None。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_absent_group_is_excluded_and_reported(self):
        for masks, targets, predictions, missing in (
            ([[1, 0, 0, 0], [1, 0, 0, 0]], [0, 0], [0, 1], "non_minimum_required"),
            ([[0, 1, 0, 0], [0, 0, 1, 0]], [1, 2], [1, 0], "minimum_budget_allowed"),
        ):
            with self.subTest(missing=missing):
                result = evaluate_predictions(masks, targets, predictions)
                self.assertEqual(result["balanced_compatible_accuracy"], 0.5)
                self.assertEqual(result["per_budget_group"][missing], {"support": 0, "compatible_accuracy": None})

    # 功能：检查指标分组根据预算实际数值定位最小预算，兼容非升序词表。
    # 输入：self 为 unittest 测试实例；内部使用 choices=(32,4,16)，检查最小预算为4且对应类别索引1。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_candidate_order_and_actual_minimum_are_respected(self):
        result = evaluate_predictions([[0, 1, 0], [1, 0, 0]], [1, 0], [1, 1], choices=(32, 4, 16))
        self.assertEqual(result["balanced_compatible_accuracy"], 0.5)
        self.assertEqual(result["per_budget_group"]["minimum_budget_allowed"]["compatible_accuracy"], 1.0)
        self.assertEqual(training.selection_metric_definition((32, 4, 16))["minimum_budget"], 4)

    # 功能：检查分组兼容准确率相同时依次按总体兼容准确率、较低损失选模。
    # 输入：self 为 unittest 测试实例；内部构造三组指标字典，比较 selection_score 返回元组的排序。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_tie_breakers_use_overall_compatibility_then_loss(self):
        base = {"balanced_compatible_accuracy": 0.8, "compatible_accuracy": 0.7, "loss": 0.5}
        overall_better = dict(base, compatible_accuracy=0.75, loss=0.8)
        lower_loss = dict(base, loss=0.4)
        self.assertGreater(training.selection_score(overall_better), training.selection_score(lower_loss))
        self.assertGreater(training.selection_score(lower_loss), training.selection_score(base))

    # 功能：检查主训练入口按新规则保存最佳权重、早停，并能重新加载所保存网络。
    # 输入：self 为 unittest 测试实例；内部构造训练/验证状态，替换数据入口及验证指标序列，但运行真实小型 Router、优化器与权重读写。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：CPU 临时实验预期最佳为第1轮、训练第3轮后停止；核对 checkpoint、metrics、setup、history，并重载检查预算映射。
    def test_main_selects_checkpoint_and_early_stops_with_new_rule(self):
        # Real optimizer/network/checkpoint I/O; controlled evaluation exposes the
        # known old-rule counterexample without relying on stochastic learning.
        records = []
        for source in ("train", "validation"):
            for handoff in range(3):
                for target in range(4):
                    records.append({
                        "source_id": source, "source_kind": "synthetic",
                        "router_feature": torch.tensor([float(target), float(handoff), 1.0]),
                        "handoff_index": handoff, "target_class": target,
                        "optimal_budget_mask": tuple(i == target for i in range(4)),
                    })
        _, _, a, b = counterexample()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["train_budget_router.py", "--screen_folder", "unused", "--exhaustive_folder", "unused",
                    "--output_folder", str(root), "--device", "cpu", "--epochs", "6", "--patience", "2",
                    "--router_hidden_dim", "8", "--handoff_embedding_dim", "2", "--batch_size", "12"]
            with patch.object(sys, "argv", argv), \
                 patch.object(training, "build_router_examples", return_value=(records, {"synthetic": True})), \
                 patch.object(training, "split_source_ids", return_value=(("train",), ("validation",))), \
                 patch.object(training, "evaluate", side_effect=[a, b, b, a, a]) as mocked, \
                 contextlib.redirect_stdout(io.StringIO()):
                training.main()
            self.assertEqual(mocked.call_count, 5)  # 3 epochs, final train and validation.
            ck = torch.load(root / "best_router.pt", map_location="cpu")
            metrics = json.loads((root / "metrics.json").read_text())
            setup = json.loads((root / "setup.json").read_text())
            with (root / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(ck["epoch"], 1)  # Old rule would save epoch 2.
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertEqual(len(history), 3)
            for report in (ck, metrics, setup):
                self.assertEqual(report["selection_metric"]["name"], training.selection_metric_definition((8, 16, 32, 64))["name"])
            self.assertEqual(float(history[0]["validation_balanced_compatible_accuracy"]), 1.0)
            self.assertEqual(int(history[0]["validation_non_minimum_required_support"]), 4)
            restored = BudgetRouter(**ck["router_config"]).eval()
            restored.load_state_dict(ck["state_dict"])
            with torch.no_grad():
                logits, indices, budgets = restored.select_budget(torch.tensor([[0., 1., 1.]]), torch.tensor([0]))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(int(budgets[0]), (8, 16, 32, 64)[int(indices[0])])


if __name__ == "__main__":
    unittest.main()
