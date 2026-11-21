# 文件作用：验证 Router 训练所用数据整理与集合标签损失。
# 主要职责：检查主/备用状态去重、来源划分无交叉，以及并列最优动作的损失处理。

import tempfile
import unittest
from pathlib import Path

import torch

from router_training_data import build_router_examples, split_source_ids
from train_budget_router import multi_positive_losses


# 功能：构造便于区分来源的三维测试特征。
# 输入：value 为起始数值。
# 输出：float32 张量 [value,value+1,value+2]，形状为 (3,)。
def feature(value):
    return torch.tensor([value, value + 1.0, value + 2.0], dtype=torch.float32)


class RouterTrainingTest(unittest.TestCase):
    # 功能：构造含初筛数据与完整搜索数据的临时目录，用于测试合并去重。
    # 输入：self 为测试实例；无外部数据参数，内部构造4道初筛题及其中第0题的主轨迹/备用状态。
    # 输出：(temporary,root)：临时目录管理对象与根 Path；同时向 screen/samples 和 exhaustive/samples 写入 .pt 文件。
    # 说明：调用方需登记 temporary.cleanup；包含一条并列最优记录，供验证掩码和去重规则。
    def make_folders(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        screen_samples = root / "screen" / "samples"
        exhaustive_samples = root / "exhaustive" / "samples"
        screen_samples.mkdir(parents=True)
        exhaustive_samples.mkdir(parents=True)

        for index in range(4):
            examples = [
                {
                    "handoff_index": handoff,
                    "router_feature": feature(index * 10 + handoff),
                    "target_budget": 8,
                    "target_class": 0,
                    "label_status": "minimum_budget_certified",
                }
                for handoff in range(3)
            ]
            torch.save(
                {
                    "dataset_index": index,
                    "task": "gsm8k",
                    "split": "train",
                    "budget_choices": (8, 16, 32, 64),
                    "minimum_budget_correct": True,
                    "primary_examples": examples,
                },
                screen_samples / f"sample_{index:06d}.pt",
            )

        reserve = []
        prefixes = [(), (8,), (16,), (8, 8)]
        for item_index, prefix in enumerate(prefixes):
            target_class = item_index % 4
            mask = [False] * 4
            mask[target_class] = True
            if prefix == (8,):
                mask[0] = True
            reserve.append(
                {
                    "handoff_index": len(prefix),
                    "prefix": prefix,
                    "router_feature": feature(100 + item_index),
                    "target_budget": (8, 16, 32, 64)[target_class],
                    "target_class": target_class,
                    "optimal_budget_mask": tuple(mask),
                    "label_status": "tied_optimum" if sum(mask) > 1 else "unique_optimum",
                }
            )
        primary = [
            {
                "handoff_index": handoff,
                "prefix": tuple(8 for _ in range(handoff)),
                "router_feature": feature(100 + handoff),
                "target_budget": 8,
                "target_class": 0,
                "label_status": "canonical_global_optimum",
            }
            for handoff in range(3)
        ]
        torch.save(
            {
                "sample_id": 0,
                "task": "gsm8k",
                "split": "train",
                "budget_choices": (8, 16, 32, 64),
                "primary_examples": primary,
                "reserve_state_examples": reserve,
            },
            exhaustive_samples / "sample_000000.pt",
        )
        return temporary, root

    # 功能：检查完整搜索的备用状态替代同题主轨迹，避免同一前缀重复进入训练。
    # 输入：self 为 unittest 测试实例；使用 make_folders() 生成临时数据，按 primary_reserve 视图整理。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：预期第0题4条备用状态加另3题各3条主轨迹，共13条；(source_id,prefix) 唯一。
    def test_primary_reserve_replaces_duplicate_primary_states(self):
        temporary, root = self.make_folders()
        self.addCleanup(temporary.cleanup)
        records, report = build_router_examples(
            root / "screen",
            root / "exhaustive",
            view="primary_reserve",
        )
        # Four reserve states for source 0 plus three primary states for each
        # of the other three screened sources.
        self.assertEqual(len(records), 13)
        self.assertEqual(report["tied_examples"], 1)
        keys = [(record["source_id"], record["prefix"]) for record in records]
        self.assertEqual(len(keys), len(set(keys)))

    # 功能：检查训练/验证按独立题目划分，来源集合不交叉且不遗漏。
    # 输入：self 为 unittest 测试实例；内部构造20个 source_id，其中15个 easy、5个 hard，以20%验证比例和 seed=9 划分。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_source_split_has_no_leakage(self):
        records = []
        for index in range(20):
            records.append(
                {
                    "source_id": f"gsm8k:train:{index}",
                    "source_kind": "easy" if index < 15 else "hard",
                }
            )
        train, validation = split_source_ids(
            records, validation_fraction=0.2, seed=9
        )
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), {x["source_id"] for x in records})

    # 功能：检查集合标签损失接受任意并列最优类别，而惩罚集合外预测。
    # 输入：self 为 unittest 测试实例；内部使用最优掩码 [True,True,False,False]，分别将高分给两个合法类别及一个错误类别。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_multi_positive_loss_accepts_probability_on_either_optimum(self):
        masks = torch.tensor([[True, True, False, False]])
        first = multi_positive_losses(
            torch.tensor([[10.0, -10.0, -10.0, -10.0]]), masks
        )
        second = multi_positive_losses(
            torch.tensor([[-10.0, 10.0, -10.0, -10.0]]), masks
        )
        wrong = multi_positive_losses(
            torch.tensor([[-10.0, -10.0, 10.0, -10.0]]), masks
        )
        self.assertLess(first.item(), 1e-5)
        self.assertLess(second.item(), 1e-5)
        self.assertGreater(wrong.item(), 10.0)


if __name__ == "__main__":
    unittest.main()
