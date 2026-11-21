# 文件作用：回归验证最小预算初筛和扩充训练/评估的预算类别约定。
# 主要职责：检查非升序预算的实际执行值、标签对应关系及错误checkpoint词表的拒绝行为。

"""Regression checks for R-006 and R-023 without loading a backbone model."""
import ast
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from budget_router import parse_budget_choices
import run_router_v2 as v2

ROOT = Path(__file__).resolve().parents[1]


# 功能：构造用于校验预算词表和标签一致性的最小合并数据。
# 输入：无参数。
# 输出：字典，含 budget_choices=(8,16,32,64) 及一条记录；代表标签为 B16，最优动作集合为 {16,32}。
def data():
    return dict(budget_choices=(8, 16, 32, 64), records=[dict(
        target_class=1, target_budget=16,
        optimal_budget_mask=(False, True, True, False))])


class BudgetContractTest(unittest.TestCase):
    # 功能：检查默认词表及包含代表标签的并列最优掩码能通过数据校验。
    # 输入：self 为 unittest 测试实例；内部调用 data() 生成合法数据。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_default_and_legal_tie(self):
        self.assertEqual(v2.validate_budget_contract(data()), (8, 16, 32, 64))

    # 功能：检查 v2 入口拒绝不符合固定约定的类别顺序。
    # 输入：self 为 unittest 测试实例；内部将词表改为 (64,32,16,8)，同步调整代表类别索引。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：分别调用 train/evaluate；应在访问数据路径或加载模型前抛出 ValueError。
    def test_permuted_consistent_vocabulary_rejected_before_training(self):
        d = data()
        d['budget_choices'] = (64, 32, 16, 8)
        d['records'][0]['target_class'] = 2
        # No model or data path is needed: reject before either can be loaded.
        with self.assertRaisesRegex(ValueError, 'budget choices'):
            v2.train(None, d, None)
        with self.assertRaisesRegex(ValueError, 'budget choices'):
            v2.evaluate(None, d, None)

    # 功能：检查类别索引、目标预算和最优动作掩码的非法组合被拒绝。
    # 输入：self 为 unittest 测试实例；内部构造越界/非整数类别、预算与类别不符、掩码长度或元素非法、代表类别不在掩码中的记录。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_inconsistent_records_are_rejected(self):
        for change in [dict(target_class=-1), dict(target_class=4),
                       dict(target_class=1.0), dict(target_budget=32),
                       dict(optimal_budget_mask=(True, False, False, False)),
                       dict(optimal_budget_mask=(False, True)),
                       dict(optimal_budget_mask=(False, 2, False, False))]:
            with self.subTest(change=change):
                d = data(); d['records'][0].update(change)
                with self.assertRaises(ValueError):
                    v2.validate_budget_contract(d)

    # 功能：检查评估入口会在加载大模型前拒绝错误词表的 Router 权重。
    # 输入：self 为 unittest 测试实例；从 evaluate 源码提取到权重校验结束的原始语句，注入模拟 checkpoint 和临时输出目录。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：仅执行该入口的校验片段，不运行 GPU 模型推理；不等同于完整端到端测试。
    def test_checkpoint_vocabulary_is_checked_before_loading_model(self):
        tree = ast.parse((ROOT / 'run_router_v2.py').read_text(encoding='utf-8'))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate')
        # Keep original evaluation code through checkpoint validation, omit imports
        # of GPU-only dependencies and stop before loading dataset or model.
        body = []
        for x in fn.body:
            if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'source_rows' for t in x.targets):
                break
            if not isinstance(x, (ast.Import, ast.ImportFrom)):
                body.append(x)
        part = copy.deepcopy(fn); part.body = body
        checkpoint = dict(router_config=dict(budget_choices=(64, 32, 16, 8)))
        ns = dict(validate_budget_contract=v2.validate_budget_contract,
                  read=lambda path: checkpoint)
        exec(compile(ast.Module(body=[part], type_ignores=[]), str(ROOT / 'run_router_v2.py'), 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            with self.assertRaisesRegex(ValueError, 'Checkpoint budget choices'):
                ns['evaluate'](args, data(), {})


class MinimumScreeningTest(unittest.TestCase):
    # 功能：从筛选脚本提取函数定义，在受控命名空间内用于轻量测试。
    # 输入：无参数，静态方法；从项目根目录读取 screen_router_candidates.py。
    # 输出：命名空间字典，可按名称取得 build_parser、make_primary_examples、main 等函数。
    # 说明：编译执行函数定义，不执行原模块的顶层模型导入和入口调用。
    @staticmethod
    def functions():
        path = ROOT / 'screen_router_candidates.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        ns = dict(torch=torch, HANDOFF_COUNT=3, parse_budget_choices=parse_budget_choices,
                  Path=Path, json=json, argparse=argparse, SCREEN_SCHEMA_VERSION=1)
        nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        return ns

    # 功能：检查初筛标签使用预算数值的最小值，同时按原词表位置给出类别索引。
    # 输入：self 为 unittest 测试实例；内部构造三次通信记录，并分别传入升序、倒序、乱序的候选预算。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_labels_use_minimum_value_and_original_class_order(self):
        result = dict(router_records=[dict(handoff_index=i, router_feature=torch.ones(2)) for i in range(3)])
        for choices in [(8, 16, 32, 64), (64, 32, 16, 8), (16, 8, 64, 32)]:
            rows = self.functions()['make_primary_examples'](result, choices)
            self.assertEqual([r['target_budget'] for r in rows], [8]*3)
            self.assertEqual([r['target_class'] for r in rows], [choices.index(8)]*3)

    # 功能：检查初筛入口真正传给推理函数的是三次最小预算，而非词表首元素。
    # 输入：self 为 unittest 测试实例；内部使用倒序词表、模拟模型/数据及临时目录；run_item 用自定义异常在推理入口处终止。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：断言实际调用的 budget_schedule 为 {0:8,1:8,2:8}，不加载大模型。
    def test_screen_main_passes_actual_minimum_schedule(self):
        class ReachedInference(Exception):
            pass
        ns = self.functions()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(candidate_budgets=[64,32,16,8], reference_budget=32,
                max_samples=1, task='gsm8k', split='train', cache_dir=None,
                output_folder=tmp, resume=False, sample_seed=42, seed=42,
                device='cpu', model_name='controlled', sink_size=4, pca_rank=8,
                latent_steps=20, max_new_tokens=20)
            method = Mock()
            method.run_item.side_effect = ReachedInference
            ns.update(build_parser=lambda: SimpleNamespace(parse_args=lambda: args),
                      load_task=lambda *a, **kw: [dict(question='q', gold='1')],
                      sample_dataset_indices=lambda *a: (0,), set_seed=lambda *a: None,
                      auto_device=lambda *a: 'cpu', ModelWrapper=Mock(), HOBF=Mock(),
                      LatentMASMethod=Mock(return_value=method), tqdm=Mock())
            with self.assertRaises(ReachedInference):
                ns['main']()
            self.assertEqual(method.run_item.call_args.kwargs['budget_schedule'], {0:8, 1:8, 2:8})


if __name__ == '__main__':
    unittest.main()
