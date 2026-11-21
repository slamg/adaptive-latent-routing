# 文件作用：回归验证数值答案提取与等价比较规则。
# 主要职责：覆盖千分位、小数、符号及精度，并检查错误答案与非法数字格式不被误接受。

import unittest
from decimal import localcontext

from utils import extract_gold, extract_gsm8k_answer, normalize_answer


class NumericScoringTest(unittest.TestCase):
    # 功能：检查千分位、正负号、小数及负零可统一成等价的数字答案。
    # 输入：self 为 unittest 测试实例；内部逐组提供 boxed 答案及预期规范字符串，例如 1,000→1000、42.0→42。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_original_regressions_and_equivalent_decimals(self):
        pairs = [
            (r"\boxed{1,000}", "1000"),
            (r"\boxed{42.0}", "42"),
            (r"\boxed{+00042.000}", "42"),
            (r"\boxed{-1,234.500}", "-1234.5"),
            (r"\boxed{.50}", "0.5"),
            (r"\boxed{-.50}", "-0.5"),
            (r"\boxed{-0.000}", "0"),
            (r"\boxed{1,000,000.12500}", "1000000.125"),
        ]
        for raw, expected in pairs:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(extract_gsm8k_answer(raw)), expected)

    # 功能：检查预测答案和 GSM8K 标准答案采用一致的数值比较规则。
    # 输入：self 为 unittest 测试实例；内部提供预测文本与含 #### 的标准答案，分别提取、规范化后比较。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_gold_and_prediction_use_same_numeric_policy(self):
        pairs = [
            (r"\boxed{1000.00}", "reasoning\n#### 1,000"),
            ("The answer is -1,234.5.", "reasoning\n#### -1234.500"),
            (r"\boxed{42}", "reasoning\n#### 42.000"),
        ]
        for raw, solution in pairs:
            with self.subTest(raw=raw):
                pred = normalize_answer(extract_gsm8k_answer(raw))
                gold = normalize_answer(extract_gold(solution))
                self.assertIsNotNone(pred)
                self.assertIsNotNone(gold)
                self.assertEqual(pred, gold)

    # 功能：检查规范化不会把数值不同的答案误判为相同。
    # 输入：self 为 unittest 测试实例；内部使用相邻整数、不同符号、不同小数及高精度小数反例。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_wrong_answers_do_not_become_equal(self):
        for raw, gold in [(r"\boxed{1,001}", "1000"),
                          (r"\boxed{-42.0}", "42"),
                          (r"\boxed{42.01}", "42"),
                          (r"\boxed{0.30000000000000000001}", "0.3")]:
            with self.subTest(raw=raw):
                self.assertNotEqual(normalize_answer(extract_gsm8k_answer(raw)), gold)

    # 功能：检查非法千分位格式不会被截出一个局部数字作为答案。
    # 输入：self 为 unittest 测试实例；内部将 12,34、1,,000 等非法数字分别放入 boxed、普通预测文本和 #### 标准答案。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：三个提取入口均应返回 None，而非忽略错误分隔符。
    def test_malformed_grouping_does_not_produce_partial_answer(self):
        for token in ("12,34", "1,00", "1,,000", "1234,567", "1,234,56", "1,,"):
            with self.subTest(token=token):
                self.assertIsNone(extract_gsm8k_answer(r"\boxed{" + token + "}"))
                self.assertIsNone(extract_gsm8k_answer("Earlier 2; final answer " + token))
                self.assertIsNone(extract_gold("reasoning\n#### " + token))

    # 功能：检查末个 boxed 答案优先、普通文本回退及文本标签处理规则。
    # 输入：self 为 unittest 测试实例；内部构造多答案文本、尾随标点、字母/真假标签、空值以及无数字文本。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_answer_selection_and_text_labels_are_preserved(self):
        self.assertEqual(extract_gsm8k_answer(r"\boxed{3}, then \boxed{42}"), "42")
        self.assertEqual(extract_gsm8k_answer(r"Earlier 77. \boxed{42}. Later 99."), "42")
        self.assertEqual(extract_gsm8k_answer("Values 12 and 34. Final answer: 1,234."), "1,234")
        self.assertEqual(extract_gsm8k_answer("The result is 42, therefore done."), "42")
        self.assertEqual(extract_gsm8k_answer(r"\boxed{ A }"), "A")
        for label in (" A ", "True", " Yes ", "NaN", "Infinity", "1/2"):
            self.assertEqual(normalize_answer(label), label.strip().lower())
        self.assertIsNone(normalize_answer(None))
        self.assertIsNone(extract_gsm8k_answer("no numeric answer"))
        self.assertIsNone(extract_gold("no answer marker"))

    # 功能：检查尚未支持的科学计数法不会被误提取为不完整数字。
    # 输入：self 为 unittest 测试实例；内部使用 1e3、1E-3、-2e+4，并检查预测及标准答案提取均返回 None。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_unsupported_exponent_not_silently_truncated(self):
        for token in ("1e3", "1E-3", "-2e+4"):
            self.assertIsNone(extract_gsm8k_answer(r"\boxed{" + token + "}"))
            self.assertIsNone(extract_gold("#### " + token))

    # 功能：检查数值规范化不受 Decimal 当前精度设置影响而丢失有效数字。
    # 输入：self 为 unittest 测试实例；内部将局部精度设为5，并规范化超过该长度的整数与小数。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：只允许去掉无意义尾零；必须保留其他有效位。
    def test_decimal_canonicalization_does_not_round_to_context_precision(self):
        value = "123456789012345678901234567890.12345678901234567890"
        with localcontext() as context:
            context.prec = 5
            self.assertEqual(normalize_answer(value), value[:-1])
            self.assertEqual(normalize_answer(value + "1"), value + "1")


if __name__ == "__main__":
    unittest.main()
