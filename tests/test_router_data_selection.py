# 文件作用：验证题目抽样、索引文件和完整搜索选题规则。
# 主要职责：使用构造名单检查随机抽样、索引读写及失败题/成功题的选择行为。

import tempfile
import unittest
from pathlib import Path

from router_data_selection import (
    choose_full_search_indices,
    read_index_file,
    sample_dataset_indices,
)


class RouterDataSelectionTest(unittest.TestCase):
    # 功能：检查相同随机种子的题目抽样可复现且无重复。
    # 输入：self 为 unittest 测试实例；内部两次从100题抽20题，均使用 seed=7，比较索引列表、数量与唯一性。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_random_indices_are_reproducible_and_unique(self):
        first = sample_dataset_indices(100, 20, seed=7)
        second = sample_dataset_indices(100, 20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)

    # 功能：检查失败题和成功抽查题分别入选，并按 hard+easy 拼接完整搜索名单。
    # 输入：self 为 unittest 测试实例；内部构造10题，前4题初筛失败；请求抽取3道失败题和2道成功题。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_full_search_selection_prioritizes_hard_and_audits_easy(self):
        indices = tuple(range(10))
        outcomes = {index: index >= 4 for index in indices}
        hard, easy, combined = choose_full_search_indices(
            indices,
            outcomes,
            hard_count=3,
            easy_count=2,
            seed=11,
        )
        self.assertEqual(len(hard), 3)
        self.assertEqual(len(easy), 2)
        self.assertTrue(all(not outcomes[index] for index in hard))
        self.assertTrue(all(outcomes[index] for index in easy))
        self.assertEqual(combined, hard + easy)

    # 功能：检查题目索引文件中的重复索引会被明确拒绝。
    # 输入：self 为 unittest 测试实例；内部在临时目录写入索引 3、5、3，预期 read_index_file 抛出含 Duplicate 的 ValueError。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_index_file_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.txt"
            path.write_text("3\n5\n3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_index_file(path)


if __name__ == "__main__":
    unittest.main()
