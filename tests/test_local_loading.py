# 文件作用：验证模型和分词器的本地及离线加载参数。
# 主要职责：检查本地路径识别、缓存路径传递和离线选项的构造。

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from models import build_pretrained_loading_kwargs


class LocalLoadingTest(unittest.TestCase):
    # 功能：检查本地模型目录强制启用严格离线加载并传递缓存路径。
    # 输入：self 为 unittest 测试实例；内部创建临时目录，并设 local_files_only=False、cache_dir 为测试路径。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：只构造加载参数，不实际下载或加载模型。
    def test_local_directory_enables_strict_offline_loading(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                cache_dir="/tmp/hf-cache",
                local_files_only=False,
            )
            source, tokenizer_kwargs, model_kwargs = (
                build_pretrained_loading_kwargs(temporary_directory, args)
            )

            self.assertEqual(source, str(Path(temporary_directory)))
            self.assertTrue(tokenizer_kwargs["local_files_only"])
            self.assertFalse(tokenizer_kwargs["fix_mistral_regex"])
            self.assertTrue(model_kwargs["local_files_only"])
            self.assertEqual(model_kwargs["cache_dir"], "/tmp/hf-cache")

    # 功能：检查仓库标识在未指定离线选项时保留默认加载行为。
    # 输入：self 为 unittest 测试实例；内部使用 Qwen/Qwen3-4B、空缓存路径和关闭的离线标记。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：只验证返回的参数，不发起网络请求。
    def test_repo_id_keeps_online_loading_by_default(self):
        args = SimpleNamespace(cache_dir=None, local_files_only=False)
        source, tokenizer_kwargs, model_kwargs = build_pretrained_loading_kwargs(
            "Qwen/Qwen3-4B",
            args,
        )

        self.assertEqual(source, "Qwen/Qwen3-4B")
        self.assertNotIn("local_files_only", tokenizer_kwargs)
        self.assertNotIn("fix_mistral_regex", tokenizer_kwargs)
        self.assertNotIn("local_files_only", model_kwargs)

    # 功能：检查仓库标识加显式离线选项时，模型与分词器均收到离线参数。
    # 输入：self 为 unittest 测试实例；内部使用模型仓库名、已有缓存路径形式的字符串以及 local_files_only=True。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    def test_explicit_offline_flag_also_works_with_cached_repo_id(self):
        args = SimpleNamespace(
            cache_dir="/root/autodl-tmp/hf_cache",
            local_files_only=True,
        )
        source, tokenizer_kwargs, model_kwargs = build_pretrained_loading_kwargs(
            "Qwen/Qwen3-4B",
            args,
        )

        self.assertEqual(source, "Qwen/Qwen3-4B")
        self.assertTrue(tokenizer_kwargs["local_files_only"])
        self.assertFalse(tokenizer_kwargs["fix_mistral_regex"])
        self.assertTrue(model_kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
