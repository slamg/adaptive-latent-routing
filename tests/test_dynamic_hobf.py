# 文件作用：验证 HOBF 是否按运行时传入的预算执行压缩。
# 主要职责：使用构造KV缓存检查不同预算对应的保留长度及默认预算行为。

import importlib
import unittest
from unittest.mock import patch

import torch

from compression_methods import HOBF


# Import the module explicitly. ``compression_methods`` also exports a class
# named ``HOBF``, so a dotted string mock target can resolve to that class on
# some Python/mock versions instead of to ``compression_methods.HOBF``.
hobf_module = importlib.import_module("compression_methods.HOBF")


class FakeDynamicCache:
    # 功能：构造压缩测试使用的简化 KV 缓存容器。
    # 输入：self 为缓存实例；layers 为逐层 (key,value) 张量对。
    # 输出：返回 None；将每层 K、V 分别保存到 self.key_cache 和 self.value_cache。
    def __init__(self, layers):
        self.key_cache = [key for key, _ in layers]
        self.value_cache = [value for _, value in layers]

    # 功能：提供压缩器所需的旧式 KV 转换接口。
    # 输入：cls 为缓存类；layers 为逐层 (key,value) 张量对。
    # 输出：返回 cls(layers) 构造的 FakeDynamicCache 实例。
    @classmethod
    def from_legacy_cache(cls, layers):
        return cls(layers)


class DynamicHOBFTest(unittest.TestCase):
    # 功能：构造一组可重复的 HOBF 压缩测试输入。
    # 输入：无参数，静态方法；内部固定随机种子为 11。
    # 输出：(cache,prompt_mask,attentions,latent_steps)；单层 KV 为 (1,2,14,4)，提示掩码为 (1,12)，latent_steps=2。
    # 说明：attentions 含两个 latent 步的逐层注意力张量；数据由随机数生成。
    @staticmethod
    def make_inputs():
        torch.manual_seed(11)
        batch = 1
        heads = 2
        prompt_length = 12
        latent_steps = 2
        total_length = prompt_length + latent_steps
        head_dim = 4

        key = torch.randn(batch, heads, total_length, head_dim)
        value = torch.randn(batch, heads, total_length, head_dim)
        cache = FakeDynamicCache(((key, value),))
        prompt_mask = torch.ones(batch, prompt_length, dtype=torch.long)
        attentions = [
            [torch.rand(batch, heads, prompt_length + step + 1)]
            for step in range(latent_steps)
        ]
        return cache, prompt_mask, attentions, latent_steps

    # 功能：检查运行时预算覆盖构造时预算，并影响压缩后缓存长度。
    # 输入：self 为 unittest 测试实例；内部构造 sink_size=4、默认预算32的 HOBF，分别用预算3和5压缩构造缓存。
    # 输出：返回 None；通过断言检查预期行为，不符合预期时测试失败。
    # 说明：两次间 reset 压缩器；断言长度分别为 4+3+2 和 4+5+2，仅验证该构造用例。
    def test_runtime_budget_changes_cache_length(self):
        compressor = HOBF(sink_size=4, kv_budget=32, pca_rank=2)

        cache, prompt_mask, attentions, latent_steps = self.make_inputs()
        with patch.object(hobf_module, "DynamicCache", FakeDynamicCache):
            compressed, _ = compressor.compress(
                past_key_values=cache,
                latent_steps=latent_steps,
                prompt_mask=prompt_mask,
                all_steps_attentions=attentions,
                kv_budget=3,
            )
        self.assertEqual(compressed.key_cache[0].shape[-2], 4 + 3 + latent_steps)

        compressor.reset()
        cache, prompt_mask, attentions, latent_steps = self.make_inputs()
        with patch.object(hobf_module, "DynamicCache", FakeDynamicCache):
            compressed, _ = compressor.compress(
                past_key_values=cache,
                latent_steps=latent_steps,
                prompt_mask=prompt_mask,
                all_steps_attentions=attentions,
                kv_budget=5,
            )
        self.assertEqual(compressed.key_cache[0].shape[-2], 4 + 5 + latent_steps)


if __name__ == "__main__":
    unittest.main()
