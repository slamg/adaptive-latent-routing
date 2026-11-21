# 文件作用：从已有完整搜索记录汇总固定预算与 Oracle 的收益空间。
# 主要职责：统计成功题数量、预算成本及共同答对题的通信收益，不重新执行模型推理。

"""Print Markdown comparisons from trusted local Oracle rollout shards."""
import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import torch


BUDGETS = (8, 16, 32, 64)
SCHEDULES = set(product(BUDGETS, repeat=3))


# 功能：读取已有完整搜索分片，检查 64 种预算组合与成本，并从成功轨迹重新确定最小总预算方案。
# 输入：paths 为可信本地 .pt 文件路径序列；每个文件应保存一题 GSM8K train 的完整搜索结果。
# 输出：返回 samples 字典，键为 sample_id，值为 (runs, oracle)；runs 按三次预算元组索引，oracle 为最优组合元组或 None。
# 规则：预算和相同则按元组字典序选代表方案；全部失败则 oracle 为 None；覆盖、成本或已保存代表方案不符时抛出异常。
def summarize(paths):
    samples = {}
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        idx = data["sample_id"]
        if idx in samples or data["task"] != "gsm8k" or data["split"] != "train":
            raise ValueError(f"Unexpected or duplicate sample: {path}")
        raw = data["rollouts"]
        runs = {tuple(r["schedule"]): r for r in raw}
        if len(raw) != len(SCHEDULES) or set(runs) != SCHEDULES:
            raise ValueError(f"Incomplete schedule grid: {path}")
        for key, run in runs.items():
            if run["cost"] != sum(key) or run["communication_overhead"] <= 0:
                raise ValueError(f"Invalid cost or missing KV measurement: {path} {key}")
        oracle = min((k for k, r in runs.items() if r["correct"]),
                     key=lambda k: (sum(k), k), default=None)
        saved = data["global_oracle"]["canonical_schedule"]
        if oracle != (tuple(saved) if saved is not None else None):
            raise ValueError(f"Oracle schedule mismatch: {path}")
        samples[idx] = (runs, oracle)
    return samples


# 功能：计算一组结果记录中某个数值字段的算术平均值。
# 输入：records 为非空结果记录集合，key 为各记录共有的数值字段名，例如 cost 或 communication_overhead。
# 输出：返回该字段的平均值；不筛选正确性，也不对空集合设置默认值，筛选由调用者完成。
def avg(records, key):
    return mean(record[key] for record in records)


# 功能：把数值格式化为表格中使用的两位小数文本。
# 输入：value 为支持浮点格式化的数值；此函数不处理 None 等缺失值。
# 输出：返回保留两位小数的字符串，仅影响显示精度，不修改原始数值。
def fmt(value):
    return f"{value:.2f}"


# 功能：输出固定 B8/B16/B32/B64 与 Oracle 的成功率表，以及各固定预算和 Oracle 的共同答对题通信对比表。
# 输入：samples 为 summarize 返回的非空字典，每题包含完整搜索 runs 及 Oracle 代表组合或 None。
# 输出：返回 None；将两张 Markdown 表打印到标准输出，不创建报告文件，也不运行模型。
# 口径：预算与 KV 均值按成功题计算；节省率在双方共同答对的同一题集上计算为 1 - Oracle平均KV/固定平均KV，没有可比题则显示横线。
def write_tables(samples):
    n = len(samples)
    oracle_rows = [runs[k] for runs, k in samples.values() if k is not None]
    print("| 策略 | 答对题数 | 答题正确率 | 答对题平均总预算 | 答对题平均 KV（MiB/题） |")
    print("|---|---:|---:|---:|---:|")
    for budget in (*BUDGETS, None):
        label = "Oracle" if budget is None else f"固定 B={budget}"
        rows = oracle_rows if budget is None else [
            runs[(budget,) * 3] for runs, _ in samples.values() if runs[(budget,) * 3]["correct"]
        ]
        budget_avg, kv = (fmt(avg(rows, k)) for k in ("cost", "communication_overhead")) if rows else ("—", "—")
        print(f"| {label} | {len(rows)} / {n} | {len(rows)/n:.2%} | {budget_avg} | {kv} |")

    print("\n| 固定预算对照 | 共同答对题数 | 固定平均总预算 | Oracle 平均总预算 | 固定平均 KV（MiB/题） | Oracle 平均 KV（MiB/题） | KV 节省比例 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for budget in BUDGETS:
        key = (budget,) * 3
        pairs = [(runs[key], runs[k]) for runs, k in samples.values() if k is not None and runs[key]["correct"]]
        if not pairs:
            print(f"| B={budget} | 0 | — | — | — | — | — |")
            continue
        fixed, oracle = zip(*pairs)
        fk, ok = (avg(rows, "communication_overhead") for rows in (fixed, oracle))
        print(f"| B={budget} | {len(pairs)} | {avg(fixed, 'cost'):.2f} | {avg(oracle, 'cost'):.2f} | {fk:.2f} | {ok:.2f} | {1-ok/fk:.2%} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_folder", type=Path)
    parser.add_argument("--expected-count", type=int, default=76)
    args = parser.parse_args()
    paths = sorted((args.oracle_folder / "samples").glob("sample_*.pt"))
    if len(paths) != args.expected_count or not paths:
        parser.error(f"Found {len(paths)} shards, expected {args.expected_count}; check the folder.")
    write_tables(summarize(paths))
