# 文件作用：提供数据采集所需的题目抽样与索引管理工具。
# 主要职责：读取和写入题号清单，选择初筛失败题及成功抽查题进入完整搜索。

"""Pure helpers for reproducible Router data screening and selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple


# 功能：从原数据集题目索引范围中随机抽取互不重复的题目。
# 输入：dataset_size 为数据集题数，sample_count 为抽题数（-1 表示全部），seed 为独立抽样种子。
# 输出：抽样顺序下的整数索引元组；索引范围是 0 到 dataset_size - 1。
# 注意：使用局部 Random 实例；即使抽取全部题目，也会返回随机排列而不是原始顺序。
def sample_dataset_indices(
    dataset_size: int,
    sample_count: int,
    seed: int,
) -> Tuple[int, ...]:
    """Sample source-example indices without replacement in a stable order."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if sample_count == -1:
        sample_count = dataset_size
    if sample_count <= 0:
        raise ValueError("sample_count must be positive or -1")
    if sample_count > dataset_size:
        raise ValueError(
            f"sample_count={sample_count} exceeds dataset_size={dataset_size}"
        )

    rng = random.Random(int(seed))
    return tuple(rng.sample(range(dataset_size), sample_count))


# 功能：读取保存原数据集题目索引的文本文件。
# 输入：path 为 UTF-8 文本路径；每个有效行包含一个非负整数，允许空行和 # 注释。
# 输出：保持文件出现顺序的索引元组；空文件、无效整数、负数或重复索引会报错。
# 注意：只读取题号，不加载题目文本，也不调用模型。
def read_index_file(path: str | Path) -> Tuple[int, ...]:
    """Read one non-negative dataset index per line, preserving file order."""

    source = Path(path)
    indices = []
    seen = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw_line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            index = int(text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid dataset index at {source}:{line_number}: {text!r}"
            ) from exc
        if index < 0:
            raise ValueError(
                f"Dataset index must be non-negative at {source}:{line_number}"
            )
        if index in seen:
            raise ValueError(f"Duplicate dataset index {index} in {source}")
        seen.add(index)
        indices.append(index)
    if not indices:
        raise ValueError(f"No dataset indices found in {source}")
    return tuple(indices)


# 功能：把题目索引列表保存成供后续搜索程序读取的文本文件。
# 输入：path 是目标路径，indices 是可迭代索引；每个值会转换为整数。
# 输出：无返回值（None）；覆盖写入 UTF-8 文件，每个索引占一行。
# 注意：调用方负责提供存在的父目录和有效索引；这里不进行抽样或重复检查。
def write_index_file(path: str | Path, indices: Iterable[int]) -> None:
    destination = Path(path)
    values = [str(int(index)) for index in indices]
    destination.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


# 功能：根据最小预算筛选结果，分别抽取失败题与成功抽查题组成完整搜索名单。
# 输入：selected_indices 为已筛题号，minimum_budget_correct 为题号到正确性的映射；hard_count/easy_count 控制两组数量，seed 控制打乱。
# 输出：返回（选中失败题元组、选中成功题元组、前两组依次拼接的完整搜索题号元组）。
# 注意：数量 -1 表示保留该组全部题；其他非负数量最多取这么多，失败题是否全部搜索取决于 hard_count。
def choose_full_search_indices(
    selected_indices: Sequence[int],
    minimum_budget_correct: Mapping[int, bool],
    hard_count: int,
    easy_count: int,
    seed: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Choose hard candidates and an easy audit sample for 64-way search.

    ``hard`` means the all-minimum schedule failed, so exhaustive search may
    discover a non-minimum label. ``easy`` examples are audited to collect
    off-trajectory reserve-state labels and keep a natural control group.
    """

    if hard_count < -1 or easy_count < -1:
        raise ValueError("hard_count and easy_count must be non-negative or -1")

    missing = [index for index in selected_indices if index not in minimum_budget_correct]
    if missing:
        raise ValueError(f"Missing screening outcomes for indices: {missing[:5]}")

    hard = [index for index in selected_indices if not minimum_budget_correct[index]]
    easy = [index for index in selected_indices if minimum_budget_correct[index]]
    rng = random.Random(int(seed))
    rng.shuffle(hard)
    rng.shuffle(easy)

    selected_hard = tuple(hard if hard_count == -1 else hard[:hard_count])
    selected_easy = tuple(easy if easy_count == -1 else easy[:easy_count])
    return selected_hard, selected_easy, selected_hard + selected_easy
