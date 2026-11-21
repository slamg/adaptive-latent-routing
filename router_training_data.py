# 文件作用：将初筛和完整搜索记录整理为 Router 可训练样本。
# 主要职责：统一特征与标签字段、选择主轨迹或备用状态、处理来源划分，并封装PyTorch数据集。

"""Build leakage-safe BudgetRouter datasets from screening/Oracle shards."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from budget_router import parse_budget_choices


# 功能：加载数据目录中的逐题 PyTorch 分片，建立原始题号到分片内容的索引。
# 输入：folder 为含 samples 子目录的数据目录，id_key 指定分片题号字段（如 dataset_index 或 sample_id）。
# 输出：以整数题号为键的分片字典；缺少目录、没有分片或题号重复时会报错。
# 注意：通过 torch.load 映射到 CPU；这里读取保存结果，不运行题目或生成新标签。
def _load_shards(folder: str | Path, id_key: str) -> Dict[int, Dict]:
    sample_folder = Path(folder) / "samples"
    if not sample_folder.is_dir():
        raise FileNotFoundError(f"Missing shard directory: {sample_folder}")
    payloads: Dict[int, Dict] = {}
    for path in sorted(sample_folder.glob("sample_*.pt")):
        payload = torch.load(path, map_location="cpu")
        dataset_index = int(payload[id_key])
        if dataset_index in payloads:
            raise ValueError(f"Duplicate dataset index {dataset_index} in {sample_folder}")
        payloads[dataset_index] = payload
    if not payloads:
        raise ValueError(f"No sample_*.pt shards found in {sample_folder}")
    return payloads


# 功能：将单个代表类别转换为只有该类为真的布尔掩码。
# 输入：target_class 为目标类别索引，num_classes 为候选类别总数。
# 输出：长度 num_classes 的布尔元组，第 target_class 项为 True，其余项为 False。
# 注意：若索引超出范围会得到全 False，合法性检查由后续标准化函数负责。
def _one_hot_mask(target_class: int, num_classes: int) -> Tuple[bool, ...]:
    return tuple(index == int(target_class) for index in range(num_classes))


# 功能：将不同采集来源的单条状态记录转换为统一的训练样本字段。
# 输入：item 含特征、阶段及标签；dataset_index/task/split 定位题目，budget_choices 指定类别顺序，source_kind/origin 记录来源类型。
# 输入：fallback_prefix 用于补齐缺失前缀；optimal_mask 可显式提供并列标签，否则读取记录掩码或构造单标签掩码。
# 输出：标准字典，含 source_id、prefix、handoff_index、CPU float32 一维特征、预算/类别标签和最优预算掩码。
# 流程：检查特征形状与有限性、预算和类别对应关系及掩码，再组装字段；此处不做均值方差标准化。
def _normalise_example(
    item: Mapping,
    *,
    dataset_index: int,
    task: str,
    split: str,
    budget_choices: Tuple[int, ...],
    source_kind: str,
    origin: str,
    fallback_prefix: Tuple[int, ...] | None = None,
    optimal_mask: Sequence[bool] | None = None,
) -> Dict:
    feature = item["router_feature"].detach().to(dtype=torch.float32, device="cpu")
    if feature.dim() != 1:
        raise ValueError(
            f"router_feature must be 1D for source {dataset_index}, got {tuple(feature.shape)}"
        )
    if not torch.isfinite(feature).all():
        raise ValueError(f"Non-finite router feature for source {dataset_index}")

    handoff_index = int(item["handoff_index"])
    prefix = tuple(int(value) for value in item.get("prefix", fallback_prefix or ()))
    target_budget = item.get("target_budget")
    target_class = item.get("target_class")
    if target_budget is None or target_class is None:
        raise ValueError("Only labelled examples may be normalised")
    target_budget = int(target_budget)
    target_class = int(target_class)
    if budget_choices[target_class] != target_budget:
        raise ValueError(
            f"target class/budget mismatch: class={target_class}, budget={target_budget}"
        )

    if optimal_mask is None:
        optimal_mask = item.get("optimal_budget_mask")
    if optimal_mask is None:
        mask = _one_hot_mask(target_class, len(budget_choices))
    else:
        mask = tuple(bool(value) for value in optimal_mask)
    if len(mask) != len(budget_choices) or not any(mask):
        raise ValueError(f"Invalid optimal mask for source {dataset_index}: {mask}")

    return {
        "source_id": f"{task}:{split}:{dataset_index}",
        "dataset_index": dataset_index,
        "task": task,
        "split": split,
        "source_kind": source_kind,
        "origin": origin,
        "handoff_index": handoff_index,
        "prefix": prefix,
        "router_feature": feature,
        "target_budget": target_budget,
        "target_class": target_class,
        "optimal_budget_mask": mask,
        "label_status": item.get("label_status"),
    }


# 功能：把初筛和完整搜索分片转换为不重复的主轨迹或主轨迹加备用状态训练记录。
# 输入：screen_folder/exhaustive_folder 为两种分片目录；view 为 primary 或 primary_reserve；budget_choices 定义类别顺序。
# 输出：返回（统一状态记录列表、包含样本/题数/类别/阶段及缺失搜索情况的统计字典）。
# 流程：完整搜索题按 view 选状态；primary_reserve 用备用状态替代主轨迹，未完整搜索的成功初筛题保留最小预算主轨迹。
# 注意：无成功标签状态不参与训练；按 (source_id, prefix) 检查重复，并检查各记录特征形状一致。
def build_router_examples(
    screen_folder: str | Path,
    exhaustive_folder: str | Path,
    *,
    view: str,
    budget_choices: Sequence[int] = (8, 16, 32, 64),
) -> tuple[list[Dict], Dict]:
    """Build primary-only or primary-plus-reserve examples without duplicates.

    For exhaustive sources, the reserve view replaces (rather than appends to)
    canonical primary states. For screened easy sources without exhaustive
    shards, the certified all-minimum primary examples remain available.
    """

    if view not in {"primary", "primary_reserve"}:
        raise ValueError("view must be 'primary' or 'primary_reserve'")
    choices = parse_budget_choices(budget_choices)
    screen = _load_shards(screen_folder, "dataset_index")
    exhaustive = _load_shards(exhaustive_folder, "sample_id")

    records: list[Dict] = []
    source_kinds = Counter()
    excluded_all_incorrect = 0
    missing_exhaustive_hard = []

    all_indices = sorted(set(screen) | set(exhaustive))
    for dataset_index in all_indices:
        screen_payload = screen.get(dataset_index)
        exhaustive_payload = exhaustive.get(dataset_index)
        base = exhaustive_payload or screen_payload
        task = str(base["task"])
        split = str(base["split"])
        payload_choices = tuple(int(value) for value in base["budget_choices"])
        if payload_choices != choices:
            raise ValueError(
                f"Budget choices differ for source {dataset_index}: {payload_choices} != {choices}"
            )

        minimum_correct = (
            bool(screen_payload["minimum_budget_correct"])
            if screen_payload is not None
            else None
        )
        if exhaustive_payload is not None:
            source_kind = "easy" if minimum_correct is True else "hard"
            primary_items = exhaustive_payload.get("primary_examples", [])
            if not primary_items:
                excluded_all_incorrect += 1
                source_kinds["all_incorrect"] += 1
                continue

            reserve_items = exhaustive_payload["reserve_state_examples"]
            reserve_by_prefix = {
                tuple(item["prefix"]): item
                for item in reserve_items
                if item.get("target_budget") is not None
            }
            if view == "primary":
                selected = []
                for item in primary_items:
                    prefix = tuple(item["prefix"])
                    reserve_item = reserve_by_prefix.get(prefix)
                    selected.append(
                        _normalise_example(
                            item,
                            dataset_index=dataset_index,
                            task=task,
                            split=split,
                            budget_choices=choices,
                            source_kind=source_kind,
                            origin="exhaustive_primary",
                            optimal_mask=(
                                reserve_item.get("optimal_budget_mask")
                                if reserve_item is not None
                                else None
                            ),
                        )
                    )
            else:
                selected = [
                    _normalise_example(
                        item,
                        dataset_index=dataset_index,
                        task=task,
                        split=split,
                        budget_choices=choices,
                        source_kind=source_kind,
                        origin="exhaustive_reserve",
                    )
                    for item in reserve_items
                    if item.get("target_budget") is not None
                ]
        else:
            if not bool(screen_payload.get("minimum_budget_correct", False)):
                missing_exhaustive_hard.append(dataset_index)
                continue
            source_kind = "easy"
            minimum_budget = choices[0]
            selected = []
            for item in screen_payload.get("primary_examples", []):
                handoff = int(item["handoff_index"])
                selected.append(
                    _normalise_example(
                        item,
                        dataset_index=dataset_index,
                        task=task,
                        split=split,
                        budget_choices=choices,
                        source_kind=source_kind,
                        origin="screen_certified_primary",
                        fallback_prefix=tuple(minimum_budget for _ in range(handoff)),
                    )
                )

        source_kinds[source_kind] += 1
        records.extend(selected)

    deduplicated: Dict[tuple, Dict] = {}
    for record in records:
        key = (record["source_id"], record["prefix"])
        if key in deduplicated:
            raise ValueError(f"Duplicate Router state: {key}")
        deduplicated[key] = record
    records = list(deduplicated.values())

    feature_shapes = Counter(tuple(record["router_feature"].shape) for record in records)
    if len(feature_shapes) != 1:
        raise ValueError(f"Inconsistent router feature shapes: {dict(feature_shapes)}")
    report = {
        "view": view,
        "screen_shards": len(screen),
        "exhaustive_shards": len(exhaustive),
        "examples": len(records),
        "labelled_sources": len({record["source_id"] for record in records}),
        "source_kinds": dict(source_kinds),
        "excluded_all_incorrect": excluded_all_incorrect,
        "missing_exhaustive_hard": missing_exhaustive_hard,
        "feature_shape": next(iter(feature_shapes)),
        "target_counts": dict(Counter(record["target_budget"] for record in records)),
        "handoff_counts": dict(Counter(record["handoff_index"] for record in records)),
        "tied_examples": sum(
            sum(record["optimal_budget_mask"]) > 1 for record in records
        ),
    }
    return records, report


# 功能：按来源题目划分训练集与验证集，并按 easy/hard 来源类别分层。
# 输入：records 为状态记录，validation_fraction 为 0 到 1 之间的比例，seed 为划分种子。
# 输出：返回（训练 source_id 元组、验证 source_id 元组）；两者互不重叠。
# 流程：先把多个状态归并为题目，再按来源类别打乱和切分；小分层保留至少一个训练来源，单题分层归训练。
# 注意：返回的是题目 ID，调用方再据此筛选状态，避免同题状态跨训练和验证泄漏。
def split_source_ids(
    records: Sequence[Mapping],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split by source question, stratified by easy/hard source kind."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    kind_by_source: Dict[str, str] = {}
    for record in records:
        source_id = str(record["source_id"])
        kind = str(record["source_kind"])
        previous = kind_by_source.setdefault(source_id, kind)
        if previous != kind:
            raise ValueError(f"Conflicting source kinds for {source_id}")

    grouped = defaultdict(list)
    for source_id, kind in kind_by_source.items():
        grouped[kind].append(source_id)

    rng = random.Random(int(seed))
    train, validation = [], []
    for kind in sorted(grouped):
        source_ids = sorted(grouped[kind])
        rng.shuffle(source_ids)
        count = max(1, round(len(source_ids) * validation_fraction))
        if count >= len(source_ids):
            count = len(source_ids) - 1
        if count <= 0:
            train.extend(source_ids)
        else:
            validation.extend(source_ids[:count])
            train.extend(source_ids[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if set(train) & set(validation):
        raise RuntimeError("Source leakage between train and validation splits")
    return tuple(train), tuple(validation)


class RouterFeatureDataset(Dataset):
    # 功能：将统一状态记录封装成可被 PyTorch DataLoader 访问的数据集。
    # 输入：records 为非空状态记录序列，每条记录应包含特征、阶段、标签掩码和来源信息。
    # 输出：无返回值（None）；把序列转为列表保存在 self.records，空序列会报错。
    # 注意：这里只复制外层列表，不深复制记录或特征张量，也不重新划分训练/验证。
    def __init__(self, records: Sequence[Mapping]) -> None:
        if not records:
            raise ValueError("RouterFeatureDataset requires at least one record")
        self.records = list(records)

    # 功能：向 DataLoader 或调用方提供当前数据集的状态样本数量。
    # 输入：无额外参数，读取 self.records。
    # 输出：整数样本条数，即记录列表长度。
    # 注意：这是历史状态数；同一题可有多条状态，因此不等于独立题目数。
    def __len__(self) -> int:
        return len(self.records)

    # 功能：取出指定历史状态，并整理成训练批次所需的输入和监督字段。
    # 输入：index 为记录列表索引，self.records 中该记录已由数据整理阶段统一字段。
    # 输出：字典含 router_feature、整数 handoff_index/target_class、布尔张量 optimal_budget_mask、source_kind 和 source_id。
    # 注意：保留完整最优掩码供并列标签损失使用，target_class 仅为代表标签；此处不执行网络前向。
    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "router_feature": record["router_feature"],
            "handoff_index": int(record["handoff_index"]),
            "target_class": int(record["target_class"]),
            "optimal_budget_mask": torch.tensor(
                record["optimal_budget_mask"], dtype=torch.bool
            ),
            "source_kind": record["source_kind"],
            "source_id": record["source_id"],
        }
