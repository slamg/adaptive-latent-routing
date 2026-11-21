# 文件作用：在已保存的历史状态上诊断 Router 输入特征的判别能力。
# 主要职责：运行线性分类器、近邻和类中心等对照，输出分类指标及可视化所需数据。

"""Diagnose whether saved Router features contain budget-prediction signal.

This script is deliberately independent from Router training.  It reuses the
same source-level split, fits every transformation on training sources only,
and evaluates simple probes on unseen validation sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from budget_router import parse_budget_choices
from router_training_data import build_router_examples, split_source_ids
from utils import auto_device, set_seed


# 功能：定义离线历史状态特征诊断所需的命令行参数及默认设置。
# 输入：无显式参数；在这里配置数据路径、来源划分、探针训练参数、kNN、设备和绘图选项。
# 输出：ArgumentParser 对象；这里只定义参数，调用方再通过 parse_args 解析命令行。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure four-way and B=8-vs-larger Router feature separability."
    )
    parser.add_argument("--screen_folder", required=True)
    parser.add_argument("--exhaustive_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--view", choices=["primary", "primary_reserve"], default="primary_reserve"
    )
    parser.add_argument("--budget_choices", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear_epochs", type=int, default=100)
    parser.add_argument("--linear_lr", type=float, default=1e-2)
    parser.add_argument("--linear_weight_decay", type=float, default=1e-3)
    parser.add_argument("--knn_k", type=int, nargs="+", default=[1, 3, 5, 11])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Do not attempt optional PCA PNG output (CSV coordinates are still saved).",
    )
    return parser


# 功能：把历史状态记录堆叠成分类诊断所需张量，并可转换为 B8/非 B8 二分类。
# 输入：records 提供 router_feature、handoff_index、target_class、optimal_budget_mask；binary 决定二分类或原预算分类。
# 输出：(features[N,D], handoffs[N], labels[N], masks[N,C])；特征为浮点，阶段/标签为整数，掩码为布尔。
# 口径：binary=True 时以原掩码第 0 列是否允许 B8 定义类别 0/1；四分类保留完整并列最优动作集合。
def _records_to_tensors(records: Sequence[Mapping], *, binary: bool) -> tuple:
    features = torch.stack([record["router_feature"] for record in records]).float()
    handoffs = torch.tensor([record["handoff_index"] for record in records], dtype=torch.long)
    masks = torch.tensor(
        [record["optimal_budget_mask"] for record in records], dtype=torch.bool
    )
    if binary:
        # 若 B=8 也属于 Oracle 最优集合，通信上应优先视为“B=8 足够”。
        labels = (~masks[:, 0]).long()
        binary_masks = torch.zeros((len(records), 2), dtype=torch.bool)
        binary_masks[torch.arange(len(records)), labels] = True
        masks = binary_masks
    else:
        labels = torch.tensor([record["target_class"] for record in records], dtype=torch.long)
    return features, handoffs, labels, masks


# 功能：仅用训练特征估计逐维标准化参数。
# 输入：features 是 [N,D] 的训练特征矩阵。
# 输出：(mean[D], scale[D])；scale 使用总体标准差，过小的标准差替换为 1，避免除零。
def _fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return mean, scale


# 功能：按训练集统计量标准化输入，并按需加入通信阶段信息。
# 输入：features[N,D]、handoffs[N]、训练 mean[D]/scale[D]；append_handoff 控制是否拼接 3 维阶段 one-hot。
# 输出：[N,D] 或 [N,D+3] 特征；验证集复用训练集参数，不在验证数据上重新拟合。
def _transform(
    features: torch.Tensor,
    handoffs: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    append_handoff: bool,
) -> torch.Tensor:
    transformed = (features - mean) / scale
    if append_handoff:
        one_hot = torch.nn.functional.one_hot(handoffs, num_classes=3).float()
        transformed = torch.cat((transformed, one_hot), dim=1)
    return transformed


# 功能：同时统计代表标签分类表现与预测动作是否属于完整最优标签集合。
# 输入：predictions[N]、labels[N] 为类别索引，optimal_masks[N,C] 标记允许动作，num_classes 指定 C。
# 输出：含准确率、集合兼容率、各类召回和混淆矩阵的字典；二分类额外给出非 B8 精确率/召回率和平衡准确率。
# 口径：混淆矩阵行是真实代表类别、列是预测类别；宏平均只包含有真实样本的类别。
def _classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    optimal_masks: torch.Tensor,
    num_classes: int,
) -> Dict:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(labels.cpu(), predictions.cpu()):
        confusion[int(target), int(prediction)] += 1
    compatible = optimal_masks.gather(1, predictions.cpu().unsqueeze(1)).squeeze(1)
    recalls = []
    per_class = {}
    for class_index in range(num_classes):
        support = int(confusion[class_index].sum())
        recall = float(confusion[class_index, class_index] / support) if support else None
        per_class[str(class_index)] = {"support": support, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    result = {
        "accuracy": float((predictions.cpu() == labels.cpu()).float().mean()),
        "compatible_accuracy": float(compatible.float().mean()),
        "macro_recall": float(sum(recalls) / max(len(recalls), 1)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "predicted_counts": dict(Counter(int(value) for value in predictions.cpu().tolist())),
    }
    if num_classes == 2:
        tp = int(confusion[1, 1])
        fp = int(confusion[0, 1])
        fn = int(confusion[1, 0])
        result.update(
            {
                "larger_budget_precision": tp / max(tp + fp, 1),
                "larger_budget_recall": tp / max(tp + fn, 1),
                "balanced_accuracy": result["macro_recall"],
            }
        )
    return result


# 功能：用训练集各类中心检查特征分离程度，并评估最近中心分类。
# 输入：train_features[N,D]/validation_features[M,D]，对应 labels 和最优 masks，以及 num_classes。
# 输出：(report, pairwise, predictions[M])；分别为训练/验证指标及类半径、两两中心距离表、验证预测。
# 流程：中心只按训练代表标签拟合；距离除以两类半径平方和的平方根得到分离度，训练缺类时直接报错。
def _centroid_diagnostics(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_masks: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_masks: torch.Tensor,
    num_classes: int,
) -> tuple[Dict, list[Dict], torch.Tensor]:
    centers = []
    radii = []
    class_counts = []
    for class_index in range(num_classes):
        members = train_features[train_labels == class_index]
        if not len(members):
            raise ValueError(f"No training examples for class {class_index}")
        center = members.mean(dim=0)
        centers.append(center)
        radii.append(float(torch.linalg.vector_norm(members - center, dim=1).mean()))
        class_counts.append(len(members))
    centers_tensor = torch.stack(centers)

    pairwise = []
    for left in range(num_classes):
        for right in range(left + 1, num_classes):
            euclidean = float(torch.linalg.vector_norm(centers[left] - centers[right]))
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    centers[left].unsqueeze(0), centers[right].unsqueeze(0)
                ).item()
            )
            pooled_radius = math.sqrt(radii[left] ** 2 + radii[right] ** 2)
            pairwise.append(
                {
                    "class_left": left,
                    "class_right": right,
                    "centroid_euclidean": euclidean,
                    "centroid_cosine_similarity": cosine,
                    "left_mean_radius": radii[left],
                    "right_mean_radius": radii[right],
                    "normalised_separation": euclidean / max(pooled_radius, 1e-12),
                }
            )

    distances = torch.cdist(validation_features, centers_tensor)
    predictions = distances.argmin(dim=1)
    train_predictions = torch.cdist(train_features, centers_tensor).argmin(dim=1)
    metrics = _classification_metrics(
        predictions, validation_labels, validation_masks, num_classes
    )
    report = {
        "train_class_counts": class_counts,
        "mean_class_radii": radii,
        "mean_normalised_pairwise_separation": float(
            np.mean([row["normalised_separation"] for row in pairwise])
        ),
        "train": _classification_metrics(
            train_predictions,
            train_labels,
            train_masks,
            num_classes,
        ),
        "validation": metrics,
    }
    return report, pairwise, predictions


# 功能：用余弦相似度最近邻诊断特征是否能区分预算类别。
# 输入：train_features[N,D]、train_labels[N]、validation_features[M,D]；k 为邻居数，num_classes 为类别数。
# 输出：验证预测类别 [M]；balanced=True 时各类票数再除以该类训练样本数，降低多数类票数优势。
# 流程：特征先做 L2 归一化；实际 k 不超过训练样本数，不训练 Router 或基础模型。
def _knn_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    k: int,
    num_classes: int,
    balanced: bool,
) -> torch.Tensor:
    k = min(int(k), len(train_features))
    train_norm = torch.nn.functional.normalize(train_features, dim=1)
    validation_norm = torch.nn.functional.normalize(validation_features, dim=1)
    similarities = validation_norm @ train_norm.T
    neighbour_indices = similarities.topk(k=k, dim=1).indices
    neighbour_labels = train_labels[neighbour_indices]
    votes = torch.nn.functional.one_hot(neighbour_labels, num_classes=num_classes).float().sum(1)
    if balanced:
        support = torch.bincount(train_labels, minlength=num_classes).float().clamp_min(1)
        votes = votes / support.unsqueeze(0)
    return votes.argmax(dim=1)


# 功能：在固定历史状态特征上训练一个线性分类器，诊断信息可分性。
# 输入：训练/验证 features、代表 labels、完整最优 masks；num_classes、epochs、learning_rate、weight_decay、device 控制训练。
# 输出：(metrics, history)，metrics 含训练和验证指标，history 记录部分轮次损失；不返回或保存模型权重。
# 损失：对最优动作集合的概率和取负对数，并按最优掩码支持度计算类别/样本权重；代表标签用于指标而非单标签监督。
# 边界：固定训练到指定轮数后评估，不按验证分数选最佳轮，也不把预测接回完整解题流程。
def _linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_masks: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_masks: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[Dict, list[Dict]]:
    model = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    features = train_features.to(device)
    masks = train_masks.to(device)
    support = masks.float().sum(dim=0).clamp_min(1)
    class_weights = (support.max() / support).sqrt()
    class_weights /= class_weights.mean()
    sample_weights = (masks.float() * class_weights).sum(dim=1) / masks.sum(dim=1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        log_probabilities = torch.log_softmax(logits, dim=1)
        losses = -torch.logsumexp(
            log_probabilities.masked_fill(~masks, float("-inf")), dim=1
        )
        loss = (losses * sample_weights).sum() / sample_weights.sum()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu())})

    model.eval()
    with torch.no_grad():
        predictions = model(validation_features.to(device)).argmax(dim=1).cpu()
    with torch.no_grad():
        train_predictions = model(train_features.to(device)).argmax(dim=1).cpu()
    return (
        {
            "train": _classification_metrics(
                train_predictions, train_labels, train_masks, num_classes
            ),
            "validation": _classification_metrics(
                predictions, validation_labels, validation_masks, num_classes
            ),
        },
        history,
    )


# 功能：用训练集拟合二维 PCA 投影，便于对照训练与验证特征分布。
# 输入：train_features[N,D] 和 validation_features[M,D]。
# 输出：(train_coordinates[N,2], validation_coordinates[M,2], explained[2])，最后一项为各投影分量占训练总能量的比例。
# 注意：验证特征沿用训练均值和投影方向，不参与拟合。
def _pca_coordinates(
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered_train = train_features - train_features.mean(dim=0)
    _, singular_values, components = torch.pca_lowrank(centered_train, q=2, center=False)
    train_coordinates = centered_train @ components
    validation_coordinates = (
        validation_features - train_features.mean(dim=0)
    ) @ components
    explained = singular_values.square()
    explained = explained / centered_train.square().sum().clamp_min(1e-12)
    return train_coordinates, validation_coordinates, explained


# 功能：把结构一致的字典记录写成带表头的 CSV。
# 输入：path 是目标 Path；rows 是字典序列，第一行键决定列名。
# 输出：返回 None；非空时写 UTF-8 文件，空序列则直接返回且不创建文件。
def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 功能：对一个任务及通信阶段范围，组织中心、kNN、线性探针和 PCA 诊断。
# 输入：scope_name、训练/验证 records；binary、choices 指定类别口径，args/device/output_folder 指定运行与输出设置。
# 输出：该范围的诊断字典；另写中心距离、PCA 坐标 CSV，并按配置绘图。训练缺类时返回不足标记并跳过模型诊断。
# 流程：标准化仅拟合训练数据；overall 范围拼接阶段 one-hot，单阶段范围不再拼接。
# 边界：PCA/绘图库问题会被记录，不等同于所有分类诊断失败。
def _diagnose_scope(
    scope_name: str,
    train_records: Sequence[Mapping],
    validation_records: Sequence[Mapping],
    *,
    binary: bool,
    choices: Sequence[int],
    args: argparse.Namespace,
    output_folder: Path,
    device: torch.device,
) -> Dict:
    train_x, train_h, train_y, train_masks = _records_to_tensors(
        train_records, binary=binary
    )
    val_x, val_h, val_y, val_masks = _records_to_tensors(
        validation_records, binary=binary
    )
    num_classes = 2 if binary else len(choices)
    mean, scale = _fit_standardizer(train_x)
    append_handoff = scope_name == "overall"
    train_x = _transform(train_x, train_h, mean, scale, append_handoff=append_handoff)
    val_x = _transform(val_x, val_h, mean, scale, append_handoff=append_handoff)

    present_train = set(train_y.tolist())
    present_validation = set(val_y.tolist())
    missing_train = sorted(set(range(num_classes)) - present_train)
    if missing_train:
        return {
            "status": "insufficient_training_classes",
            "missing_training_classes": missing_train,
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
        }

    centroid, pairwise, _ = _centroid_diagnostics(
        train_x, train_y, train_masks, val_x, val_y, val_masks, num_classes
    )
    pairwise_rows = [{"scope": scope_name, **row} for row in pairwise]
    task_name = "binary_b8_sufficient" if binary else "four_way"
    _write_csv(output_folder / f"centroid_distances_{task_name}_{scope_name}.csv", pairwise_rows)

    knn = {}
    for k in args.knn_k:
        for balanced in (False, True):
            key = f"k={k}_{'balanced' if balanced else 'plain'}"
            prediction = _knn_predictions(
                train_x,
                train_y,
                val_x,
                k=k,
                num_classes=num_classes,
                balanced=balanced,
            )
            knn[key] = _classification_metrics(
                prediction, val_y, val_masks, num_classes
            )

    linear, linear_history = _linear_probe(
        train_x,
        train_y,
        train_masks,
        val_x,
        val_y,
        val_masks,
        num_classes=num_classes,
        epochs=args.linear_epochs,
        learning_rate=args.linear_lr,
        weight_decay=args.linear_weight_decay,
        device=device,
    )

    pca_path = output_folder / f"pca_{task_name}_{scope_name}.csv"
    pca_status = "written"
    plot_status = "disabled"
    explained = torch.empty(0)
    try:
        train_pca, validation_pca, explained = _pca_coordinates(train_x, val_x)
        pca_rows = []
        for split, coordinates, labels in (
            ("train", train_pca, train_y),
            ("validation", validation_pca, val_y),
        ):
            for index in range(len(coordinates)):
                pca_rows.append(
                    {
                        "split": split,
                        "pc1": float(coordinates[index, 0]),
                        "pc2": float(coordinates[index, 1]),
                        "class_index": int(labels[index]),
                    }
                )
        _write_csv(pca_path, pca_rows)
        if not args.no_plots:
            try:
                import matplotlib.pyplot as plt

                figure, axis = plt.subplots(figsize=(8, 6))
                for class_index in range(num_classes):
                    selected = validation_pca[val_y == class_index]
                    label = (
                        ("B=8 sufficient" if class_index == 0 else "B>8 required")
                        if binary
                        else f"B={choices[class_index]}"
                    )
                    axis.scatter(
                        selected[:, 0].numpy(),
                        selected[:, 1].numpy(),
                        s=18,
                        alpha=0.7,
                        label=label,
                    )
                axis.set_title(f"Validation PCA: {task_name}, {scope_name}")
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")
                axis.legend()
                figure.tight_layout()
                figure.savefig(output_folder / f"pca_{task_name}_{scope_name}.png", dpi=160)
                plt.close(figure)
                plot_status = "written"
            except ImportError:
                plot_status = "matplotlib_not_installed"
    except RuntimeError as error:
        # PCA 是辅助可视化；缺少 LAPACK 等情况不能中断核心数值诊断。
        pca_status = f"unavailable: {error}"
        plot_status = "pca_unavailable"

    return {
        "status": "ok",
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "train_class_counts": dict(Counter(int(value) for value in train_y.tolist())),
        "validation_class_counts": dict(Counter(int(value) for value in val_y.tolist())),
        "missing_validation_classes": sorted(set(range(num_classes)) - present_validation),
        "centroid": centroid,
        "knn": knn,
        "linear_probe": linear,
        "linear_history": linear_history,
        "pca_explained_variance_ratio": explained.tolist(),
        "pca_csv": str(pca_path),
        "pca_status": pca_status,
        "pca_plot_status": plot_status,
    }


# 功能：把多任务、多阶段诊断结果整理成便于写 CSV 的平面行。
# 输入：report 含各任务/范围结果；choices 参数保留在接口中，当前函数体未使用。
# 输出：字典列表，逐行对应一种探针的验证指标；缺类等未运行情况输出 unavailable 及原因。
def _summary_rows(report: Mapping, choices: Sequence[int]) -> list[Dict]:
    rows = []
    for task_name, scopes in report["diagnostics"].items():
        for scope_name, result in scopes.items():
            if result["status"] != "ok":
                rows.append(
                    {
                        "task": task_name,
                        "scope": scope_name,
                        "probe": "unavailable",
                        "accuracy": "",
                        "compatible_accuracy": "",
                        "macro_recall": "",
                        "larger_budget_recall": "",
                        "note": result["status"],
                    }
                )
                continue
            probes = {"nearest_centroid": result["centroid"]["validation"]}
            probes.update({f"knn_{key}": value for key, value in result["knn"].items()})
            probes["linear_probe"] = result["linear_probe"]["validation"]
            for probe_name, metrics in probes.items():
                rows.append(
                    {
                        "task": task_name,
                        "scope": scope_name,
                        "probe": probe_name,
                        "accuracy": metrics["accuracy"],
                        "compatible_accuracy": metrics["compatible_accuracy"],
                        "macro_recall": metrics["macro_recall"],
                        "larger_budget_recall": metrics.get("larger_budget_recall", ""),
                        "note": "",
                    }
                )
    return rows


# 功能：执行离线历史状态特征诊断，从数据整理一直到结果报告输出。
# 输入：无显式参数；从命令行读取筛选/搜索数据目录、预算候选、来源划分、探针超参数和设备。
# 输出：返回 None；保存 separability_report.json、probe_summary.csv 及各范围诊断文件。
# 流程：固定随机种子并按来源题划分，检查题目不交叉，再分别运行预算分类/二分类及整体/各通信阶段诊断。
def main() -> None:
    args = build_parser().parse_args()
    if args.linear_epochs <= 0:
        raise ValueError("linear_epochs must be positive")
    if not args.knn_k or any(value <= 0 for value in args.knn_k):
        raise ValueError("knn_k values must be positive")
    choices = parse_budget_choices(args.budget_choices)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    records, data_report = build_router_examples(
        args.screen_folder,
        args.exhaustive_folder,
        view=args.view,
        budget_choices=choices,
    )
    train_sources, validation_sources = split_source_ids(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    train_source_set = set(train_sources)
    validation_source_set = set(validation_sources)
    train_records = [record for record in records if record["source_id"] in train_source_set]
    validation_records = [
        record for record in records if record["source_id"] in validation_source_set
    ]
    if {record["source_id"] for record in train_records} & {
        record["source_id"] for record in validation_records
    }:
        raise RuntimeError("Source-level train/validation leakage detected")

    device = auto_device(args.device)
    report = {
        "schema_version": 1,
        "purpose": "router_feature_separability",
        "budget_choices": list(choices),
        "data": data_report,
        "split": {
            "train_sources": len(train_sources),
            "validation_sources": len(validation_sources),
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
            "split_seed": args.split_seed,
        },
        "diagnostics": {"four_way": {}, "binary_b8_sufficient": {}},
    }
    scopes = {"overall": (train_records, validation_records)}
    for handoff_index in range(3):
        scopes[f"handoff_{handoff_index}"] = (
            [r for r in train_records if int(r["handoff_index"]) == handoff_index],
            [r for r in validation_records if int(r["handoff_index"]) == handoff_index],
        )

    for binary, task_name in ((False, "four_way"), (True, "binary_b8_sufficient")):
        for scope_name, (scope_train, scope_validation) in scopes.items():
            print(f"Diagnosing {task_name}/{scope_name} ...", flush=True)
            report["diagnostics"][task_name][scope_name] = _diagnose_scope(
                scope_name,
                scope_train,
                scope_validation,
                binary=binary,
                choices=choices,
                args=args,
                output_folder=output_folder,
                device=device,
            )

    report_path = output_folder / "separability_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_rows = _summary_rows(report, choices)
    _write_csv(output_folder / "probe_summary.csv", summary_rows)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "summary": str(output_folder / "probe_summary.csv"),
                "train_examples": len(train_records),
                "validation_examples": len(validation_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
