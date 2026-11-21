# 文件作用：使用已标注历史状态训练和验证预算 Router。
# 主要职责：构建数据加载器、计算集合标签损失与平衡权重，按分组集合兼容指标选模并保存权重和训练记录。

"""Train BudgetRouter on leakage-safe GSM8K Oracle features."""

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
from torch.utils.data import DataLoader

from budget_router import BudgetRouter, parse_budget_choices
from router_training_data import (
    RouterFeatureDataset,
    build_router_examples,
    split_source_ids,
)
from utils import auto_device, set_seed


# 功能：集中定义预算 Router 训练所需的数据、网络、优化器和划分参数。
# 输入：无显式参数；配置数据目录、预算候选、隐藏层/阶段嵌入维度、随机种子、轮数及类别平衡强度等。
# 输出：ArgumentParser 对象，调用方负责 parse_args；dry_run 表示只核对并整理数据，不更新权重。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the adaptive HOBF BudgetRouter.")
    parser.add_argument("--screen_folder", required=True)
    parser.add_argument("--exhaustive_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--view",
        choices=["primary", "primary_reserve"],
        default="primary_reserve",
    )
    parser.add_argument("--budget_choices", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--handoff_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--class_balance_power",
        type=float,
        default=0.5,
        help="0 disables class balancing; 0.5 applies inverse-square-root support weights.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build, validate, split, and report the data without updating Router weights.",
    )
    return parser


# 功能：计算每条状态预测落在完整 Oracle 最优动作集合之外的惩罚。
# 输入：logits[N,C] 是未归一化分数，optimal_mask[N,C] 是布尔掩码，每行至少有一个最优动作。
# 输出：逐样本损失 [N]，公式为 -log(最优动作集合内的 softmax 概率之和)，保留梯度用于反向传播。
# 含义：并列最优动作都可被接受；不是只对某一个代表标签做交叉熵。形状不匹配或空最优集合会报错。
def multi_positive_losses(logits: torch.Tensor, optimal_mask: torch.Tensor) -> torch.Tensor:
    """Negative log probability assigned to any Oracle-optimal action."""

    if logits.shape != optimal_mask.shape:
        raise ValueError(
            f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(optimal_mask.shape)}"
        )
    if not optimal_mask.any(dim=-1).all():
        raise ValueError("Every example must contain at least one optimal class")
    log_probs = torch.log_softmax(logits, dim=-1)
    masked = log_probs.masked_fill(~optimal_mask, float("-inf"))
    return -torch.logsumexp(masked, dim=-1)


# 功能：仅根据训练状态，计算预算类别权重和三个通信阶段的权重。
# 输入：records 含 optimal_budget_mask 和 handoff_index；num_classes 为候选数，class_balance_power 控制类别逆频率强度。
# 输出：(class_weights[C], handoff_weights[3], report)，前两项为浮点张量，report 保存支持度和权重的普通列表。
# 口径：一条并列最优状态把 1 份支持均摊给各最优类；阶段按状态数统计。任一类别或阶段无训练支持时拒绝训练。
def compute_balance_weights(
    records: Sequence[Mapping],
    num_classes: int,
    class_balance_power: float,
) -> tuple[torch.Tensor, torch.Tensor, Dict]:
    class_support = torch.zeros(num_classes, dtype=torch.float64)
    handoff_support = torch.zeros(3, dtype=torch.float64)
    for record in records:
        mask = torch.tensor(record["optimal_budget_mask"], dtype=torch.float64)
        class_support += mask / mask.sum()
        handoff_support[int(record["handoff_index"])] += 1
    if (class_support <= 0).any():
        raise ValueError(f"Training split has an unsupported class: {class_support.tolist()}")
    if (handoff_support <= 0).any():
        raise ValueError(f"Training split has an unsupported handoff: {handoff_support.tolist()}")

    maximum = class_support.max()
    class_weights = (maximum / class_support).pow(float(class_balance_power))
    class_weights /= class_weights.mean()
    handoff_weights = handoff_support.sum() / (len(handoff_support) * handoff_support)
    handoff_weights /= (
        handoff_weights * handoff_support / handoff_support.sum()
    ).sum()
    report = {
        "class_support": class_support.tolist(),
        "class_weights": class_weights.tolist(),
        "handoff_support": handoff_support.tolist(),
        "handoff_weights": handoff_weights.tolist(),
    }
    return class_weights.float(), handoff_weights.float(), report


# 功能：把逐状态集合损失按类别稀缺程度及通信阶段加权为一个训练标量。
# 输入：losses[N]、optimal_mask[N,C]、handoff_ids[N]，以及训练集得到的 class_weights[C]、handoff_weights[3]。
# 输出：可反向传播的标量损失；每条状态先对其最优类权重取均值，再乘对应阶段权重，最后按样本权重和归一化。
def weighted_training_loss(
    losses: torch.Tensor,
    optimal_mask: torch.Tensor,
    handoff_ids: torch.Tensor,
    class_weights: torch.Tensor,
    handoff_weights: torch.Tensor,
) -> torch.Tensor:
    positive_count = optimal_mask.sum(dim=-1).clamp_min(1)
    sample_class_weight = (
        optimal_mask.to(class_weights.dtype) * class_weights.unsqueeze(0)
    ).sum(dim=-1) / positive_count
    sample_weight = sample_class_weight * handoff_weights[handoff_ids]
    return (losses * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


# 功能：给模型选择规则生成可保存、可解释的定义信息。
# 输入：budget_choices 为实际候选预算序列，最低值用于定义两类状态组。
# 输出：字典，注明最低预算可选/不可选两组的集合兼容率等权平均、空组处理和三个选模指标的优先顺序。
# 口径：代表标签指标仅作诊断；最佳模型与早停优先依据分组平衡的集合兼容率。
def selection_metric_definition(budget_choices: Sequence[int]) -> Dict:
    """Describe the mask-based selection rule saved with each training run."""
    return {
        "name": "minimum_budget_group_balanced_compatible_accuracy_v1",
        "minimum_budget": min(budget_choices),
        "groups": ["minimum_budget_allowed", "non_minimum_required"],
        "definition": "Mean compatible accuracy over nonempty optimal-mask groups; "
        "a prediction is correct when it belongs to the full optimal action set.",
        "empty_group_policy": "Excluded from the mean; support and accuracy are reported.",
        "priority": ["balanced_compatible_accuracy", "compatible_accuracy", "negative_loss"],
        "canonical_metrics": "Diagnostic only; not used for selection or early stopping.",
    }


# 功能：把验证指标转换成按字典序比较的模型选择分数。
# 输入：metrics 需含 balanced_compatible_accuracy、compatible_accuracy 和 loss。
# 输出：三元组 (分组平衡集合兼容率, 总体集合兼容率, -损失)；越大越好，只有前一项相等才比较后一项。
def selection_score(metrics: Mapping) -> tuple[float, float, float]:
    """Maximize group balance, then overall set compatibility, then minimize loss."""
    return (
        float(metrics["balanced_compatible_accuracy"]),
        float(metrics["compatible_accuracy"]),
        -float(metrics["loss"]),
    )


# 功能：在固定历史状态上评估 Router，区分完整集合兼容表现与代表标签分类表现。
# 输入：router 为网络，loader 提供特征/阶段/目标/掩码/来源类型，device 为设备，budget_choices 将类别索引映射为 B。
# 输出：指标字典，含未加权集合损失、总体/分组兼容率、代表标签准确率及混淆矩阵、分阶段/来源类型表现和平均预测 B。
# 分组：按最优掩码是否允许最低候选预算分成两组，再对非空组的兼容率等权平均；预测属于任一最优动作即算兼容。
# 副作用：将 router 置为 eval 模式并关闭梯度，不更新权重，也不执行 Agent 后续解题。
def evaluate(
    router: BudgetRouter,
    loader: DataLoader,
    device: torch.device,
    budget_choices: Sequence[int],
) -> Dict:
    router.eval()
    total_loss = 0.0
    total = 0
    compatible = 0
    canonical_correct = 0
    confusion = torch.zeros(len(budget_choices), len(budget_choices), dtype=torch.long)
    per_handoff = {index: [0, 0] for index in range(3)}
    per_kind: Dict[str, list[int]] = {}
    predicted_budgets = []
    minimum_class = tuple(budget_choices).index(min(budget_choices))
    group_counts = {"minimum_budget_allowed": [0, 0], "non_minimum_required": [0, 0]}

    with torch.no_grad():
        for batch in loader:
            features = batch["router_feature"].to(device)
            handoffs = batch["handoff_index"].to(device)
            targets = batch["target_class"].to(device)
            masks = batch["optimal_budget_mask"].to(device)
            logits = router(features, handoffs)
            losses = multi_positive_losses(logits, masks)
            predictions = logits.argmax(dim=-1)
            batch_compatible = masks.gather(1, predictions.unsqueeze(1)).squeeze(1)
            minimum_allowed = masks[:, minimum_class]
            for name, group in (
                ("minimum_budget_allowed", minimum_allowed),
                ("non_minimum_required", ~minimum_allowed),
            ):
                group_counts[name][0] += int((batch_compatible & group).sum().item())
                group_counts[name][1] += int(group.sum().item())

            count = predictions.numel()
            total += count
            total_loss += float(losses.sum().item())
            compatible += int(batch_compatible.sum().item())
            canonical_correct += int((predictions == targets).sum().item())
            predicted_budgets.extend(int(budget_choices[index]) for index in predictions.cpu())

            for target, prediction in zip(targets.cpu(), predictions.cpu()):
                confusion[int(target), int(prediction)] += 1
            for index in range(count):
                handoff = int(handoffs[index].item())
                per_handoff[handoff][1] += 1
                per_handoff[handoff][0] += int(batch_compatible[index].item())
                kind = str(batch["source_kind"][index])
                per_kind.setdefault(kind, [0, 0])
                per_kind[kind][1] += 1
                per_kind[kind][0] += int(batch_compatible[index].item())

    recalls = []
    per_class_recall = {}
    for class_index, budget in enumerate(budget_choices):
        support = int(confusion[class_index].sum().item())
        recall = (
            float(confusion[class_index, class_index].item() / support)
            if support
            else None
        )
        per_class_recall[str(budget)] = {"support": support, "recall": recall}
        if recall is not None:
            recalls.append(recall)

    per_budget_group = {
        name: {"support": support, "compatible_accuracy": correct / support if support else None}
        for name, (correct, support) in group_counts.items()
    }
    group_accuracies = [
        group["compatible_accuracy"] for group in per_budget_group.values() if group["support"]
    ]
    return {
        "loss": total_loss / max(total, 1),
        "compatible_accuracy": compatible / max(total, 1),
        "canonical_accuracy": canonical_correct / max(total, 1),
        "macro_recall": sum(recalls) / max(len(recalls), 1),
        "balanced_compatible_accuracy": sum(group_accuracies) / max(len(group_accuracies), 1),
        "per_budget_group": per_budget_group,
        "per_class_recall": per_class_recall,
        "per_handoff_compatible_accuracy": {
            str(key): values[0] / max(values[1], 1)
            for key, values in per_handoff.items()
        },
        "per_source_kind_compatible_accuracy": {
            key: values[0] / max(values[1], 1) for key, values in per_kind.items()
        },
        "confusion_matrix": confusion.tolist(),
        "average_predicted_budget": sum(predicted_budgets) / max(len(predicted_budgets), 1),
        "examples": total,
    }


# 功能：从标注状态构造训练集，随机初始化 Router，训练、按验证表现选模并保存结果。
# 输入：无显式参数；读取命令行数据路径、候选预算、来源划分、网络/优化器设置和设备。
# 输出：返回 None；保存 source_split.json、setup.json；实际训练还保存 best_router.pt、history.csv、metrics.json。
# 流程：按来源题隔离训练/验证，采用集合标签加权损失、AdamW 和梯度裁剪；按 selection_score 保存最佳权重并决定早停。
# 结束：重新加载最佳权重评估训练/验证状态；dry_run 只整理和检查数据，不进行参数更新。
def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    choices = parse_budget_choices(args.budget_choices)
    selection_metric = selection_metric_definition(choices)
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

    feature_dim = int(train_records[0]["router_feature"].numel())
    device = auto_device(args.device)
    router = BudgetRouter(
        hidden_dim=feature_dim,
        budget_choices=choices,
        router_hidden_dim=args.router_hidden_dim,
        handoff_embedding_dim=args.handoff_embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    class_weights, handoff_weights, balance_report = compute_balance_weights(
        train_records, len(choices), args.class_balance_power
    )
    class_weights = class_weights.to(device)
    handoff_weights = handoff_weights.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        RouterFeatureDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    train_eval_loader = DataLoader(
        RouterFeatureDataset(train_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        RouterFeatureDataset(validation_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    split_payload = {
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "train_sources": train_sources,
        "validation_sources": validation_sources,
    }
    (output_folder / "source_split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    setup_report = {
        "arguments": vars(args),
        "data": data_report,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "balance": balance_report,
        "feature_dim": feature_dim,
        "selection_metric": selection_metric,
    }
    (output_folder / "setup.json").write_text(
        json.dumps(setup_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(setup_report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run completed; no Router parameters were updated.")
        return

    checkpoint_path = output_folder / "best_router.pt"
    best_score = (-math.inf, -math.inf, -math.inf)
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        router.train()
        running_loss = 0.0
        running_examples = 0
        for batch in train_loader:
            features = batch["router_feature"].to(device)
            handoffs = batch["handoff_index"].to(device)
            masks = batch["optimal_budget_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(features, handoffs)
            raw_losses = multi_positive_losses(logits, masks)
            loss = weighted_training_loss(
                raw_losses,
                masks,
                handoffs,
                class_weights,
                handoff_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * features.shape[0]
            running_examples += features.shape[0]

        validation_metrics = evaluate(router, validation_loader, device, choices)
        row = {
            "epoch": epoch,
            "weighted_train_loss": running_loss / max(running_examples, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_compatible_accuracy": validation_metrics["compatible_accuracy"],
            "validation_canonical_accuracy": validation_metrics["canonical_accuracy"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_balanced_compatible_accuracy": validation_metrics["balanced_compatible_accuracy"],
            "selection_metric": selection_metric["name"],
            "minimum_budget": selection_metric["minimum_budget"],
        }
        for name, group in validation_metrics["per_budget_group"].items():
            row[f"validation_{name}_support"] = group["support"]
            row[f"validation_{name}_compatible_accuracy"] = group["compatible_accuracy"]
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        score = selection_score(validation_metrics)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": router.state_dict(),
                    "router_config": {
                        "hidden_dim": feature_dim,
                        "budget_choices": choices,
                        "num_handoffs": 3,
                        "router_hidden_dim": args.router_hidden_dim,
                        "handoff_embedding_dim": args.handoff_embedding_dim,
                        "dropout": args.dropout,
                    },
                    "training_view": args.view,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "selection_metric": selection_metric,
                    "selection_score": score,
                    "source_split": split_payload,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with (output_folder / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    router.load_state_dict(checkpoint["state_dict"])
    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": selection_metric,
        "selection_score": checkpoint["selection_score"],
        "train": evaluate(router, train_eval_loader, device, choices),
        "validation": evaluate(router, validation_loader, device, choices),
    }
    (output_folder / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"Saved best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
