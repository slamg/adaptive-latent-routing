# 文件作用：以三次最小预算快速筛选题目并准备完整搜索名单。
# 主要职责：为答对题生成最小预算主轨迹标签，再选择失败题和成功抽查题供后续完整搜索。

"""Screen random source questions with the all-minimum HOBF schedule.

The all-minimum schedule is a useful exact shortcut: when it is correct, it
is necessarily the unique minimum-cost global trajectory.  When it fails,
the source question is a high-value candidate for the exhaustive 64-way
Oracle collection performed by ``collect_router_data.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from budget_router import parse_budget_choices
from collect_router_data import HANDOFF_COUNT, load_task, summarize_result
from compression_methods import HOBF
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from router_data_selection import (
    choose_full_search_indices,
    sample_dataset_indices,
    write_index_file,
)
from utils import auto_device, set_seed


SCREEN_SCHEMA_VERSION = 1


# 功能：声明最小预算初筛程序的命令行参数及默认值。
# 输入：无函数参数；定义模型与数据、抽样种子、推理配置、成功/失败搜索配额及续跑选项。
# 输出：尚未解析命令行的 ArgumentParser，供 main 调用 parse_args() 使用。
# 注意：hard_full_count 和 easy_full_count 分别控制两类完整搜索题目的选择数量，-1 表示该类全部。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly screen Router source questions with (B_min,B_min,B_min)."
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--task",
        default="gsm8k",
        choices=[
            "gsm8k",
            "aime2024",
            "aime2025",
            "gpqa",
            "arc_easy",
            "arc_challenge",
            "mbppplus",
            "humanevalplus",
            "medqa",
        ],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output_folder", default="router_data/gsm8k_screen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--candidate_budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--reference_budget", type=int, default=32)
    parser.add_argument("--pca_rank", type=int, default=8)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=20260928,
        help="Seed used only for random source-question sampling.",
    )
    parser.add_argument(
        "--hard_full_count",
        type=int,
        default=30,
        help="Maximum failed-minimum questions proposed for exhaustive search; -1 keeps all.",
    )
    parser.add_argument(
        "--easy_full_count",
        type=int,
        default=20,
        help="Correct-minimum audit questions proposed for exhaustive search; -1 keeps all.",
    )
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


# 功能：为全最小预算轨迹成功的题目构造三条主轨迹监督样本。
# 输入：result 含三条 router_records；budget_choices 给出候选预算及其类别顺序。
# 输出：样本字典列表，保存阶段、Agent 信息、CPU float32 特征，以及最小预算对应的预算值和类别标签。
# 前提：调用方须已确认全最小预算轨迹答对；此函数不复查答案，只检查记录数是否为三条。
def make_primary_examples(result: dict, budget_choices: tuple[int, ...]) -> list[dict]:
    """Create exact primary labels when the absolute-minimum schedule succeeds."""

    minimum_budget = min(budget_choices)
    target_class = budget_choices.index(minimum_budget)
    examples = []
    for record in result.get("router_records", []):
        examples.append(
            {
                "handoff_index": int(record["handoff_index"]),
                "agent_name": record.get("agent_name"),
                "agent_role": record.get("agent_role"),
                "router_feature": record["router_feature"].detach().to(
                    dtype=torch.float32, device="cpu"
                ),
                "target_budget": minimum_budget,
                "target_class": target_class,
                "label_status": "minimum_budget_certified",
            }
        )
    if len(examples) != HANDOFF_COUNT:
        raise RuntimeError(
            f"Expected {HANDOFF_COUNT} router records, got {len(examples)}"
        )
    return examples


# 功能：用三次最小预算筛题，生成成功题标签，并为后续完整搜索准备题号名单。
# 输入：命令行指定数据、模型、抽样和推理配置；续跑时读取 selection.json 与已有逐题分片。
# 输出：返回 None；保存筛选选择、逐题 .pt、manifest.jsonl、各类题号文件和 summary.json。
# 流程：抽题或恢复题号 → 运行全最小预算轨迹 → 为答对题生成主轨迹标签 → 分组选出完整搜索题。
# 注意：本函数只写完整搜索名单，不执行 64 种组合；失败题是否全选由 hard_full_count 控制。
def main() -> None:
    args = build_parser().parse_args()
    args.method = "latent_mas"
    args.text_mas_context_length = -1
    args.latent_only = False
    args.sequential_info_only = False

    budget_choices = parse_budget_choices(args.candidate_budgets)
    minimum_budget = min(budget_choices)
    if args.reference_budget not in budget_choices:
        raise ValueError(
            f"reference_budget={args.reference_budget} must be one of {budget_choices}"
        )
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be positive or -1")

    dataset = list(load_task(args.task, args.split, cache_dir=args.cache_dir))
    output_folder = Path(args.output_folder)
    shard_folder = output_folder / "samples"
    shard_folder.mkdir(parents=True, exist_ok=True)
    selection_path = output_folder / "selection.json"
    manifest_path = output_folder / "manifest.jsonl"

    if args.resume and selection_path.exists():
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_payload.get("schema_version") != SCREEN_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported screening selection: {selection_path}")
        if selection_payload.get("task") != args.task or selection_payload.get("split") != args.split:
            raise RuntimeError("Resume task/split does not match the existing selection")
        selected_indices = tuple(int(x) for x in selection_payload["selected_indices"])
    else:
        selected_indices = sample_dataset_indices(
            len(dataset), args.max_samples, args.sample_seed
        )
        selection_payload = {
            "schema_version": SCREEN_SCHEMA_VERSION,
            "task": args.task,
            "split": args.split,
            "dataset_size": len(dataset),
            "sample_seed": args.sample_seed,
            "selected_indices": selected_indices,
        }
        selection_path.write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest_path.write_text("", encoding="utf-8")

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    compressor = HOBF(
        sink_size=args.sink_size,
        kv_budget=args.reference_budget,
        pca_rank=args.pca_rank,
    )
    method = LatentMASMethod(
        model,
        compressor=compressor,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        generate_bs=1,
        budget_policy="fixed",
        kv_budget=args.reference_budget,
        budget_choices=budget_choices,
        args=args,
    )

    outcomes: dict[int, bool] = {}
    progress = tqdm(total=len(selected_indices), desc="Screening minimum budget")
    minimum_schedule = {index: minimum_budget for index in range(HANDOFF_COUNT)}

    for dataset_index in selected_indices:
        shard_path = shard_folder / f"sample_{dataset_index:06d}.pt"
        if args.resume and shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu")
            if existing.get("schema_version") != SCREEN_SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported screening shard: {shard_path}")
            outcomes[dataset_index] = bool(existing["minimum_budget_correct"])
            progress.update(1)
            continue

        set_seed(args.seed + dataset_index * 1000)
        result = method.run_item(
            dataset[dataset_index], budget_schedule=minimum_schedule
        )
        result_summary = summarize_result(result)
        correct = bool(result_summary["correct"])
        primary_examples = (
            make_primary_examples(result, budget_choices) if correct else []
        )
        payload = {
            "schema_version": SCREEN_SCHEMA_VERSION,
            "dataset_index": dataset_index,
            "task": args.task,
            "split": args.split,
            "question": dataset[dataset_index].get("question"),
            "gold": dataset[dataset_index].get("gold"),
            "budget_choices": budget_choices,
            "minimum_schedule": tuple(minimum_schedule.values()),
            "minimum_budget_correct": correct,
            "requires_exhaustive_search": not correct,
            "result": result_summary,
            "primary_examples": primary_examples,
        }
        torch.save(payload, shard_path)
        outcomes[dataset_index] = correct
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "dataset_index": dataset_index,
                        "shard": str(shard_path),
                        "minimum_budget_correct": correct,
                        "primary_targets": [
                            x["target_budget"] for x in primary_examples
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        progress.update(1)

    progress.close()
    correct_indices = tuple(index for index in selected_indices if outcomes[index])
    incorrect_indices = tuple(index for index in selected_indices if not outcomes[index])
    hard_indices, easy_indices, full_search_indices = choose_full_search_indices(
        selected_indices,
        outcomes,
        hard_count=args.hard_full_count,
        easy_count=args.easy_full_count,
        seed=args.sample_seed + 1,
    )

    write_index_file(output_folder / "minimum_budget_correct_indices.txt", correct_indices)
    write_index_file(output_folder / "minimum_budget_incorrect_indices.txt", incorrect_indices)
    write_index_file(output_folder / "hard_full_indices.txt", hard_indices)
    write_index_file(output_folder / "easy_full_indices.txt", easy_indices)
    write_index_file(output_folder / "full_search_indices.txt", full_search_indices)

    summary = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "task": args.task,
        "split": args.split,
        "dataset_size": len(dataset),
        "screened_samples": len(selected_indices),
        "minimum_budget": minimum_budget,
        "minimum_budget_correct": len(correct_indices),
        "minimum_budget_incorrect": len(incorrect_indices),
        "minimum_budget_accuracy": len(correct_indices) / len(selected_indices),
        "hard_selected_for_full_search": len(hard_indices),
        "easy_selected_for_full_search": len(easy_indices),
        "full_search_total": len(full_search_indices),
        "full_search_indices_file": str(output_folder / "full_search_indices.txt"),
    }
    (output_folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
