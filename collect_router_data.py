# 文件作用：通过完整预算组合搜索生成 Router 训练标签和状态记录。
# 主要职责：加载题目与模型，枚举通信预算轨迹，记录答案、成本及特征，构建并保存Oracle标签。

"""Collect joint-Oracle trajectories and labels for training BudgetRouter.

For every source example, the script exhaustively evaluates the Cartesian
product of the handoff budgets.  With four choices and three handoffs this is
4**3 = 64 complete rollouts.  It writes both the three examples on one
canonical globally optimal path and up to 21 prefix-state examples reserved
for later training.

The script does not train the router.  It writes one resumable ``.pt`` shard per
source example plus a compact JSONL manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm

from budget_router import (
    build_primary_examples,
    build_reserve_state_examples,
    enumerate_budget_schedules,
    parse_budget_choices,
    select_canonical_schedule,
    select_global_optimal_schedules,
)
from compression_methods import HOBF
from data import (
    load_aime2024,
    load_aime2025,
    load_arc_challenge,
    load_arc_easy,
    load_gpqa_diamond,
    load_gsm8k,
    load_humanevalplus,
    load_mbppplus,
    load_medqa,
)
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from router_data_selection import read_index_file
from utils import auto_device, set_seed


HANDOFF_COUNT = 3
SHARD_SCHEMA_VERSION = 2


# 功能：按任务名称分派到相应数据集加载函数。
# 输入：task 为受支持的任务名，split 为数据划分名称，cache_dir 为可选缓存目录。
# 输出：对应加载器返回的可迭代题目字典集合；不支持的任务名会触发 ValueError。
# 注意：这里负责读取数据，不负责抽题、模型推理或生成预算标签。
def load_task(
    task: str,
    split: str,
    cache_dir: str | None = None,
) -> Iterable[Dict]:
    if task == "gsm8k":
        return load_gsm8k(split=split, cache_dir=cache_dir)
    if task == "aime2024":
        return load_aime2024(split=split, cache_dir=cache_dir)
    if task == "aime2025":
        return load_aime2025(split=split, cache_dir=cache_dir)
    if task == "gpqa":
        return load_gpqa_diamond(split=split, cache_dir=cache_dir)
    if task == "arc_easy":
        return load_arc_easy(split=split, cache_dir=cache_dir)
    if task == "arc_challenge":
        return load_arc_challenge(split=split, cache_dir=cache_dir)
    if task == "mbppplus":
        return load_mbppplus(split=split, cache_dir=cache_dir)
    if task == "humanevalplus":
        return load_humanevalplus(split=split, cache_dir=cache_dir)
    if task == "medqa":
        return load_medqa(split=split, cache_dir=cache_dir)
    raise ValueError(f"Unsupported task: {task}")


# 功能：从完整推理返回值中提取搜索和报告需要的答案、正确性与成本字段。
# 输入：result 是一次整题推理结果字典，可包含答案、KV 通信量和各阶段耗时。
# 输出：新的摘要字典，包含 correct、prediction、gold、raw_prediction、communication_overhead 和三类耗时。
# 注意：缺失 correct 时按 False 处理，缺失成本按 0.0 处理；不保留 router_records 等大型中间记录。
def summarize_result(result: Dict) -> Dict:
    return {
        "correct": bool(result.get("correct", False)),
        "prediction": result.get("prediction"),
        "gold": result.get("gold"),
        "raw_prediction": result.get("raw_prediction"),
        "communication_overhead": float(result.get("communication_overhead", 0.0)),
        "compression_time": float(result.get("compression_time", 0.0)),
        "latent_inference_time": float(result.get("latent_inference_time", 0.0)),
        "text_inference_time": float(result.get("text_inference_time", 0.0)),
    }


# 功能：按已执行的预算前缀归并决策前特征，并记录重复到达同一前缀时的特征差异。
# 输入：schedule 为本次预算轨迹，router_records 应有三条通信记录，state_records 为可原地更新的前缀到状态字典。
# 输出：无返回值（None）；将 CPU float32 特征、观察次数和最大绝对差写入 state_records。
# 流程：首次到达前缀保存特征副本；后续到达只比较形状与数值并累计差异，不用新值覆盖原特征。
# 注意：prefix 不包含当前待选 B；这里记录差异，是否警告由主函数依据阈值决定。
def collect_state_features(
    schedule: Sequence[int],
    router_records: Sequence[Mapping],
    state_records: Dict[Tuple[int, ...], Dict],
) -> None:
    """Deduplicate router features by the budgets selected before a handoff."""

    if len(router_records) != HANDOFF_COUNT:
        raise RuntimeError(
            f"Expected {HANDOFF_COUNT} router records, got {len(router_records)}"
        )

    schedule = tuple(int(value) for value in schedule)
    for handoff_index, feature_record in enumerate(router_records):
        prefix = schedule[:handoff_index]
        feature = feature_record["router_feature"].detach().to(
            dtype=torch.float32,
            device="cpu",
        )
        if prefix not in state_records:
            state_records[prefix] = {
                "handoff_index": handoff_index,
                "agent_name": feature_record.get("agent_name"),
                "agent_role": feature_record.get("agent_role"),
                "router_feature": feature.clone(),
                "feature_observations": 1,
                "feature_max_abs_diff": 0.0,
            }
            continue

        state = state_records[prefix]
        stored_feature = state["router_feature"]
        if stored_feature.shape != feature.shape:
            raise RuntimeError(
                f"Router feature shape changed for prefix {prefix}: "
                f"{tuple(stored_feature.shape)} vs {tuple(feature.shape)}"
            )
        max_abs_diff = float((stored_feature - feature).abs().max().item())
        state["feature_observations"] += 1
        state["feature_max_abs_diff"] = max(
            float(state["feature_max_abs_diff"]),
            max_abs_diff,
        )


# 功能：把一道题的 Oracle 与状态标签统计加入全局汇总计数。
# 输入：shard_payload 为逐题保存记录，后三个 Counter 分别计全局状态、主轨迹预算标签和备用状态标签状态。
# 输出：无返回值（None）；原地增加 global_status_counts、primary_label_counts、reserve_status_counts。
# 注意：主轨迹标签按状态条数计数，全局状态按题目计数，两者不是同一个统计口径。
def update_summary_counts(
    shard_payload: Mapping,
    global_status_counts: Counter,
    primary_label_counts: Counter,
    reserve_status_counts: Counter,
) -> None:
    global_status_counts[shard_payload["global_oracle"]["status"]] += 1
    for record in shard_payload.get("primary_examples", []):
        primary_label_counts[int(record["target_budget"])] += 1
    for record in shard_payload.get("reserve_state_examples", []):
        reserve_status_counts[record["label_status"]] += 1


# 功能：声明完整预算搜索程序接受的命令行参数及默认值。
# 输入：无函数参数；参数定义覆盖模型、数据集、题号文件、预算候选、推理设置、输出路径与续跑开关。
# 输出：argparse.ArgumentParser 对象；调用方还需执行 parse_args() 才会读取实际命令行。
# 注意：reference_budget 用于初始化和回退，完整搜索会为三次通信显式指定轨迹中的 B。
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Oracle BudgetRouter features and labels."
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-4B",
        help="Hugging Face model ID or an absolute/local model snapshot path.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face model and dataset cache directory.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer only from local files. Local model paths enable this automatically.",
    )
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum selected source examples to collect; -1 uses every selected index.",
    )
    parser.add_argument(
        "--sample_indices_file",
        default=None,
        help=(
            "Optional text file containing one original dataset index per line. "
            "The exhaustive 64-way search is restricted to these examples."
        ),
    )
    parser.add_argument("--output_folder", default="router_data/gsm8k_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--candidate_budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument(
        "--reference_budget",
        type=int,
        default=32,
        help=(
            "Compressor initialization/fallback budget. Joint collection "
            "overrides all three handoffs with candidate schedules."
        ),
    )
    parser.add_argument("--pca_rank", type=int, default=8)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


# 功能：执行完整组合搜索，从最终答案正确性构造全局最优轨迹及前缀状态标签。
# 输入：从命令行读取配置，并加载题目、可选题号文件和模型；--resume 时还读取已有逐题分片。
# 输出：返回 None；保存 samples/sample_*.pt、manifest.jsonl、summary.json，并打印汇总和进度。
# 流程：每题枚举三次通信的全部预算组合，按前缀收集特征，再生成主轨迹与备用状态标签。
# 注意：同题各组合重置同一随机种子；已有分片在续跑时直接计入汇总，不重新运行该题。
def main() -> None:
    args = build_parser().parse_args()
    args.method = "latent_mas"
    args.text_mas_context_length = -1
    args.latent_only = False
    args.sequential_info_only = False

    budget_choices = parse_budget_choices(args.candidate_budgets)
    if args.reference_budget not in budget_choices:
        raise ValueError(
            f"reference_budget={args.reference_budget} must be one of {budget_choices}"
        )
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be positive or -1")

    output_folder = Path(args.output_folder)
    shard_folder = output_folder / "samples"
    shard_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = output_folder / "manifest.jsonl"
    if not args.resume:
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

    schedules = enumerate_budget_schedules(budget_choices, HANDOFF_COUNT)
    global_status_counts: Counter = Counter()
    primary_label_counts: Counter = Counter()
    reserve_status_counts: Counter = Counter()
    dataset = list(load_task(args.task, args.split, cache_dir=args.cache_dir))
    if args.sample_indices_file:
        selected_indices = list(read_index_file(args.sample_indices_file))
    else:
        selected_indices = list(range(len(dataset)))

    invalid_indices = [
        index for index in selected_indices if index < 0 or index >= len(dataset)
    ]
    if invalid_indices:
        raise ValueError(
            f"Dataset indices out of range for split size {len(dataset)}: "
            f"{invalid_indices[:10]}"
        )
    if args.max_samples != -1:
        selected_indices = selected_indices[: args.max_samples]
    if not selected_indices:
        raise ValueError("No source examples selected for collection")

    progress = tqdm(total=len(selected_indices), desc="Collecting router data")

    for sample_index in selected_indices:
        item = dataset[sample_index]

        shard_path = shard_folder / f"sample_{sample_index:06d}.pt"
        if args.resume and shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu")
            if existing.get("schema_version") != SHARD_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Cannot resume legacy shard {shard_path}. Use a new output "
                    "folder so the old conditional-Oracle data is preserved."
                )
            update_summary_counts(
                existing,
                global_status_counts,
                primary_label_counts,
                reserve_status_counts,
            )
            progress.update(1)
            continue

        # ------------------------------------------------------------------
        # 旧折中方案（保留但不执行）：
        # 先运行一次三个 handoff 都使用 reference_budget（默认 B=32）的
        # baseline，再一次只改变一个 handoff。四个候选预算、三个 handoff
        # 共需 1 + 3 * (4 - 1) = 10 次完整推理。该方案得到的是“其他
        # handoff 固定为 B=32 条件下”的局部标签，不是 64 种联合组合中的
        # 全局最优标签。为了以后可能退回折中版本，原核心代码完整注释保留。
        #
        # set_seed(args.seed + sample_index * 1000)
        # baseline = method.run_item(item)
        # baseline_summary = summarize_result(baseline)
        # baseline_records = baseline.get("router_records", [])
        # handoff_records = []
        # for handoff_index in range(HANDOFF_COUNT):
        #     feature_record = baseline_records[handoff_index]
        #     budget_results = {
        #         args.reference_budget: dict(baseline_summary)
        #     }
        #     for budget in budget_choices:
        #         if budget == args.reference_budget:
        #             continue
        #         set_seed(args.seed + sample_index * 1000)
        #         candidate = method.run_item(
        #             item,
        #             budget_schedule={handoff_index: budget},
        #         )
        #         budget_results[budget] = summarize_result(candidate)
        #     target_budget, target_class, label_status = choose_oracle_budget(
        #         budget_results,
        #         budget_choices,
        #     )
        #     handoff_records.append(
        #         {
        #             "sample_id": sample_index,
        #             "task": args.task,
        #             "split": args.split,
        #             "handoff_index": handoff_index,
        #             "agent_name": feature_record["agent_name"],
        #             "agent_role": feature_record["agent_role"],
        #             "router_feature": feature_record["router_feature"].to(
        #                 torch.float32
        #             ),
        #             "budget_results": budget_results,
        #             "target_budget": target_budget,
        #             "target_class": target_class,
        #             "label_status": label_status,
        #         }
        #     )
        # ------------------------------------------------------------------

        rollout_records = []
        schedule_results: Dict[Tuple[int, ...], Dict] = {}
        state_records: Dict[Tuple[int, ...], Dict] = {}

        for schedule in schedules:
            # 所有组合使用相同随机种子，使预算组合成为预期中的唯一干预。
            set_seed(args.seed + sample_index * 1000)
            result = method.run_item(
                item,
                budget_schedule={
                    handoff_index: budget
                    for handoff_index, budget in enumerate(schedule)
                },
            )
            result_summary = summarize_result(result)
            schedule_results[schedule] = result_summary
            collect_state_features(
                schedule,
                result.get("router_records", []),
                state_records,
            )
            rollout_records.append(
                {
                    "schedule": schedule,
                    "cost": sum(schedule),
                    **result_summary,
                }
            )

        minimum_cost, optimal_schedules, global_status = (
            select_global_optimal_schedules(schedule_results)
        )
        canonical_schedule = select_canonical_schedule(optimal_schedules)
        primary_examples = build_primary_examples(
            canonical_schedule,
            state_records,
            budget_choices,
        )
        reserve_state_examples = build_reserve_state_examples(
            schedule_results,
            state_records,
            budget_choices,
            HANDOFF_COUNT,
        )

        feature_max_abs_diff = max(
            float(record["feature_max_abs_diff"])
            for record in state_records.values()
        )
        if feature_max_abs_diff > 1e-3:
            print(
                f"Warning: sample {sample_index} repeated-prefix router features "
                f"differ by up to {feature_max_abs_diff:.6g}."
            )

        shard_payload = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "sample_id": sample_index,
            "task": args.task,
            "split": args.split,
            "model_name": args.model_name,
            "cache_dir": args.cache_dir,
            "local_files_only": bool(args.local_files_only or Path(args.model_name).is_dir()),
            "question": item.get("question"),
            "gold": item.get("gold"),
            "budget_choices": budget_choices,
            "num_handoffs": HANDOFF_COUNT,
            "rollouts": rollout_records,
            "global_oracle": {
                "status": global_status,
                "minimum_cost": minimum_cost,
                "optimal_schedules": optimal_schedules,
                "canonical_schedule": canonical_schedule,
            },
            "primary_examples": primary_examples,
            "reserve_state_examples": reserve_state_examples,
            "feature_max_abs_diff": feature_max_abs_diff,
        }
        torch.save(shard_payload, shard_path)

        update_summary_counts(
            shard_payload,
            global_status_counts,
            primary_label_counts,
            reserve_status_counts,
        )

        manifest_record = {
            "sample_id": sample_index,
            "shard": str(shard_path),
            "global_status": global_status,
            "minimum_cost": minimum_cost,
            "optimal_schedule_count": len(optimal_schedules),
            "canonical_schedule": canonical_schedule,
            "primary_targets": [
                record["target_budget"] for record in primary_examples
            ],
            "reserve_valid_count": sum(
                record["target_budget"] is not None
                for record in reserve_state_examples
            ),
        }
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

        progress.update(1)

    progress.close()
    summary = {
        "task": args.task,
        "split": args.split,
        "requested_samples": len(selected_indices),
        "dataset_size": len(dataset),
        "sample_indices_file": args.sample_indices_file,
        "selected_indices": selected_indices,
        "model_name": args.model_name,
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only or Path(args.model_name).is_dir()),
        "budget_choices": list(budget_choices),
        "schedules_per_sample": len(schedules),
        "primary_label_counts": {
            str(key): value for key, value in sorted(primary_label_counts.items())
        },
        "global_status_counts": dict(global_status_counts),
        "reserve_status_counts": dict(reserve_status_counts),
    }
    summary_path = output_folder / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
