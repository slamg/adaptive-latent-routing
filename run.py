# 文件作用：作为通用实验入口组织数据加载、方法配置和题目评估。
# 主要职责：解析参数、初始化模型与压缩/预算策略，分批运行题目并保存答案、轨迹和指标。

import time

import os
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
from typing import Dict, List, Tuple

try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from compression_methods import *
from budget_router import BudgetRouter, parse_budget_choices
from utils import auto_device, set_seed

METRIC_NAMES = [
    "r_perp",
    "recovery_R",
    "recovery_cos",
    "pca_evr",
    "ad_over_as",
    "inj_norm",
]

WANDB_FILE_SAVED = False


def init_metric_store(metric_names=None):
    metric_names = metric_names or METRIC_NAMES
    return {
        "sample_ids": [],
        "correct": [],
        **{
            f"{name}_{i}": []
            for name in metric_names
            for i in range(3)
        },
    }


def append_metrics(metric_store, res, problem_idx, metric_names=None):
    metric_names = metric_names or METRIC_NAMES

    metric_store["sample_ids"].append(problem_idx)
    metric_store["correct"].append(int(bool(res.get("correct"))))

    for name in metric_names:
        for i in range(3):
            key = f"{name}_{i}"
            x = res.get(key)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            elif x is not None:
                x = np.asarray(x)
            metric_store[key].append(x)


def save_metric_store(metric_store, output_path, use_wandb=False):
    save_dict = {
        "sample_ids": np.array(metric_store["sample_ids"], dtype=np.int64),
        "correct": np.array(metric_store["correct"], dtype=np.int64),
    }

    for name in METRIC_NAMES:
        for i in range(3):
            key = f"{name}_{i}"
            save_dict[key] = np.stack(metric_store[key], axis=0)  # (N, L, H)

    np.savez_compressed(output_path, **save_dict)

    if use_wandb:
        wandb.save(output_path, policy="live")


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def process_batch(
        method,
        batch,
        processed,
        preds,
        progress,
        max_samples,
        args,
        metric_store,
        output_file,
):
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds

    current_batch = batch[:remaining]
    results = method.run_batch(current_batch)

    if len(results) > remaining:
        results = results[:remaining]

    batch_start = processed

    output_folder = getattr(args, "output_folder", "results")
    os.makedirs(output_folder, exist_ok=True)

    use_wandb = getattr(args, "use_wandb", False)
    verbose = getattr(args, "verbose", False)

    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1

        # collect metric into global store
        append_metrics(metric_store, res, problem_idx)

        agents = res.get("agents", [])
        agent_traces = [
            {
                "name": a.get("name", "Agent"),
                "role": a.get("role", ""),
                "input": a.get("input", "").rstrip(),
                "output": a.get("output", "").rstrip(),
                "latent_steps": a.get("latent_steps", None),
            }
            for a in agents
        ]
        # 新增输出整理：从逐题router_records提取通信阶段、预算来源、所选B等便于阅读的轨迹字段。
        # 输入：方法返回的单题res；输出：写入日志的router_trace列表，不重新调用网络。
        router_trace = [
            {
                "handoff_index": record.get("handoff_index"),
                "agent_name": record.get("agent_name"),
                "agent_role": record.get("agent_role"),
                "selected_budget": record.get("selected_budget"),
                "budget_source": record.get("budget_source"),
                "budget_logits": (
                    record["budget_logits"].tolist()
                    if isinstance(record.get("budget_logits"), torch.Tensor)
                    else record.get("budget_logits")
                ),
            }
            for record in res.get("router_records", [])
        ]

        log_record = {
            "id": problem_idx,
            "question": res.get("question", "").strip(),
            "prediction": res.get("prediction"),
            "gold": res.get("gold"),
            "correct": res.get("correct"),
            "agent_traces": agent_traces,
            "router_trace": router_trace,
        }

        if verbose:
            print(f"\n==================== Problem #{problem_idx} ====================")
            print(f"Question: {log_record['question']}")
            for trace in agent_traces:
                print(f"----- Agent: {trace['name']} ({trace['role']}) -----")
                print("[To Tokenize]")
                print(trace["input"])
                if trace["latent_steps"] is not None:
                    print("[Latent Steps]")
                    print(trace["latent_steps"])
                print("[Output]")
                print(trace["output"])
                print("----------------------------------------------")
            print(
                f"Result: Pred={log_record['prediction']} | "
                f"Gold={log_record['gold']} | OK={log_record['correct']}"
            )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))

    return processed, preds


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for this LatentMAS research fork."
    )

    # Experiment selection
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas"],
        required=True,
        help="Experiment method. In this fork, only 'latent_mas' is maintained.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model ID or an absolute/local model snapshot path.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face model and dataset cache directory.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer only from local files. Local model paths enable this automatically.",
    )
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Number of samples to evaluate (-1 uses the full split).")
    parser.add_argument("--task",
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus",
                                 'humanevalplus', 'medqa'], default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential",
                        help="Prompt style for MAS methods.")

    # Runtime and generation
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device string passed to torch (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split when the loader supports custom splits.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max generated tokens for baseline/text_mas and the latent_mas judger.")
    parser.add_argument("--latent_steps", type=int, default=10,
                        help="Number of latent rollout steps for each non-judger agent in latent_mas.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for text decoding.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling threshold for text decoding.")
    parser.add_argument("--generate_bs", type=int, default=4, help="Batch size for method.run_batch().")
    parser.add_argument("--text_mas_context_length", type=int, default=-1,
                        help="TextMAS context truncation length in prompts.py (-1 usually means no truncation).")
    parser.add_argument("--think", action="store_true", help="Append '<think>' to agent prompts in latent_mas.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print per-sample agent traces to stdout.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.", default=False)

    # Output and compression
    parser.add_argument("--output_folder", type=str, default="results", help="Path to save logs")
    parser.add_argument("--compressor", type=str, default='full',
                        help="KV compressor for latent_mas (e.g., full, gonly, headwise, layerwise, shadow, lobf, hobf).")
    # 新增命令行接口：固定B、预算策略、候选预算及Router权重路径由用户配置。
    # 解析结果传给LatentMASMethod；这些参数决定每次通信如何选择压缩预算。
    parser.add_argument("--kv_budget", type=int, default=32,
                        help="Fixed prompt-KV budget and fallback budget for relay compression.")
    parser.add_argument(
        "--budget_policy",
        choices=["fixed", "random", "router"],
        default="fixed",
        help="How to select the prompt-KV budget at each handoff.",
    )
    parser.add_argument(
        "--budget_choices",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Candidate budgets predicted by BudgetRouter.",
    )
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--router_checkpoint",
        type=str,
        default=None,
        help="Optional trained BudgetRouter checkpoint. Without it, router weights are randomly initialized.",
    )

    args = parser.parse_args()

    # clean output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for name in os.listdir(args.output_folder):
        path = os.path.join(args.output_folder, name)
        os.remove(path)

    if args.method != "latent_mas":
        raise NotImplementedError(
            f"Method '{args.method}' is not maintained in this fork. "
            "Use '--method latent_mas'."
        )

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    budget_choices = parse_budget_choices(args.budget_choices)

    # 新增功能：在router策略下构建预算网络，并可从checkpoint恢复结构配置和权重。
    # 输入：候选预算、基础模型hidden维度，以及可选的router_checkpoint。
    # 输出：评估模式的BudgetRouter对象；其他策略保持None，实际选择B仍由LatentMASMethod执行。
    # 注意：未提供checkpoint时使用可复现的随机初始化，不代表已经训练好的Router。
    budget_router = None
    if args.budget_policy == "router":
        # Reset the seed so random router initialization is reproducible and
        # does not depend on implementation details inside model loading.
        set_seed(args.seed)
        checkpoint = None
        router_config = {}
        if args.router_checkpoint:
            checkpoint = torch.load(args.router_checkpoint, map_location=device)
            if isinstance(checkpoint, dict):
                router_config = checkpoint.get("router_config", {})
            checkpoint_choices = tuple(
                int(value)
                for value in router_config.get("budget_choices", budget_choices)
            )
            if checkpoint_choices != budget_choices:
                raise ValueError(
                    "Router checkpoint budget choices do not match --budget_choices: "
                    f"{checkpoint_choices} != {budget_choices}"
                )
            checkpoint_hidden_dim = int(
                router_config.get("hidden_dim", model.model.config.hidden_size)
            )
            if checkpoint_hidden_dim != int(model.model.config.hidden_size):
                raise ValueError(
                    "Router checkpoint hidden dimension does not match the model: "
                    f"{checkpoint_hidden_dim} != {model.model.config.hidden_size}"
                )
        budget_router = BudgetRouter(
            hidden_dim=int(model.model.config.hidden_size),
            budget_choices=budget_choices,
            num_handoffs=int(router_config.get("num_handoffs", 3)),
            router_hidden_dim=int(
                router_config.get("router_hidden_dim", args.router_hidden_dim)
            ),
            handoff_embedding_dim=int(
                router_config.get("handoff_embedding_dim", 32)
            ),
            dropout=float(router_config.get("dropout", 0.1)),
        ).to(device)
        if args.router_checkpoint:
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            budget_router.load_state_dict(state_dict)
            print(f"Loaded BudgetRouter checkpoint: {args.router_checkpoint}")
        else:
            print("BudgetRouter is using deterministic random initialization (no checkpoint supplied).")
        budget_router.eval()
        set_seed(args.seed)

    model_name = os.path.basename(os.path.normpath(args.model_name))
    name = f"{args.task}_{model_name}_{args.compressor}_{args.latent_steps}"
    # setup wandb symlink files
    output_file = os.path.join(args.output_folder, "raw_output.txt")
    obf_output_file = os.path.join(args.output_folder, "obf_metric.npz")

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or run without --use_wandb.")
        wandb.init(
            project=f"LatentMAS_{args.seed}",
            name=name,
            config=vars(args),
            reinit=True,
        )

        with open(output_file, "a", encoding="utf-8") as f:
            pass
        if args.use_wandb:
            wandb.save(output_file, policy="live")

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.compressor == "full":
        compressor = Full()
    elif args.compressor == "gonly":
        compressor = GenerationOnly(kv_budget=args.kv_budget)
    elif args.compressor == "headwise":
        compressor = Headwise(kv_budget=args.kv_budget)
    elif args.compressor == "layerwise":
        compressor = Layerwise(kv_budget=args.kv_budget)
    elif args.compressor == "lobf":
        compressor = LOBF(kv_budget=args.kv_budget)
    elif args.compressor == "lobf_metric":
        compressor = LOBFMetric()
    elif args.compressor == "lobf_navie":
        compressor = LOBFNaive()
    elif args.compressor == "lobf_no_proj":
        compressor = LOBFNoProj()
    elif args.compressor == "lobf_no_scale":
        compressor = LOBFNoScale()
    elif args.compressor == "lobf_max_p":
        compressor = LOBFMaxP()
    elif args.compressor == "hobf":
        compressor = HOBF(kv_budget=args.kv_budget)
    else:
        raise ValueError(f"Unknown compressor {args.compressor}")

    method = LatentMASMethod(
        model,
        compressor=compressor,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        **common_kwargs,
        generate_bs=args.generate_bs,
        args=args,
        # 新增传参：将预算网络和后续预算配置交给多Agent方法，供其在每次KV压缩前选择B。
        budget_router=budget_router,
        budget_policy=args.budget_policy,
        kv_budget=args.kv_budget,
        budget_choices=budget_choices,
    )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split, cache_dir=args.cache_dir)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train", cache_dir=args.cache_dir)
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train', cache_dir=args.cache_dir)
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test', cache_dir=args.cache_dir)
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test', cache_dir=args.cache_dir)
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test', cache_dir=args.cache_dir)
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test', cache_dir=args.cache_dir)
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test', cache_dir=args.cache_dir)
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test', cache_dir=args.cache_dir)
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)

    metric_store = init_metric_store()

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break

        batch.append(item)

        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method=method,
                batch=batch,
                processed=processed,
                preds=preds,
                progress=progress,
                max_samples=args.max_samples,
                args=args,
                metric_store=metric_store,
                output_file=output_file,
            )
            batch = []

            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method=method,
            batch=batch,
            processed=processed,
            preds=preds,
            progress=progress,
            max_samples=args.max_samples,
            args=args,
            metric_store=metric_store,
            output_file=output_file,
        )

    save_metric_store(metric_store, obf_output_file, args.use_wandb)
    progress.close()

    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    sum_lat_inf_time = sum(item.get("latent_inference_time", 0) for item in preds)
    sum_txt_inf_time = sum(item.get("text_inference_time", 0) for item in preds)
    sum_comp_time = sum(item.get("compression_time", 0) for item in preds)
    sum_comm_bits = sum(item.get("communication_overhead", 0) for item in preds)
    sum_prompt = sum(item.get("prompt_len", 0) for item in preds)
    sum_token = sum(item.get("token_usage", 0) for item in preds)
    sum_peak = sum(item.get("peak_overhead", 0) for item in preds)
    # 新增统计：展开已经完成题目的通信预算记录，用于后续计算平均所选预算。
    # 输入：逐题结果中的router_records；输出：实际预算数值列表，不是实际通信字节数。
    selected_budgets = [
        int(record["selected_budget"])
        for item in preds
        for record in item.get("router_records", [])
    ]
    metrics = {
        "method": args.method,
        "compressor": args.compressor,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4),

        "avg_latent_inference_time_s": round(sum_lat_inf_time / args.max_samples, 4),
        "avg_text_inference_time_s": round(sum_txt_inf_time / args.max_samples, 4),
        "avg_compression_time_s": round(sum_comp_time / args.max_samples, 4),
        "avg_communication_MB": round(sum_comm_bits / args.max_samples, 4),
        "prompt_len": round(sum_prompt / args.max_samples, 4),
        "token_usage": round(sum_token / args.max_samples, 4),
        "peak_overhead": round(sum_peak / args.max_samples, 4),
        "avg_selected_budget": (
            round(sum(selected_budgets) / len(selected_budgets), 4)
            if selected_budgets
            else None
        ),
    }
    print(json.dumps(metrics, ensure_ascii=False))

    if args.use_wandb and wandb is not None:
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
