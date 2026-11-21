# 文件作用：比较原500道题上的固定预算与 Oracle 收益。
# 主要职责：核查已有结果、补跑缺失固定B32任务，汇总正确率及共同答对题的通信成本。

"""Audit, complete fixed B32, and report on the original screening questions.

Only run mode loads the model. Original screen/Oracle shards are read-only.
All imports of the inference stack are delayed until a validated plan exists.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean

import torch


# 功能：把已有实验分片加载到 CPU，供结果检查和统计使用。
# 输入：path 为可信的本地 PyTorch 文件路径；文件中可以包含张量和普通 Python 对象。
# 输出：返回文件保存的对象；本函数只读取文件，不执行模型推理。
def load(path):
    # Only load the user's trusted local experiment shards.
    return torch.load(path, map_location="cpu", weights_only=False)


# 功能：先写同目录临时文件，再替换目标文件，保存一份完整的 JSON 记录。
# 输入：path 为目标 Path，value 为可 JSON 序列化的数据；父目录需由调用者准备。
# 输出：返回 None；写入 UTF-8 JSON 文件，并通过 os.replace 替换原目标文件。
def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# 功能：检查结果是否包含布尔型正确性标记，以及有限且大于零的 KV 通信量。
# 输入：result 为单次推理结果字典，至少需要 correct 和 communication_overhead 字段。
# 输出：通过检查时原样返回 result；字段不合法时抛出 ValueError，不修改记录。
# 边界：这里只检查记录格式和数值范围，不重新判答案，也不验证实际通信过程。
def result_check(result):
    if not isinstance(result.get("correct"), bool):
        raise ValueError("Missing boolean answer correctness")
    kv = result.get("communication_overhead")
    if kv is None or not math.isfinite(float(kv)) or float(kv) <= 0:
        raise ValueError("Missing or invalid KV measurement")
    return result


# 功能：用题目文本和标准答案生成指纹，检查不同实验记录是否对应同一道题。
# 输入：payload 为包含 question、gold 字段的字典；按这两个原始值进行 JSON 编码。
# 输出：返回 SHA-256 十六进制字符串；不包含模型参数，也不做文本归一化。
def fingerprint(payload):
    text = json.dumps([payload["question"], payload["gold"]], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


# 功能：核对筛选记录与完整搜索记录，检查题目范围、64 种组合、预算和及保存的 Oracle 选择。
# 输入：screen_folder 为筛选目录，oracle_folder 为完整搜索目录，expected_count 为应有独立题数。
# 输出：返回 (ids, screen, fixed, oracle, meta)，分别为题目索引、B8 筛选信息、可复用 B32 结果、Oracle 结果和搜索来源信息。
# 口径：全部组合失败时 oracle[i] 为 None；未完整搜索的题必须在 (8,8,8) 下成功，才能用预算和 24 作为 Oracle 结果。
# 副作用：只读取已有文件；发现重复、覆盖缺失、B8 重叠结果或已保存最优方案不一致时抛出异常。
def audit(screen_folder, oracle_folder, expected_count):
    selection = json.loads((screen_folder / "selection.json").read_text(encoding="utf-8"))
    ids = [int(i) for i in selection["selected_indices"]]
    if selection["task"] != "gsm8k" or selection["split"] != "train":
        raise ValueError("Expected the original GSM8K train selection")
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {expected_count} distinct question IDs, found {len(ids)}")
    screen = {}
    for path in sorted((screen_folder / "samples").glob("sample_*.pt")):
        d = load(path)
        i = int(d["dataset_index"])
        if i in screen or i not in ids:
            raise ValueError(f"Duplicate or out-of-selection screening shard: {path}")
        if d["task"] != "gsm8k" or d["split"] != "train" or tuple(d["minimum_schedule"]) != (8, 8, 8):
            raise ValueError(f"Unexpected screening settings: {path}")
        r = dict(result_check(d["result"]))
        if r["correct"] != d["minimum_budget_correct"]:
            raise ValueError(f"Screen correctness fields disagree: {path}")
        screen[i] = {"question": d["question"], "gold": d["gold"],
                     "fingerprint": fingerprint(d), "b8": r}
    if set(screen) != set(ids):
        raise ValueError("Original selection and screening shard IDs differ")
    fixed, oracle, meta = {}, {}, {}
    grid = set(itertools.product((8, 16, 32, 64), repeat=3))
    for path in sorted((oracle_folder / "samples").glob("sample_*.pt")):
        d = load(path)
        i = int(d["sample_id"])
        if i not in screen or i in meta:
            raise ValueError(f"Duplicate or out-of-selection Oracle shard: {path}")
        if d["task"] != "gsm8k" or d["split"] != "train" or fingerprint(d) != screen[i]["fingerprint"]:
            raise ValueError(f"Question identity mismatch: {path}")
        runs = {tuple(r["schedule"]): r for r in d["rollouts"]}
        if len(d["rollouts"]) != 64 or set(runs) != grid:
            raise ValueError(f"Incomplete 64-way search: {path}")
        for key, r in runs.items():
            result_check(r)
            if r["cost"] != sum(key):
                raise ValueError(f"Invalid Oracle cost: {path}")
        old, new = screen[i]["b8"], runs[(8, 8, 8)]
        if old["correct"] != new["correct"] or old.get("prediction") != new.get("prediction"):
            raise ValueError(f"B8 screening and exhaustive answers differ at question {i}; investigate before mixing runs")
        if not math.isclose(float(old["communication_overhead"]), float(new["communication_overhead"]), rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f"B8 screening and exhaustive KV differ at question {i}")
        best = min((k for k, r in runs.items() if r["correct"]), key=lambda k: (sum(k), k), default=None)
        saved = d["global_oracle"]["canonical_schedule"]
        if best != (tuple(saved) if saved is not None else None):
            raise ValueError(f"Saved Oracle choice differs from rollouts: {path}")
        oracle[i] = dict(runs[best]) if best is not None else None
        fixed[i] = dict(runs[(32, 32, 32)])
        meta[i] = {"model_name": d["model_name"], "path": str(path)}
    if not meta:
        raise ValueError("No Oracle shards found")
    if len({m["model_name"] for m in meta.values()}) != 1:
        raise ValueError("Oracle shards contain multiple models")
    for i in ids:
        if i not in oracle:
            if not screen[i]["b8"]["correct"]:
                raise ValueError(f"Failed B8 question {i} lacks complete search; Oracle comparison would be incomplete")
            oracle[i] = dict(screen[i]["b8"], schedule=[8, 8, 8], cost=24)
    return ids, screen, fixed, oracle, meta


# 功能：提取用于比较新旧运行是否一致的一组模型与生成参数。
# 输入：args 为参数对象，需具有 model_name、prompt、latent_steps、max_new_tokens 等被列出的属性。
# 输出：返回参数名到参数值的字典；未列出的设备、目录等参数不会进入这份设置摘要。
def settings(args):
    return {k: getattr(args, k) for k in ("model_name", "prompt", "latent_steps", "max_new_tokens", "pca_rank", "sink_size", "seed", "think", "latent_space_realign")}


# 功能：读取已经补跑并保存的 B32 结果，与已有完整搜索中的 B32 结果合并，支持断点续跑。
# 输入：args 提供输出目录及当前设置；ids 为目标题目索引，screen 提供题目指纹，fixed 为已有 B32 结果。
# 输出：返回新的 B32 结果字典 merged；补跑记录统一标记 schedule=[32,32,32]、cost=96。
# 检查：索引重复或越界、指纹或设置不一致、结果字段不合法时拒绝合并；不改传入的 fixed，也不写文件。
def read_completed(args, ids, screen, fixed):
    merged = dict(fixed)
    for path in sorted((args.output_folder / "samples").glob("sample_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        i = int(d["dataset_index"])
        if i not in ids or i in merged:
            raise ValueError(f"Unexpected or duplicated B32 completion: {path}")
        if d["fingerprint"] != screen[i]["fingerprint"] or d["settings"] != settings(args):
            raise ValueError(f"Completion configuration or question mismatch: {path}")
        merged[i] = dict(result_check(d["result"]), schedule=[32, 32, 32], cost=96)
    return merged


# 功能：加载语言模型和 HOBF，对尚缺结果的题目完整执行固定 (32,32,32) 的多 Agent 解题流程。
# 输入：args 提供模型、设备、生成参数和输出目录；missing 为待补跑题目索引；screen 提供原题指纹。
# 输出：返回 None；每题向 output_folder/samples 写入 sample_XXXXXX.json，保存索引、指纹、设置和推理结果。
# 过程：先核对当前数据集题目身份，再按 seed + 题目索引 * 1000 设种子；会补充或覆盖 args 中若干推理配置属性。
def run_missing(args, missing, screen):
    from collect_router_data import load_task, summarize_result
    from compression_methods import HOBF
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device, set_seed
    from tqdm import tqdm

    dataset = list(load_task("gsm8k", "train", cache_dir=args.cache_dir))
    for i in missing:
        if i >= len(dataset) or fingerprint(dataset[i]) != screen[i]["fingerprint"]:
            raise ValueError(f"Current GSM8K question/gold differs from saved question {i}")
    args.method = "latent_mas"
    args.task = "gsm8k"
    args.split = "train"
    args.text_mas_context_length = -1
    args.latent_only = False
    args.sequential_info_only = False
    set_seed(args.seed)
    model = ModelWrapper(args.model_name, auto_device(args.device), args=args)
    method = LatentMASMethod(model, compressor=HOBF(sink_size=args.sink_size, kv_budget=32, pca_rank=args.pca_rank),
        latent_steps=args.latent_steps, judger_max_new_tokens=args.max_new_tokens,
        temperature=0.0, top_p=1.0, generate_bs=1, budget_policy="fixed", kv_budget=32,
        budget_choices=(8, 16, 32, 64), args=args)
    for i in tqdm(missing, desc="Completing fixed B32"):
        set_seed(args.seed + i * 1000)
        result = result_check(summarize_result(method.run_item(dataset[i], budget_schedule={0:32, 1:32, 2:32})))
        save_json(args.output_folder / "samples" / f"sample_{i:06d}.json",
                  {"dataset_index": i, "fingerprint": screen[i]["fingerprint"], "settings": settings(args), "result": result})


# 功能：在同一批题目上汇总固定 B8、固定 B32 与 Oracle 的正确率、成功题预算和 KV 通信量。
# 输入：args 提供输出目录和设置；ids 为完整题目范围，screen、fixed、oracle 为按题目索引组织的结果。
# 输出：返回 None；写入 comparison.md、comparison.json 并打印报告，JSON 同时保留逐题结果及配对题目索引。
# 口径：成功题均值分别在各策略答对的题上计算；KV 节省只在 B32 与 Oracle 共同答对的题上计算为 1 - Oracle均值/B32均值。
# 前提：调用者需先补齐目标题目的 B32 结果；Oracle 全失败题用 None 表示，不计为零成本成功题。
def report(args, ids, screen, fixed, oracle):
    rows = {"固定 B=8": {i: dict(screen[i]["b8"], cost=24) for i in ids},
            "固定 B=32": fixed, "Oracle": oracle}
    lines = ["# 同一批 GSM8K train 题目的固定预算与 Oracle 对比", "",
             f"同一批 {len(ids)} 道独立题目；按原始题目索引匹配。Oracle 优化预算和，非训练后 Router。", "",
             "| 策略 | 答对题数 | 答题正确率 | 答对题平均总预算 | 答对题平均 KV（MiB/题） |",
             "|---|---:|---:|---:|---:|"]
    metrics = {}
    for name, runs in rows.items():
        good = [r for r in runs.values() if r is not None and r["correct"]]
        m = {"correct": len(good), "accuracy": len(good)/len(ids),
             "mean_success_budget": mean(r["cost"] for r in good) if good else None,
             "mean_success_kv_mib": mean(r["communication_overhead"] for r in good) if good else None}
        metrics[name] = m
        num = lambda v: "—" if v is None else f"{v:.2f}"
        lines.append(f"| {name} | {len(good)}/{len(ids)} | {m['accuracy']:.2%} | {num(m['mean_success_budget'])} | {num(m['mean_success_kv_mib'])} |")
    common = [i for i in ids if fixed[i]["correct"] and oracle[i] is not None and oracle[i]["correct"]]
    paired = {"question_ids": common, "n": len(common)}
    lines += ["", "各行答对题目范围不同，不能直接用上述均值计算节省比例。", "", "## B32 与 Oracle 共同答对题上的通信对比", ""]
    if common:
        f = mean(fixed[i]["communication_overhead"] for i in common)
        o = mean(oracle[i]["communication_overhead"] for i in common)
        paired.update(fixed_kv_mib=f, oracle_kv_mib=o, kv_saving=1-o/f)
        lines += ["| 共同答对题数 | B32 平均 KV（MiB/题） | Oracle 平均 KV（MiB/题） | KV 节省 |", "|---:|---:|---:|---:|", f"| {len(common)} | {f:.2f} | {o:.2f} | {1-o/f:.2%} |"]
    else:
        lines.append("没有共同答对题，节省比例不适用。")
    lines += ["", "KV 指三次传递张量大小之和，不是实测网络流量。所有组合都失败的题没有 Oracle 成功成本。此表为训练集探索，不是测试集结果；未测 Router 新增时延。", "",
              "旧样本未保存全部生成参数，必须人工核对新运行参数与原命令一致。脚本额外核对重叠题的 B8 答案及 KV 一致性，但该核对不等于完整标签复核。"]
    text = "\n".join(lines) + "\n"
    (args.output_folder / "comparison.md").write_text(text, encoding="utf-8")
    save_json(args.output_folder / "comparison.json", {"question_ids": ids, "settings": settings(args), "metrics": metrics, "paired": paired,
        "per_question": [{"dataset_index": i, "b8": rows["固定 B=8"][i], "b32": fixed[i], "oracle": oracle[i]} for i in ids]})
    print(text)


# 功能：组织 plan、run、report 三种入口，完成已有结果检查、缺失 B32 补跑和最终报告生成。
# 输入：从命令行读取筛选/搜索/输出目录、模型设置、预期题数及 limit、confirm_settings 等选项。
# 输出：正常结束返回 None；plan 只打印覆盖情况，run 按需写运行计划和逐题补跑结果，结果齐全时生成比较报告。
# 控制：report 要求全部题目已完成；limit 只限制本次补跑数量，结果仍缺失时不生成完整报告。
# 确认：补跑前要求 confirm_settings，且模型路径需与已有搜索记录一致；该标志不能替代对旧生成参数的人工核对。
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["plan", "run", "report"])
    p.add_argument("--screen_folder", type=Path, required=True)
    p.add_argument("--oracle_folder", type=Path, required=True)
    p.add_argument("--output_folder", type=Path, required=True)
    p.add_argument("--expected_count", type=int, default=500)
    p.add_argument("--model_name", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    p.add_argument("--latent_steps", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--pca_rank", type=int, default=8)
    p.add_argument("--sink_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--think", action="store_true")
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Run only this many missing questions for a smoke test")
    p.add_argument("--confirm_settings", action="store_true", help="Original inference settings have been checked against the old command")
    args = p.parse_args()
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    ids, screen, reuse, oracle, meta = audit(args.screen_folder, args.oracle_folder, args.expected_count)
    original_model = next(iter(meta.values()))["model_name"]
    if args.model_name != original_model:
        p.error(f"Use the original model path: {original_model}")
    fixed = read_completed(args, ids, screen, reuse)
    missing = [i for i in ids if i not in fixed]
    print(json.dumps({"questions":len(ids), "reused_b32":len(reuse), "new_b32_completed":len(fixed)-len(reuse),
        "missing_b32":len(missing), "oracle_success":sum(r is not None for r in oracle.values()), "settings":settings(args)}, indent=2), flush=True)
    if args.mode == "plan":
        return
    if args.mode == "report" and missing:
        p.error(f"Report requires all questions; still missing {len(missing)}")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    if args.mode == "run" and missing:
        if not args.confirm_settings:
            p.error("Check original settings, then supply --confirm_settings; original shards lack full configuration")
        (args.output_folder / "samples").mkdir(exist_ok=True)
        save_json(args.output_folder / "run_plan.json", {"question_ids":ids, "missing_ids_at_start":missing, "settings":settings(args)})
        run_missing(args, missing if args.limit is None else missing[:args.limit], screen)
        fixed = read_completed(args, ids, screen, reuse)
        if len(fixed) != len(ids):
            print(f"Smoke test complete; {len(ids)-len(fixed)} questions remain. Rerun without --limit.", flush=True)
            return
    report(args, ids, screen, fixed, oracle)


if __name__ == "__main__":
    main()
