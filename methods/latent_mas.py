# 文件作用：组织通过潜在状态和KV缓存通信的多Agent解题流程。
# 主要职责：依次执行Agent推理，在压缩前选择固定或动态预算B，传递压缩缓存并记录答案和成本。

from typing import Dict, List, Optional, Sequence, Tuple
import time

from . import default_agents
from models import ModelWrapper
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
from compression_methods import *
from budget_router import BudgetRouter, parse_budget_choices

import torch
import argparse
import gc

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        compressor = None,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        budget_router: Optional[BudgetRouter] = None,
        budget_policy: str = "fixed",
        kv_budget: int = 32,
        budget_choices: Sequence[int] = (8, 16, 32, 64),
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        # 新增功能：保存两Agent通信时的预算选择方式。
        # 输入：budget_policy为fixed/random/router，kv_budget为固定值，budget_choices为候选表，budget_router为网络。
        # 输出：实例预算配置；动态模式还会检查压缩器支持情况，并将Router放到模型设备和评估模式。
        self.budget_policy = str(budget_policy)
        self.kv_budget = int(kv_budget)
        self.budget_choices = parse_budget_choices(budget_choices)
        self.budget_router = budget_router
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if compressor is None:
            compressor = Full()
        self.compressor = compressor

        if self.budget_policy not in {"fixed", "random", "router"}:
            raise ValueError(
                "budget_policy must be one of {'fixed', 'random', 'router'}, "
                f"got {self.budget_policy!r}"
            )
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be positive, got {self.kv_budget}")
        if self.budget_policy == "router" and self.budget_router is None:
            raise ValueError("budget_policy='router' requires a BudgetRouter")
        if self.budget_policy in {"random", "router"} and not isinstance(
            self.compressor, (HOBF, Headwise)
        ):
            raise ValueError(
                "Dynamic budget policies currently support only the 'hobf' "
                "and 'headwise' compressors."
            )
        if self.budget_router is not None:
            self.budget_router.to(self.model.device).eval()

        if self.latent_only:
            self.sequential_info_only = True

        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(
        self,
        items: List[Dict],
        *,
        budget_schedule: Optional[Dict[int, int]] = None,
    ) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        device = self.model.device
        # 新增功能：接收数据搜索时指定的预算轨迹，并核对通信阶段与预算值。
        # 输入：budget_schedule是阶段编号0/1/2到预算B的映射，可只指定部分阶段。
        # 输出：规范化后的局部字典；指定阶段优先于fixed/random/router策略，动态逐题预算要求批大小为1。
        budget_schedule = {
            int(index): int(budget)
            for index, budget in (budget_schedule or {}).items()
        }
        if any(index < 0 or index >= 3 for index in budget_schedule):
            raise ValueError(
                "budget_schedule keys must be handoff indices 0, 1, or 2, "
                f"got {sorted(budget_schedule)}"
            )
        if any(budget <= 0 for budget in budget_schedule.values()):
            raise ValueError(
                f"budget_schedule values must be positive, got {budget_schedule}"
            )
        if budget_schedule and not isinstance(self.compressor, (HOBF, Headwise)):
            raise ValueError(
                "budget_schedule currently supports only HOBF and Headwise."
            )
        if batch_size != 1 and (
            self.budget_policy in {"random", "router"} or budget_schedule
        ):
            raise ValueError(
                "Dynamic per-sample budgets currently require generate_bs=1 "
                "because compressed KV lengths must match within a batch."
            )

        # KV / mask state (assumed to be precomputed correctly by your pipeline)
        past_kv: Optional[Tuple] = None

        # External position cursor (one per sample)
        pos_cursor = torch.zeros(batch_size, device=self.model.device, dtype=torch.long)

        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        # 新增记录容器：每道题保存各次通信前特征、选中预算、预算来源及网络分数，最后放入该题结果。
        router_records: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = [""] * batch_size

        all_communication_overhead = 0
        all_compression_time = 0.0
        latent_inference_time = 0.0
        text_inference_time = 0.0
        prompt_len = [0] * batch_size
        token_usage = []
        peak_overhead = 0

        metadatas = []

        self.compressor.reset()
        # ----------------------------
        # Helpers
        # ----------------------------
        def _build_batch_messages(agent, items_: List[Dict]) -> List[List[Dict]]:
            if self.args.prompt == "sequential":
                return [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items_
                ]
            if self.args.prompt == "hierarchical":
                return [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items_
                ]
            raise ValueError(f"Unknown prompt style: {self.args.prompt}")

        def _maybe_add_think(prompts: List[str]) -> List[str]:
            if getattr(self.args, "think", False):
                return [f"{p}<think>" for p in prompts]
            return prompts

        def _encode_prompts(prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
            enc = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            tokens_batch: List[List[str]] = []
            for ids_row, mask_row in zip(ids, mask):
                active_ids = ids_row[mask_row.bool()].tolist()
                tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
            return ids, mask, tokens_batch

        def _append_trace(
                idx: int,
                agent,
                prompt: str,
                input_ids: torch.Tensor,
                input_tokens: List[str],
                output: str,
                *,
                latent_steps: Optional[int] = None,
        ) -> None:
            entry = {
                "name": agent.name,
                "role": agent.role,
                "input": prompt,
                "input_ids": input_ids,
                "input_tokens": input_tokens,
                "output": output,
            }
            if latent_steps is not None:
                entry["latent_steps"] = latent_steps
            agent_traces[idx].append(entry)

        def _score_item(task: str, final_text: str, item: Dict) -> Tuple[bool, str, str, Optional[str]]:
            """
            returns: ok, pred, gold, error_msg
            """
            if task in ["mbppplus", "humanevalplus"]:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    return False, "", gold, "python error: No python code block found"
                python_code_to_exe = pred + "\n" + gold
                ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                return ok, pred, gold, error_msg

            if task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    ok = (int(pred) == int(gold))
                    return ok, pred, gold, None
                except (ValueError, TypeError):
                    return False, pred, gold, f"Value/Type error in parsing answer. Pred: {pred}, Gold: {gold}"

            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item.get("gold", "")
            ok = (pred == gold) if (pred and gold) else False
            return ok, pred, gold, None

        # ----------------------------
        # Main loop over agents
        # ----------------------------
        handoff_idx = 0
        for agent in self.agents:
            batch_messages = _build_batch_messages(agent, items)
            prompts, _, _, _ = self.model.prepare_chat_batch(batch_messages, add_generation_prompt=True)

            if agent.role != "judger":
                wrapped_prompts = _maybe_add_think(prompts)
                wrapped_ids, wrapped_mask, wrapped_tokens_batch = _encode_prompts(wrapped_prompts)

                # wrapped_mask: (B, L), dtype could be bool / int
                if wrapped_mask.dtype == torch.bool:
                    lengths = wrapped_mask.sum(dim=1).to(torch.long)  # (B,)
                else:
                    lengths = (wrapped_mask > 0).sum(dim=1).to(torch.long)  # (B,)
                lengths_list = lengths.detach().cpu().tolist()
                for i in range(len(prompt_len)):
                    prompt_len[i] += lengths_list[i]

                # ---- latent forward ----
                start = time.time()
                past_kv, final_attention, pos_cursor, router_feature = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,  # 2D only
                    pos_cursor=pos_cursor,  # (B,)
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                latent_inference_time += time.time() - start

                # 新增功能：在当前Agent完成latent推理之后、压缩KV之前，决定本次通信的B。
                # 输入：当前router_feature、handoff_idx，以及指定轨迹/固定预算/候选表/Router。
                # 输出：selected_budget是实际预算值，不是类别编号；budget_logits仅在Router分支中有值。
                # 流程：先看指定轨迹，再依次按所选固定、随机或网络策略决策，并记录本次状态。
                budget_logits = None
                if handoff_idx in budget_schedule:
                    selected_budget = int(budget_schedule[handoff_idx])
                    budget_source = "schedule"
                elif self.budget_policy == "fixed":
                    selected_budget = self.kv_budget
                    budget_source = "fixed"
                elif self.budget_policy == "random":
                    choice_idx = int(
                        torch.randint(
                            len(self.budget_choices),
                            size=(1,),
                            device=router_feature.device,
                        ).item()
                    )
                    selected_budget = self.budget_choices[choice_idx]
                    budget_source = "random"
                else:
                    handoff_ids = torch.full(
                        (batch_size,),
                        handoff_idx,
                        device=router_feature.device,
                        dtype=torch.long,
                    )
                    budget_logits, _, selected_budgets = self.budget_router.select_budget(
                        router_feature,
                        handoff_ids,
                    )
                    selected_budget = int(selected_budgets[0].item())
                    budget_source = "router"

                logits_cpu = (
                    budget_logits.detach().to(dtype=torch.float32, device="cpu")
                    if budget_logits is not None
                    else None
                )
                features_cpu = router_feature.detach().to(dtype=torch.float32, device="cpu")
                for sample_idx in range(batch_size):
                    router_records[sample_idx].append(
                        {
                            "handoff_index": handoff_idx,
                            "agent_name": agent.name,
                            "agent_role": agent.role,
                            "router_feature": features_cpu[sample_idx].clone(),
                            "selected_budget": selected_budget,
                            "budget_source": budget_source,
                            "budget_logits": (
                                logits_cpu[sample_idx].clone()
                                if logits_cpu is not None
                                else None
                            ),
                        }
                    )

                # ---- compress ----
                # past_kv, compress_time, metadata = self.compressor.compress(
                #     past_key_values=past_kv,
                #     latent_steps=self.latent_steps,
                #     prompt_mask=wrapped_mask,
                #     all_steps_attentions=final_attention,
                # )
                # 新增接入：把当前KV、注意力及提示词掩码整理成压缩参数，并为支持动态预算的压缩器传入selected_budget。
                # 输出：压缩后的past_kv继续传给下一Agent；选择预算本身不替代压缩算法。
                compress_kwargs = dict(
                    past_key_values=past_kv,
                    latent_steps=self.latent_steps,
                    prompt_mask=wrapped_mask,
                    all_steps_attentions=final_attention,
                )
                if isinstance(self.compressor, (HOBF, Headwise)):
                    compress_kwargs["kv_budget"] = selected_budget
                compress_result = self.compressor.compress(**compress_kwargs)

                if len(compress_result) == 3:
                    past_kv, compress_time, metadata = compress_result
                elif len(compress_result) == 2:
                    past_kv, compress_time = compress_result
                    metadata = None
                else:
                    raise ValueError(f"Unexpected compressor return length: {len(compress_result)}")
                all_compression_time += compress_time
                metadatas.append(metadata)
                kv_size = self.kv_size_mb(past_kv)
                all_communication_overhead += kv_size
                peak_overhead = max(peak_overhead, kv_size)

                # ---- tracing ----
                for i in range(batch_size):
                    m = wrapped_mask[i].bool()
                    trimmed_ids = wrapped_ids[i][m].to("cpu").tolist()
                    _append_trace(
                        idx=i,
                        agent=agent,
                        prompt=wrapped_prompts[i],
                        input_ids=trimmed_ids,
                        input_tokens=wrapped_tokens_batch[i],
                        output="",
                        latent_steps=self.latent_steps,
                    )

                # 新增阶段推进：一次非最终Agent的通信完成后递增编号，使下一次Router使用对应阶段嵌入。
                handoff_idx += 1

            else:
                # Judger decoding stage
                past_for_decoding = past_kv if self.latent_steps > 0 else None
                judger_prompts = _maybe_add_think(prompts)
                judger_ids, judger_mask, judger_tokens_batch = _encode_prompts(judger_prompts)

                start = time.time()
                generated_batch, past_kv, pos_cursor, token_usage = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    pos_cursor=pos_cursor,  # (B,)
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )

                text_inference_time += time.time() - start

                # If you need to continue after judger, uncomment the next line:
                # past_pos = decode_start_pos  # plus generated lengths if you track them
                for i in range(batch_size):
                    final_text = generated_batch[i].strip()
                    final_texts[i] = final_text

                    m = judger_mask[i].bool()
                    trimmed_ids = judger_ids[i][m].to("cpu").tolist()
                    _append_trace(
                        idx=i,
                        agent=agent,
                        prompt=judger_prompts[i],
                        input_ids=trimmed_ids,
                        input_tokens=judger_tokens_batch[i],
                        output=final_text,
                    )

                    # print(final_text)
                    # print("=" * 40)

        # ----------------------------
        # Package results
        # ----------------------------

        obf_metrics = {}
        if metadata is not None:
            for round_idx, meta in enumerate(metadatas):
                obf_metrics[f"r_perp_{round_idx}"] = meta["r_perp"][i] if meta is not None else None
                obf_metrics[f"recovery_R_{round_idx}"] = meta["recovery_R"][i] if meta is not None else None
                obf_metrics[f"recovery_cos_{round_idx}"] = meta["recovery_cos"][i] if meta is not None else None
                obf_metrics[f"pca_evr_{round_idx}"] = meta["pca_evr"][i] if meta is not None else None
                obf_metrics[f"ad_over_as_{round_idx}"] = meta["ad_over_as"][i] if meta is not None else None
                obf_metrics[f"inj_norm_{round_idx}"] = meta["inj_norm"][i] if meta is not None else None

        results: List[Dict] = []
        for i, item in enumerate(items):
            ok, pred, gold, error_msg = _score_item(self.task, final_texts[i], item)
            if error_msg:
                print(error_msg)

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_texts[i],
                    "agents": agent_traces[i],
                    "router_records": router_records[i],
                    "correct": ok,
                    "communication_overhead": all_communication_overhead / batch_size,
                    "compression_time": all_compression_time / batch_size,
                    "latent_inference_time": latent_inference_time / batch_size,
                    "text_inference_time": text_inference_time / batch_size,
                    "prompt_len": prompt_len[i],
                    "token_usage": token_usage[i],
                    "peak_overhead": peak_overhead,

                    # OBF metric
                    **obf_metrics,

                }
            )

        # Cleanup
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def run_item(
        self,
        item: Dict,
        *,
        budget_schedule: Optional[Dict[int, int]] = None,
    ) -> Dict:
        return self.run_batch([item], budget_schedule=budget_schedule)[0]

    @staticmethod
    def kv_size_mb(past_kv) -> float:
        if past_kv is None:
            return 0.0

        total_bytes = 0

        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
            return total_bytes / (1024 ** 2)

        # legacy
        for layer in past_kv:
            for t in layer:
                if torch.is_tensor(t):
                    total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 ** 2)
