# 文件作用：定义历史状态与在线推理共用的特征计算和采集工具。
# 主要职责：汇总最后一步及最近五步hidden，计算hidden和注意力统计，并支持在目标状态处采集。

"""Read-only feature capture for stage-2 experiments; no backbone updates."""
import math
import torch


class StopAtState(Exception):
    pass


# 功能：把一次 latent 推理的隐藏状态序列整理为三种可供后续使用的特征。
# 输入：states 是逐步的 [1, D] 隐藏向量列表；要求至少 2 步且批大小为 1。
# 输出：字典含 last_hidden[D]、mean_last5[D] 和 statistics[7]；最近不足 5 步时使用已有步骤。
# 统计含义：隐藏向量范数的均值/标准差/末值、相邻步差值范数的均值/末值、相邻步余弦相似度的均值/末值。
def hidden_features(states):
    h = torch.stack(states).float()  # steps, batch=1, hidden
    if len(h) < 2 or h.shape[1] != 1:
        raise ValueError("Capture requires at least 2 latent steps and batch size 1")
    norms = h[:, 0].norm(dim=1)
    changes = (h[1:, 0] - h[:-1, 0]).norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(h[1:, 0], h[:-1, 0], dim=1)
    stats = torch.tensor([norms.mean(), norms.std(unbiased=False), norms[-1],
                          changes.mean(), changes[-1], cosine.mean(), cosine[-1]])
    return {"last_hidden": h[-1, 0], "mean_last5": h[-5:, 0].mean(0), "statistics": stats}


# 功能：从最后一层提示词注意力统计信息集中程度，形成 11 维输入。
# 输入：scores 为 [1, KV 头数, 提示词长度] 的聚合得分；mask 为 [1, 提示词长度] 的有效位置标记。
# 输出：CPU 浮点向量 [11]，含各头归一化熵的均值/标准差、有效长度，以及 B=8/16/32/64 下 top-B 概率质量的均值/最小值。
# 流程：先去除填充位置并逐头归一化；候选 B 超过提示词长度时取实际可用长度。
def attention_features(scores, mask):
    # Last-layer scores per KV head, aggregated over latent steps, prompt only.
    s = scores[0].float().cpu().clamp_min(0) * mask[0].float().cpu().unsqueeze(0)
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-12)
    n = max(int(mask.sum()), 1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / (math.log(n) if n>1 else 1.0)
    sorted_p = p.sort(dim=-1, descending=True).values
    values = [float(entropy.mean()), float(entropy.std(unbiased=False)), float(n)]
    for b in (8,16,32,64):
        mass = sorted_p[:, :b].sum(-1)
        values += [float(mass.mean()), float(mass.min())]
    return torch.tensor(values)


# 功能：为模型和压缩器安装临时采集包装，在指定通信节点压缩前截取状态特征。
# 输入：model 提供 generate_latent_batch 和底层模型；compressor 提供压缩及注意力聚合接口。
# 输出：可变 state 字典，调用方设置 target、读取 captured；同时替换两个对象的方法，达到目标时通过 StopAtState 结束重放。
# 注意：这里只采集特征并核对旧 hidden，不决定预算最优性，也不重新生成 Oracle 标签。
def install_capture(model, compressor):
    """Wrap this diagnostic instance only. Return mutable capture state."""
    state = {"captured": [], "target": 2, "pending": None}
    original_generate = model.generate_latent_batch
    original_compress = compressor.compress

    # 功能：包裹 latent 生成过程，保留每步 hidden 并准备当前压缩前的隐藏特征。
    # 输入：args、kwargs 原样传给原生成函数；kwargs 中 latent_steps 用来剔除预填充等非 latent 步输出。
    # 输出：原生成函数的结果原样返回；额外把特征写入外层 state 的 pending，退出时移除 forward hook。
    # 校验：采到的最后一步 hidden 必须与原返回元组第 4 项一致，绝对容差为 1e-6。
    def generate(*args, **kwargs):
        hidden = []
        # 功能：在底层模型每次前向后记录最后一层、最后一个位置的 hidden。
        # 输入：module、inputs、output 是 PyTorch forward hook 参数；实际读取 output.hidden_states。
        # 输出：返回 None，不替换模型输出；将 [批大小, D] 的 detached CPU 浮点副本追加到外层 hidden 列表。
        def hook(module, inputs, output):
            if output.hidden_states is not None:
                hidden.append(output.hidden_states[-1][:, -1, :].detach().float().cpu().clone())
        handle = model.model.register_forward_hook(hook)
        try:
            result = original_generate(*args, **kwargs)
        finally:
            handle.remove()
        steps = kwargs["latent_steps"]
        features = hidden_features(hidden[-steps:])
        if not torch.allclose(features["last_hidden"], result[3][0].float().cpu(), atol=1e-6, rtol=0):
            raise ValueError("Hidden capture does not match original router feature")
        state["pending"] = features
        return result

    # 功能：在真正压缩前组合隐藏统计与注意力统计，并按通信节点控制重放停止位置。
    # 输入：kwargs 含 past_key_values、prompt_mask、latent_steps 和 all_steps_attentions；隐藏特征来自 state.pending。
    # 输出：未到目标时原样返回原压缩器结果；到达 target 时先保存特征再抛出 StopAtState，不执行该节点压缩。
    # 副作用：把最后一步 hidden、最近 5 步均值和 18 维统计特征追加到 state.captured，并检查特征均为有限值。
    def compress(**kwargs):
        f = state["pending"]
        past = kwargs["past_key_values"]
        legacy = compressor._as_legacy_tuple(past)
        total = int(legacy[-1][0].shape[-2])
        prompt = kwargs["prompt_mask"]
        history = total - int(prompt.shape[1]) - kwargs["latent_steps"]
        scores, _ = compressor._aggregate_prompt_scores_kv(
            device=legacy[-1][0].device, B=1, H_kv=int(legacy[-1][0].shape[1]),
            L_total=total, L_history=history, L_prompt=int(prompt.shape[1]),
            all_steps_attentions=kwargs["all_steps_attentions"], layer_idx=len(legacy)-1)
        f["statistics"] = torch.cat((f["statistics"], attention_features(scores, prompt)))
        if not all(torch.isfinite(x).all() for x in f.values()):
            raise ValueError("Non-finite captured feature")
        state["captured"].append(f)
        if len(state["captured"])-1 == state["target"]:
            raise StopAtState()
        return original_compress(**kwargs)

    model.generate_latent_batch = generate
    compressor.compress = compress
    return state
