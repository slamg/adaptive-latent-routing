# 文件作用：为扩充数据中的已有历史状态补采输入特征。
# 主要职责：按预算前缀重放推理，采集新特征并核对原hidden；沿用已有标签。

from pathlib import Path
import torch
from stage2_teacher_table import read, digest

# 功能：按已保存预算前缀重放扩充题目，在压缩前补采最后一步 hidden、最近 5 步均值和统计特征。
# 输入：a 提供模型、数据缓存、生成设置、输出目录和 limit；plan 指定来源题及筛选文件；records 提供待采状态、旧特征和标签。
# 输出：返回 None；按题写入 features/sample_*.pt，保存前缀到特征的映射、计划摘要及旧特征最大误差。
# 流程：优先重放较长前缀以同时取得父状态；旧 hidden 误差须不超过 1e-3，重复前缀的新特征也须一致。
# 注意：沿用原标签，不重新跑 64 组合复核标签；limit 限制本次新采集题数，已有同计划文件会跳过。
def collect(a,plan,records):
    from collect_router_data import load_task
    from compression_methods import HOBF
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from utils import set_seed,auto_device
    from stage2_features import install_capture,StopAtState
    dataset=list(load_task("gsm8k","train",cache_dir=a.cache_dir))
    a.method="latent_mas"; a.task="gsm8k"; a.split="train"
    a.text_mas_context_length=-1; a.latent_only=False; a.sequential_info_only=False
    set_seed(a.seed)
    model=ModelWrapper(a.model_name,auto_device(a.device),args=a)
    compressor=HOBF(sink_size=a.sink_size,kv_budget=32,pca_rank=a.pca_rank)
    method=LatentMASMethod(model,compressor=compressor,latent_steps=a.latent_steps,judger_max_new_tokens=a.max_new_tokens,
        temperature=0.0,top_p=1.0,generate_bs=1,budget_policy="fixed",kv_budget=32,budget_choices=(8,16,32,64),args=a)
    capture=install_capture(model,compressor)
    folder=a.output_folder/"features"; folder.mkdir(exist_ok=True)
    completed=0
    for c in plan["sources"]:
        i=c["dataset_index"]; path=folder/f"sample_{i:06d}.pt"
        if path.exists():
            existing=read(path)
            if existing['plan_digest']!=digest(plan): raise ValueError('Resume feature plan mismatch')
            continue
        saved=read(Path(c["screen_folder"])/"samples"/f"sample_{i:06d}.pt")
        if saved["question"]!=dataset[i]["question"] or saved["gold"]!=dataset[i]["gold"]:
            raise ValueError(f"Current dataset identity differs at {i}")
        wanted={tuple(r["prefix"]):r for r in records if r["dataset_index"]==i}
        # Replay leaf prefixes first: they also yield their root/parent features.
        collected={}; maxdiff=0.0
        for prefix in sorted(wanted,key=lambda p:(-len(p),p)):
            if prefix in collected: continue
            capture["captured"]=[]; capture["target"]=len(prefix)
            set_seed(a.seed+i*1000)
            schedule=prefix+(8,)*(3-len(prefix))
            try:
                method.run_item(dataset[i],budget_schedule=dict(enumerate(schedule)))
            except StopAtState:
                pass
            else:
                raise ValueError("Capture did not stop at requested pre-compression state")
            for h,f in enumerate(capture["captured"]):
                key=prefix[:h]
                if key not in wanted: continue
                diff=float((f["last_hidden"]-wanted[key]["router_feature"]).abs().max())
                maxdiff=max(maxdiff,diff)
                if diff>1e-3:
                    raise ValueError(f"Old/new state mismatch at source {i}, prefix {key}: {diff}; cannot reuse old label")
                if key in collected:
                    for name in f:
                        if not torch.allclose(f[name],collected[key][name],atol=1e-3,rtol=0):
                            raise ValueError("Repeated prefix new features differ")
                collected[key]=f
        if set(collected)!=set(wanted): raise ValueError("Missing collected states")
        tmp=path.with_suffix(".tmp")
        torch.save({"dataset_index":i,"plan_digest":digest(plan),"features":collected,"max_old_feature_diff":maxdiff},tmp)
        tmp.replace(path)
        completed+=1
        print(f"Captured source {i}, {len(collected)} states, max old feature diff={maxdiff}",flush=True)
        if a.limit and completed>=a.limit: break

