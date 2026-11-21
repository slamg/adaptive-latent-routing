# 文件作用：组织早期标签复核和特征诊断实验。
# 主要职责：准备抽查计划、重跑预算组合复核标签、采集特征并汇总对照报告。

"""Plan, audit labels, collect new features and compare frozen feature inputs."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean

import torch


# 功能：从项目 .pt 文件还原保存的数据对象。
# 输入：path 为文件路径；torch.load 将所有张量映射到 CPU。
# 输出：保存对象本身，通常含题目、搜索轨迹、标签或特征。
def read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# 功能：为 JSON 可序列化内容生成确定性的摘要，标记计划版本。
# 输入：value 为可 JSON 序列化的对象，编码时按字典键排序。
# 输出：SHA-256 十六进制字符串；只在内存计算，不写文件。
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# 功能：从一题的搜索记录中提取可用于复核的 Oracle 标签签名。
# 输入：data 包含 global_oracle 和 reserve_state_examples。
# 输出：含全局状态、最低成本、排序后的最优方案集合及状态前缀/目标预算/最优掩码的字典。
# 注意：这里只提取现有标签用于比较，不重新运行题目或计算最优方案。
def label_signature(data):
    return {"status":data["global_oracle"]["status"],
            "cost":data["global_oracle"]["minimum_cost"],
            "schedules": sorted(tuple(k) for k in data["global_oracle"]["optimal_schedules"]),
            "states": sorted((tuple(r["prefix"]), r.get("target_budget"), tuple(r.get("optimal_budget_mask") or ())) for r in data["reserve_state_examples"])}


# 功能：从运行参数中选出需要固定比较的模型与生成设置。
# 输入：a 为包含模型路径、prompt、latent_steps、生成长度、压缩参数和随机种子等字段的命名空间。
# 输出：上述指定字段组成的字典，用于保存及核对实验计划。
def config(a):
    return {k:getattr(a,k) for k in ["model_name","prompt","latent_steps","max_new_tokens","pca_rank","sink_size","seed","think","latent_space_realign"]}


# 功能：把实验计划或结果写入 JSON，并用临时文件替换目标。
# 输入：path 为目标 Path，value 为可 JSON 序列化内容。
# 输出：返回 None；生成 UTF-8 文件，保留中文并缩进。
def write_json(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


# 功能：从原来源划分中选择小规模特征诊断题和具有完整搜索记录的标签抽查题。
# 输入：a 提供筛选/Oracle 目录、模型与生成配置、输出目录及运行模式。
# 输出：(plan, selected, full)，依次是计划、入选状态记录、题目索引到原搜索文件/数据的映射；首次 plan 模式保存 plan.json。
# 流程：训练侧 easy/hard 各取 12 题，验证侧各取 4 题；四个分组各挑 2 题，共 8 题用于标签复核。
# 校验：检查模型一致、分组题数及二分类覆盖，并摘要绑定入选标签与旧 hidden，后续阶段必须使用同一计划。
def prepare(a):
    from router_training_data import build_router_examples, split_source_ids
    records,_=build_router_examples(a.screen_folder,a.oracle_folder,view="primary_reserve")
    train,val=split_source_ids(records,validation_fraction=0.2,seed=20261001)
    full={int(d["sample_id"]):(path,d) for path in sorted((a.oracle_folder/"samples").glob("sample_*.pt")) for d in [read(path)]}
    if not full or {d["model_name"] for _,d in full.values()}!={a.model_name}:
        raise ValueError("Use the same model path as the original Oracle shards")
    sources={r["source_id"]:r for r in records}
    rng=random.Random(20261107)
    chosen=[]; audit=[]
    for split,source_ids,n in [("train",train,12),("validation",val,4)]:
        for kind in ["easy","hard"]:
            pool=sorted(s for s in source_ids if sources[s]["source_kind"]==kind)
            rng.shuffle(pool)
            # Prefer easy audit sources for which full 64-way labels already exist.
            pool.sort(key=lambda s:sources[s]["dataset_index"] not in full)
            if len(pool)<n:
                raise ValueError(f"Insufficient independent {kind}/{split} sources")
            for s in pool[:n]:
                r=sources[s]
                chosen.append(dict(source_id=s,dataset_index=r["dataset_index"],kind=kind,partition=split))
            candidates=[s for s in pool[:n] if sources[s]["dataset_index"] in full]
            if len(candidates)<2:
                raise ValueError(f"Need 2 exhaustive audit sources in {kind}/{split}")
            audit += [sources[s]["dataset_index"] for s in candidates[:2]]
    ids={c["source_id"] for c in chosen}
    selected=[r for r in records if r["source_id"] in ids]
    part={c["source_id"]:c["partition"] for c in chosen}
    counts={}
    for split in ["train","validation"]:
        subset=[r for r in selected if part[r["source_id"]]==split]
        counts[split]={"states":len(subset),"non_b8_states":sum(not r["optimal_budget_mask"][0] for r in subset)}
        if counts[split]["non_b8_states"] in (0,len(subset)):
            raise ValueError(f"Binary comparison lacks a class in {split}")
    # Detect subsequent edits to labels and saved old features across stages.
    h=hashlib.sha256()
    for r in selected:
        h.update(json.dumps([r["source_id"],r["prefix"],r["target_class"],r["optimal_budget_mask"]]).encode())
        h.update(r["router_feature"].contiguous().numpy().tobytes())
    plan={"schema_version":1,"settings":config(a),"sources":chosen,"audit_ids":audit,"old_data_digest":h.hexdigest(),
          "train_sources":24,"validation_sources":8,"state_count":len(selected),"counts":counts,"original_split_seed":20261001}
    path=a.output_folder/"plan.json"
    if path.exists():
        saved=json.loads(path.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError("Saved plan differs from current inputs/configuration; use a new output folder")
    else:
        if a.mode!="plan":
            raise ValueError("Run plan mode first")
        a.output_folder.mkdir(parents=True,exist_ok=True)
        write_json(path,plan)
    return plan,selected,full


# 功能：对子集题重新执行完整 64 种预算组合，并与原搜索结果逐项核对。
# 输入：a 提供运行设置；plan.audit_ids 指定抽查题；full 提供对应的原搜索数据。
# 输出：返回 None；运行 collect_router_data.py 保存 label_replay 数据，并写 label_audit.json 各项一致性及 passed。
# 口径：passed 要求每题 64 条正确性、最低成本、完整最优方案集合及状态标签一致；预测文本一致数量另记，但不参与 passed。
def audit_labels(a,plan,full):
    folder=a.output_folder/"label_replay"
    indices=a.output_folder/"audit_indices.txt"
    indices.write_text("".join(f"{i}\n" for i in plan["audit_ids"]),encoding="utf-8")
    cmd=[sys.executable,"-u",str(Path(__file__).with_name("collect_router_data.py")),
         "--model_name",a.model_name,"--task","gsm8k","--split","train",
         "--sample_indices_file",str(indices),"--max_samples","-1","--output_folder",str(folder),"--resume",
         "--prompt",a.prompt,"--latent_steps",str(a.latent_steps),"--max_new_tokens",str(a.max_new_tokens),
         "--pca_rank",str(a.pca_rank),"--sink_size",str(a.sink_size),"--seed",str(a.seed),"--device",a.device]
    if a.cache_dir: cmd += ["--cache_dir",a.cache_dir]
    for flag in ["local_files_only","think","latent_space_realign"]:
        if getattr(a,flag): cmd.append("--"+flag)
    subprocess.run(cmd,check=True)
    rows=[]
    for i in plan["audit_ids"]:
        old=full[i][1]
        new=read(folder/"samples"/f"sample_{i:06d}.pt")
        if old["question"]!=new["question"] or old["gold"]!=new["gold"]:
            raise ValueError("Audit question identity mismatch")
        oldruns={tuple(r["schedule"]):r for r in old["rollouts"]}
        newruns={tuple(r["schedule"]):r for r in new["rollouts"]}
        if set(oldruns)!=set(newruns) or len(oldruns)!=64:
            raise ValueError("Audit grid incomplete")
        identity=next(c for c in plan["sources"] if c["dataset_index"]==i)
        sig_old,sig_new=label_signature(old),label_signature(new)
        rows.append({"dataset_index":i,"kind":identity["kind"],"partition":identity["partition"],
            "same_correctness":sum(oldruns[k]["correct"]==newruns[k]["correct"] for k in oldruns),
            "same_prediction":sum(oldruns[k].get("prediction")==newruns[k].get("prediction") for k in oldruns),
            "same_minimum_cost":sig_old["cost"]==sig_new["cost"],
            "same_optimal_schedules":sig_old["schedules"]==sig_new["schedules"],
            "same_state_labels":sig_old["states"]==sig_new["states"]})
    passed=all(r["same_correctness"]==64 and r["same_minimum_cost"] and r["same_optimal_schedules"] and r["same_state_labels"] for r in rows)
    write_json(a.output_folder/"label_audit.json",{"rows":rows,"passed":passed})
    print(f"Label audit completed: passed={passed}. Review label_audit.json.",flush=True)


# 功能：在小规模标签抽查通过后，重放选定预算前缀补采三类输入特征。
# 输入：a 为模型/生成和目录配置，plan 为固定题目计划，records 提供目标状态、原 hidden 与已有标签。
# 输出：返回 None；按题保存 features/sample_*.pt，含前缀到特征的映射、计划摘要和旧 hidden 最大误差。
# 流程：到目标压缩前停止，校验旧 hidden 与重复前缀特征，已有特征文件跳过；limit 统计本次新增采集题。
# 边界：补采本身沿用旧标签，完整标签重新搜索发生在 audit_labels。
def collect(a,plan,records):
    from collect_router_data import load_task
    from compression_methods import HOBF
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from utils import set_seed,auto_device
    from stage2_features import install_capture,StopAtState
    audit=json.loads((a.output_folder/"label_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Label audit changed: investigate before feature comparison")
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
        if path.exists(): continue
        saved=read(a.screen_folder/"samples"/f"sample_{i:06d}.pt")
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


# 功能：汇总标签抽查，并在同一批历史状态上比较旧 hidden、最近 5 步均值和统计特征。
# 输入：a 指定目录与设备，plan 指定采集/划分计划，records 提供待替换特征的原状态记录。
# 输出：返回 None；保存 second_table.json 和 second_table.md，并打印报告。
# 流程：核对抽查通过及特征计划，训练侧拟合标准化；二/四分类各跑三个种子的固定轮次线性探针与平衡 kNN。
# 边界：缺训练类别的任务会跳过；这里只诊断历史状态，不测完整 Router 解题正确率、通信量或时延。
def report(a,plan,records):
    from diagnose_router_features import _records_to_tensors,_fit_standardizer,_transform,_linear_probe,_knn_predictions,_classification_metrics
    from utils import set_seed,auto_device
    audit=json.loads((a.output_folder/"label_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise ValueError("Audit did not pass; investigate before comparing features")
    captured={}
    for c in plan["sources"]:
        d=read(a.output_folder/"features"/f"sample_{c['dataset_index']:06d}.pt")
        if d["plan_digest"]!=digest(plan): raise ValueError("Feature file belongs to different plan")
        captured[c["dataset_index"]]=d
    partition={c["source_id"]:c["partition"] for c in plan["sources"]}
    result=[]
    for feature in ["last_hidden","mean_last5","statistics"]:
        alternate=[]
        for r in records:
            row=copy.copy(r)
            row["router_feature"]=captured[r["dataset_index"]]["features"][tuple(r["prefix"])][feature]
            alternate.append(row)
        train=[r for r in alternate if partition[r["source_id"]]=="train"]
        val=[r for r in alternate if partition[r["source_id"]]=="validation"]
        for binary in [True,False]:
            tx,th,ty,tm=_records_to_tensors(train,binary=binary)
            vx,vh,vy,vm=_records_to_tensors(val,binary=binary)
            n=2 if binary else 4
            if len(set(ty.tolist()))<n:
                result.append({"feature":feature,"task":"binary" if binary else "four_way","status":"missing_training_classes"})
                continue
            mu,scale=_fit_standardizer(tx)
            tx=_transform(tx,th,mu,scale,append_handoff=True)
            vx=_transform(vx,vh,mu,scale,append_handoff=True)
            for seed in [42,43,44]:
                set_seed(seed)
                linear,_=_linear_probe(tx,ty,tm,vx,vy,vm,num_classes=n,epochs=100,learning_rate=0.01,weight_decay=0.001,device=auto_device(a.device))
                result.append({"feature":feature,"task":"binary" if binary else "four_way","probe":"linear","seed":seed,"train":linear["train"],"validation":linear["validation"],"feature_dim":int(train[0]["router_feature"].numel())})
            pred=_knn_predictions(tx,ty,vx,k=5,num_classes=n,balanced=True)
            result.append({"feature":feature,"task":"binary" if binary else "four_way","probe":"balanced_knn5","validation":_classification_metrics(pred,vy,vm,n)})
    audit=json.loads((a.output_folder/"label_audit.json").read_text(encoding="utf-8"))
    lines=["# 标签复现与新旧特征对照", "", "小规模筛选实验：32道独立题，24训练、8验证；沿用原来源题划分，每侧easy/hard各半。状态数为辅助统计。", "", "## 标签复核", "", "| 独立题数 | 每题组合数 | 所有组合正确性均一致的题数 | 最低预算一致题数 | 完整最优集合一致题数 | 状态标签一致题数 |", "|---:|---:|---:|---:|---:|---:|"]
    ar=audit["rows"]
    lines.append(f"| {len(ar)} | 64 | {sum(r['same_correctness']==64 for r in ar)} | {sum(r['same_minimum_cost'] for r in ar)} | {sum(r['same_optimal_schedules'] for r in ar)} | {sum(r['same_state_labels'] for r in ar)} |")
    lines += ["", "## 相同题目和状态的二分类特征对照", "", "| 输入特征 | 维度 | 验证平衡准确率 | 非B8召回率 | 非B8精确率 |", "|---|---:|---:|---:|---:|"]
    for feature in ["last_hidden","mean_last5","statistics"]:
        group=[r for r in result if r["feature"]==feature and r["task"]=="binary" and r.get("probe")=="linear"]
        if not group:
            lines.append(f"| {feature} | — | 训练缺类 | — | — |")
            continue
        # 功能：将当前特征三个线性探针种子的验证指标取平均并格式化。
        # 输入：key 是验证指标名称；样本来自外层 group 中当前特征的二分类线性探针结果。
        # 输出：保留两位小数的百分比字符串，用于 Markdown 表格。
        def avg(key):return f"{mean(r['validation'][key] for r in group):.2%}"
        lines.append(f"| {feature} | {group[0]['feature_dim']} | {avg('macro_recall')} | {avg('larger_budget_recall')} | {avg('larger_budget_precision')} |")
    lines += ["", "线性探针固定同一训练设置，表中为3个初始化种子的均值；不是3个不同题目划分。完整四分类、kNN及混淆矩阵见JSON。本次不选择最佳epoch、不根据验证分数调参。", "", f"训练状态数：{sum(partition[r['source_id']]=='train' for r in records)}；验证状态数：{sum(partition[r['source_id']]=='validation' for r in records)}。",
              "", "所有特征均在当前压缩前采集。二分类仍问B8是否在成本最优动作集合中；不改为成功可行性目标。训练/验证题从原集合选取，不是新的独立测试集；困难题过采样且验证来源少，只用于判断方向。统计特征维度更低，差异同时包含信息汇总及降维效果，不能单独归因于新增信息。未评估完整Router的答题准确率、KV和新增时延。"]
    (a.output_folder/"second_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(a.output_folder/"second_table.json",{"plan":plan,"label_audit":audit,"probes":result,"train_states":sum(partition[r['source_id']]=='train' for r in records),"validation_states":sum(partition[r['source_id']]=='validation' for r in records)})
    print("\n".join(lines))


# 功能：提供早期小规模诊断的 plan、audit、collect、report 命令行入口。
# 输入：无显式参数；命令行给出模型、数据目录、运行阶段、生成设置和可选 limit。
# 输出：返回 None；校验参数与输出目录、准备固定计划，然后调度标签复核、特征补采或报告。
# 流程：audit/collect 阶段要求 confirm_settings 标志，提醒调用方已经核对原生成设置。
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name",required=True)
    p.add_argument("--output_folder",default="/root/autodl-tmp/router_stage2/v1")
    p.add_argument("--cache_dir",default=None)
    p.add_argument("--local_files_only",action="store_true")
    p.add_argument("--device",default="cuda")
    p.add_argument("--prompt",choices=["sequential","hierarchical"],default="sequential")
    p.add_argument("--latent_steps",type=int,default=20)
    p.add_argument("--max_new_tokens",type=int,default=2048)
    p.add_argument("--pca_rank",type=int,default=8)
    p.add_argument("--sink_size",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--think",action="store_true")
    p.add_argument("--latent_space_realign",action="store_true")
    p.add_argument("mode",choices=["plan","audit","collect","report"])
    p.add_argument("--screen_folder",type=Path,required=True)
    p.add_argument("--oracle_folder",type=Path,required=True)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--confirm_settings",action="store_true")
    p.set_defaults(output_folder="/root/autodl-tmp/router_stage2/v1")
    a=p.parse_args(); a.output_folder=Path(a.output_folder)
    if a.latent_steps<2: p.error("latent_steps must be >=2")
    if a.limit is not None and a.limit<=0:p.error("limit must be positive")
    for source in [a.screen_folder.resolve(),a.oracle_folder.resolve()]:
        target=a.output_folder.resolve()
        if target==source or source in target.parents or target in source.parents:
            p.error("Use a separate stage2 output directory")
    plan,records,full=prepare(a)
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)
    if a.mode=="plan":return
    if a.mode in ["audit","collect"] and not a.confirm_settings:
        p.error("Check original run settings, then supply --confirm_settings")
    if a.mode=="audit":audit_labels(a,plan,full)
    elif a.mode=="collect":collect(a,plan,records)
    else:report(a,plan,records)


if __name__=="__main__":main()
