# 文件作用：组织扩充数据上的同状态新旧特征对照。
# 主要职责：核对固定来源划分、准备并执行特征补采，比较不同输入并统计按题重采样的不确定性。

"""Frozen merged-source feature capture and diagnostic comparison."""
import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# 功能：读取已保存的 PyTorch 数据对象，并统一映射到 CPU。
# 输入：path 指向项目生成的 .pt 文件。
# 输出：torch.load 还原出的对象，通常是含特征、标签或计划信息的字典。
def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


# 功能：计算文件内容摘要，用于判断数据、划分或代码是否改变。
# 输入：path 是文件路径；按 1 MiB 块读取，不一次性载入整个文件。
# 输出：SHA-256 十六进制字符串，不修改原文件。
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


# 功能：通过同目录临时文件写入 JSON 结果。
# 输入：path 为目标 Path，obj 为可 JSON 序列化的对象。
# 输出：返回 None；写 UTF-8、缩进且保留中文的 JSON，再用 .tmp 文件替换目标文件。
def write(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


# 功能：以独立来源题为重采样单位，估计二分类指标波动和按题等权的非 B8 召回率。
# 输入：records、labels、predictions 按状态逐项对应；二分类 0 表示允许 B8，1 表示非 B8。
# 输出：含逐题混淆矩阵、有效非 B8 来源题等权召回率及 2000 次来源题 bootstrap 的 95% 区间的字典。
# 口径：同题所有状态一起抽样；重采样缺少某真实类别时跳过，无预测正例时不计算该次精确率。
def source_uncertainty(records,labels,predictions):
    ids=sorted({r['source_id'] for r in records})
    index={s:i for i,s in enumerate(ids)}
    cm=np.zeros((len(ids),2,2),dtype=np.int64)
    for r,y,p in zip(records,labels,predictions):cm[index[r['source_id']],int(y),int(p)]+=1
    rng=np.random.default_rng(20261112)
    bas=[];recalls=[];precisions=[]
    for _ in range(2000):
        m=cm[rng.integers(0,len(ids),size=len(ids))].sum(axis=0)
        if np.any(m.sum(axis=1)==0):continue
        bas.append(float(np.mean(np.diag(m)/m.sum(axis=1))))
        recalls.append(float(m[1,1]/m[1].sum()))
        if m[:,1].sum():precisions.append(float(m[1,1]/m[:,1].sum()))
    # 功能：把有效重采样指标转换为双侧 95% 百分位区间。
    # 输入：x 是一次指标的有效 bootstrap 数值列表。
    # 输出：[2.5% 分位数, 97.5% 分位数]；空列表返回 None。
    def ci(x):return np.quantile(x,[.025,.975]).tolist() if x else None
    rows=[dict(source_id=s,confusion_matrix=cm[i].tolist(),non_b8_states=int(cm[i,1].sum())) for i,s in enumerate(ids)]
    positive=[m[1,1]/m[1].sum() for m in cm if m[1].sum()]
    return dict(per_source=rows,source_equal_non_b8_recall=float(np.mean(positive)) if positive else None,
                cluster_bootstrap=dict(repetitions=2000,valid_binary_resamples=len(bas),
                                       balanced_accuracy_ci95=ci(bas),recall_ci95=ci(recalls),precision_ci95=ci(precisions)))


# 功能：按固定训练设置拟合线性探针，返回训练与验证分类结果。
# 输入：tx[N,D]、tm[N,C] 最优布尔掩码、vx[M,D]，以及 device、初始化 seed。
# 输出：(训练预测[N], 验证预测[M])，均为 CPU 类别索引；不保存模型。
# 流程：AdamW 固定训练 100 轮，使用集合标签损失和按类别支持度计算的样本权重，不根据验证表现挑选轮次。
def linear_predictions(tx,tm,vx,device,seed):
    from utils import set_seed
    set_seed(seed)
    model=torch.nn.Linear(tx.shape[1],tm.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    x=tx.to(device); masks=tm.to(device)
    support=masks.float().sum(0).clamp_min(1)
    weights=(support.max()/support).sqrt(); weights/=weights.mean()
    sample=(masks.float()*weights).sum(1)/masks.sum(1)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        logp=torch.log_softmax(model(x),dim=-1)
        losses=-torch.logsumexp(logp.masked_fill(~masks,float('-inf')),dim=-1)
        loss=(losses*sample).sum()/sample.sum();loss.backward();opt.step()
    with torch.no_grad():
        return model(x).argmax(1).cpu(),model(vx.to(device)).argmax(1).cpu()


# 功能：核对扩充数据和固定来源划分，创建或验证特征对照计划。
# 输入：a 提供 merged_folder、模型路径、生成参数、运行模式和输出目录。
# 输出：(plan, records, split)；plan 模式首次写 plan.json，已有计划必须与当前数据、设置及代码摘要一致。
# 校验：训练/验证来源互斥且覆盖全部记录，状态前缀唯一、预算顺序固定、任务为 GSM8K train、原搜索模型路径一致。
def prepare(a):
    data=read(a.merged_folder/'merged_dataset.pt')
    split=json.loads((a.merged_folder/'source_split.json').read_text(encoding='utf-8'))
    if split!=data['source_split']:raise ValueError('Saved splits disagree')
    records=data['records'];train=set(split['train_sources']);val=set(split['validation_sources'])
    if train&val or train|val!={r['source_id'] for r in records}:raise ValueError('Invalid source split')
    if tuple(data['budget_choices'])!=(8,16,32,64):raise ValueError('Budget choices mismatch')
    seen=set();sources={}
    for r in records:
        key=(r['source_id'],tuple(r['prefix']))
        if key in seen:raise ValueError('Duplicate state')
        seen.add(key)
        if r['task']!='gsm8k' or r['split']!='train':raise ValueError('Wrong dataset')
        sources[r['source_id']]=r
    paths=data['manifest']['arguments']
    old=set(split['old_train_sources'])|set(split['old_validation_sources'])
    source_rows=[dict(source_id=s,dataset_index=r['dataset_index'],partition='train' if s in train else 'validation',
                      screen_folder=paths['old_screen' if s in old else 'new_screen']) for s,r in sorted(sources.items())]
    for key in ('old_exhaustive','new_exhaustive'):
        files=sorted((Path(paths[key])/'samples').glob('sample_*.pt'))
        if not files or any(read(p)['model_name']!=a.model_name for p in files):
            raise ValueError('Model path differs from original collection')
    root=Path(__file__).resolve().parent
    names=['stage2_features.py','capture_expanded_impl.py','models.py','methods/latent_mas.py',
           'compression_methods/HOBF.py','expanded_feature_comparison.py','diagnose_router_features.py']
    plan=dict(schema_version=1,merged_sha256=sha(a.merged_folder/'merged_dataset.pt'),
              split_sha256=sha(a.merged_folder/'source_split.json'),sources=source_rows,state_count=len(records),
              settings={k:getattr(a,k) for k in ('model_name','prompt','latent_steps','max_new_tokens','pca_rank','sink_size','seed','think','latent_space_realign')},
              code_sha256={name:sha(root/name) for name in names})
    path=a.output_folder/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Inputs/code/settings changed; do not mix feature runs')
    else:
        if a.mode!='plan':raise ValueError('Run plan first')
        if a.output_folder.exists() and any(a.output_folder.iterdir()):raise ValueError('Output not empty')
        a.output_folder.mkdir(parents=True,exist_ok=True);write(path,plan)
    print(f"Sources: train={len(train)}, validation={len(val)}; states={len(records)}",flush=True)
    return plan,records,split


# 功能：在同一批历史状态、同一标签及固定划分上比较三类特征输入。
# 输入：a、plan 指定采集结果与运行设置；records 保留状态/标签，split 指定训练/验证来源题。
# 输出：返回 None；写 comparison.json、validation_predictions.json 和 comparison.md，含逐状态预测及按题重采样区间。
# 流程：核对旧 hidden 后替换输入，比较 last_hidden、mean_last5、18 维 statistics；分别运行二/四分类及三个种子线性探针、平衡 kNN。
# 边界：这里只评估固定历史状态上的分类，没有把预测预算接入后续 Agent 解题。
def report(a,plan,records,split):
    from stage2_teacher_table import digest
    from diagnose_router_features import _records_to_tensors,_fit_standardizer,_transform,_knn_predictions,_classification_metrics
    captured={}
    for s in plan['sources']:
        d=read(a.output_folder/'features'/f"sample_{s['dataset_index']:06d}.pt")
        if d['plan_digest']!=digest(plan):raise ValueError('Wrong feature plan')
        captured[s['dataset_index']]=d
    train_ids=set(split['train_sources']); results=[]
    predictions=[]
    for feature,dimension in [('last_hidden',2560),('mean_last5',2560),('statistics',18)]:
        alternate=[]
        for r in records:
            f=captured[r['dataset_index']]['features'][tuple(r['prefix'])]
            if not torch.allclose(f['last_hidden'],r['router_feature'],atol=1e-3,rtol=0):raise ValueError('Old feature mismatch')
            x=f[feature]
            if x.shape!=(dimension,) or not torch.isfinite(x).all():raise ValueError('Invalid feature')
            row=copy.copy(r);row['router_feature']=x;alternate.append(row)
        tr=[r for r in alternate if r['source_id'] in train_ids]
        va=[r for r in alternate if r['source_id'] not in train_ids]
        for binary in (True,False):
            print('Comparing',feature,'binary' if binary else 'four_way',flush=True)
            tx,th,ty,tm=_records_to_tensors(tr,binary=binary);vx,vh,vy,vm=_records_to_tensors(va,binary=binary)
            n=2 if binary else 4
            if len(set(ty.tolist()))!=n:raise ValueError('Missing training class')
            mu,scale=_fit_standardizer(tx)
            tx=_transform(tx,th,mu,scale,append_handoff=True);vx=_transform(vx,vh,mu,scale,append_handoff=True)
            runs=[]
            for seed in (42,43,44):
                tp,vp=linear_predictions(tx,tm,vx,torch.device(a.device),seed)
                runs.append(('linear',seed,tp,vp))
            vp=_knn_predictions(tx,ty,vx,k=5,num_classes=n,balanced=True)
            runs.append(('balanced_knn5',None,None,vp))
            for probe,seed,tp,vp in runs:
                result=dict(feature=feature,task='binary' if binary else 'four_way',probe=probe,seed=seed,
                            validation=_classification_metrics(vp,vy,vm,n))
                if tp is not None:result['train']=_classification_metrics(tp,ty,tm,n)
                if binary:result['source_analysis']=source_uncertainty(va,vy.tolist(),vp.tolist())
                results.append(result)
                predictions.append(dict(feature=feature,task=result['task'],probe=probe,seed=seed,
                                        rows=[dict(source_id=r['source_id'],prefix=list(r['prefix']),handoff=r['handoff_index'],
                                                   target=int(y),prediction=int(p)) for r,y,p in zip(va,vy,vp)]))
    lines=['# 扩充后的同题特征对照','',f"训练{len(split['train_sources'])}题；验证{len(split['validation_sources'])}题；总状态{len(records)}。",'',
           '| 输入 | 方法/种子 | 平衡准确率 | 非B8召回率 | 非B8精确率 | 按题等权非B8召回率 | 按题重采样BA 95%区间 |',
           '|---|---|---:|---:|---:|---:|---|']
    for r in results:
        if r['task']!='binary':continue
        m=r['validation'];s=r['source_analysis'];ci=s['cluster_bootstrap']['balanced_accuracy_ci95']
        interval='无' if ci is None else f'{ci[0]:.2%}–{ci[1]:.2%}'
        lines.append(f"| {r['feature']} | {r['probe']}/{r['seed']} | {m['balanced_accuracy']:.2%} | {m['larger_budget_recall']:.2%} | {m['larger_budget_precision']:.2%} | {s['source_equal_non_b8_recall']:.2%} | {interval} |")
    lines += ['', '所有方法使用同一来源划分，标准化只拟合训练数据；线性分类器固定100轮，种子42/43/44，不选择验证最佳轮次。',
              '按题等权召回率：先算每道有非B8状态的题的召回率，再平均，避免状态多的题占主导。',
              '95%区间按验证来源题整组重采样2000次；反映固定训练模型下验证题目抽样波动，不包含重新训练或改变划分的不确定性。',
              '非B8仍指B8不在成本最优集合。此为GSM8K train内部诊断，尚未测量完整Router答题表现、通信量与时延。',
              '补采检查原hidden与重复前缀特征一致性，不代表对新增题重新完成了64组合标签复核。']
    write(a.output_folder/'comparison.json',dict(plan=plan,probes=results))
    write(a.output_folder/'validation_predictions.json',predictions)
    (a.output_folder/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


# 功能：为扩充特征实验提供 plan、collect、report 三个命令行阶段。
# 输入：无显式参数；解析合并数据目录、模型/生成设置、输出目录、设备和可选采集 limit。
# 输出：返回 None；先核验计划，再按模式保存计划、调用补采程序或生成对照报告。
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','collect','report'])
    p.add_argument('--merged_folder',type=Path,required=True)
    p.add_argument('--output_folder',type=Path,required=True)
    p.add_argument('--model_name',required=True)
    p.add_argument('--cache_dir',default='/root/autodl-tmp/hf_cache')
    p.add_argument('--local_files_only',action='store_true')
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,default=None,help='Maximum newly captured sources; for smoke run only')
    p.set_defaults(prompt='sequential',latent_steps=20,max_new_tokens=2048,pca_rank=8,sink_size=4,seed=42,think=False,latent_space_realign=False)
    a=p.parse_args()
    if a.limit is not None and a.limit<=0:p.error('limit must be positive')
    x=a.merged_folder.resolve();y=a.output_folder.resolve()
    if x==y or x in y.parents or y in x.parents:p.error('Use separate output folder')
    plan,records,split=prepare(a)
    if a.mode=='collect':
        from capture_expanded_impl import collect
        collect(a,plan,records)
    elif a.mode=='report':report(a,plan,records,split)


if __name__=='__main__':main()
