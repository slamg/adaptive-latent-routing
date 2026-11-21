# 文件作用：组织扩充数据上的 Router 训练及完整解题评估。
# 主要职责：核对固定划分和预算标签，调用训练程序，并在线比较两类Router与固定B8/B32的正确率、KV量及时延。

"""Train fresh routers on the frozen expanded split and evaluate live rollouts."""
import argparse
import copy
import hashlib
import json
import random
import sys
import time
from numbers import Integral
from pathlib import Path
import torch


# 功能：读取扩充数据、补采特征或 Router 检查点。
# 输入：p 为项目生成的 PyTorch 文件路径。
# 输出：映射到 CPU 的原保存对象，允许还原张量以外的元数据。
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
# 功能：对文件内容计算 SHA-256，用于记录实验依赖版本。
# 输入：p 为文件路径；按 1 MiB 分块读取。
# 输出：十六进制摘要字符串，不修改文件。
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
# 功能：通过临时文件保存 UTF-8 JSON。
# 输入：p 为目标 Path，x 为可 JSON 序列化的结果。
# 输出：返回 None；先写同目录 .tmp，再替换目标文件。
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
# 功能：在计时边界等待当前 CUDA 设备的工作完成。
# 输入：无参数；检查当前环境是否可用 CUDA。
# 输出：返回 None；有 CUDA 时调用 synchronize，无 CUDA 时直接结束。
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()


EXPANDED_BUDGET_CHOICES = (8, 16, 32, 64)


# 功能：校验预算数值、类别索引及并列最优掩码使用同一套语义，避免类别对应关系错位。
# 输入：data 包含 budget_choices 和 records；每条记录需有 target_class、target_budget、optimal_budget_mask。
# 输出：固定顺序的候选元组 (8,16,32,64)；非整数/顺序错误、目标不对应或掩码无效时抛出 ValueError。
def validate_budget_contract(data):
    """Expanded v2 artifacts use one fixed vocabulary, including class order."""
    choices = tuple(data.get('budget_choices', ()))
    if (any(isinstance(b, bool) or not isinstance(b, Integral) for b in choices)
            or choices != EXPANDED_BUDGET_CHOICES):
        raise ValueError('Expanded budget choices must be exactly (8, 16, 32, 64) in that order')
    for index, record in enumerate(data['records']):
        target = record.get('target_class')
        budget = record.get('target_budget')
        mask = record.get('optimal_budget_mask')
        if (isinstance(target, bool) or not isinstance(target, Integral)
                or not 0 <= target < len(choices)
                or isinstance(budget, bool) or not isinstance(budget, Integral)
                or choices[target] != budget):
            raise ValueError(f'Budget class/value mismatch in record {index}')
        if (mask is None or len(mask) != len(choices)
                or any(value not in (False, True) for value in mask)
                or not any(mask) or not mask[target]):
            raise ValueError(f'Invalid optimal budget mask in record {index}')
    return choices


# 功能：读取扩充数据和固定划分，并确认符合本次 v2 实验约定。
# 输入：a.data 指向含 merged_dataset.pt、source_split.json 的目录。
# 输出：(d,s)，分别为合并数据对象与划分字典；要求保存的两份划分一致、来源互斥且完整覆盖。
# 范围：当前实现还固定检查 773 道训练来源、193 道验证来源和 4466 条状态，不是任意规模数据入口。
def load_data(a):
    d=read(a.data/'merged_dataset.pt')
    validate_budget_contract(d)
    s=json.loads((a.data/'source_split.json').read_text(encoding='utf-8'))
    if s!=d['source_split']:raise ValueError('Split mismatch')
    tr=set(s['train_sources']);va=set(s['validation_sources'])
    if tr&va or tr|va!={r['source_id'] for r in d['records']}:raise ValueError('Invalid source split')
    if (len(tr),len(va),len(d['records']))!=(773,193,4466):raise ValueError('Unexpected expanded dataset')
    return d,s


# 功能：把选定特征接入原训练器，在扩充数据的固定来源划分上重新训练 Router。
# 输入：a 指定特征类型、补采/数据/输出目录和设备，d 为合并记录，s 为固定来源划分。
# 输出：返回 None；在 train_<feature> 保存训练产物，并给 best_router.pt 补入数据摘要、特征计划摘要和划分元数据。
# 流程：按题及预算前缀读取补采特征、核对旧 hidden 后替换输入；拒绝覆盖非空训练目录。
# 副作用：临时用途是替换训练模块的数据构造/划分函数及 sys.argv 后调用其 main；当前函数不会恢复这些进程内替换。
def train(a,d,s):
    choices=validate_budget_contract(d)
    import train_budget_router as original
    from stage2_teacher_table import digest
    feature=a.feature
    records=d['records']
    plan=json.loads((a.features/'plan.json').read_text(encoding='utf-8'))
    if plan['merged_sha256']!=sha(a.data/'merged_dataset.pt'):raise ValueError('Features from different data')
    captured={}
    changed=[]
    for r in records:
        i=r['dataset_index']
        if i not in captured:
            x=read(a.features/'features'/f'sample_{i:06d}.pt')
            if x['plan_digest']!=digest(plan):raise ValueError('Feature plan mismatch')
            captured[i]=x['features']
        f=captured[i][tuple(r['prefix'])]
        if not torch.allclose(f['last_hidden'],r['router_feature'],atol=1e-3,rtol=0):raise ValueError('Feature replay mismatch')
        row=copy.copy(r);row['router_feature']=f[feature];changed.append(row)
    out=a.output/('train_'+feature)
    if out.exists() and any(out.iterdir()):raise ValueError('Training output exists; preserve it and use a new output root for retraining')
    original.build_router_examples=lambda *args,**kw:(changed,{'view':'primary_reserve','feature':feature,'examples':len(changed)})
    original.split_source_ids=lambda *args,**kw:(tuple(s['train_sources']),tuple(s['validation_sources']))
    sys.argv=['train_budget_router.py','--screen_folder','unused','--exhaustive_folder','unused',
              '--output_folder',str(out),'--device',a.device,'--router_hidden_dim','256','--dropout','0.1',
              '--learning_rate','0.001','--weight_decay','0.0001','--epochs','100','--patience','15','--seed','42',
              '--budget_choices',*[str(b) for b in choices]]
    original.main()
    ck=read(out/'best_router.pt')
    ck['experiment_v2']=dict(feature=feature,data_sha256=sha(a.data/'merged_dataset.pt'),
                              feature_plan_sha256=sha(a.features/'plan.json'),source_split=s)
    temp=out/'best_router.tmp';torch.save(ck,temp);temp.replace(out/'best_router.pt')


# 功能：给模型安装在线特征包装，使完整解题时的统计 Router 使用与离线采集相同的特征算法。
# 输入：model 为模型包装对象，compressor 提供 KV/注意力辅助接口，state 共享当前策略及特征耗时计数。
# 输出：返回 None；替换 model.generate_latent_batch。仅 statistics 策略计算新特征，其他策略保留原返回行为。
def install_live_features(model,compressor,state):
    from stage2_features import hidden_features,attention_features
    original=model.generate_latent_batch
    # 功能：在统计策略下，随当前真实推理路径计算压缩前的 18 维特征。
    # 输入：args/kwargs 传给原 latent 生成函数；使用 latent_steps、attention_mask 和外层 state.feature。
    # 输出：统计策略返回原结果前三项加 [1,18] 特征；其他策略原样返回原结果。特征放到原 hidden 所在设备。
    # 副作用：临时注册/移除 forward hook，累计 state.feature_seconds，并校验 hidden 一致性及统计特征形状/有限值。
    def generate(*args,**kwargs):
        if state['feature']!='statistics':return original(*args,**kwargs)
        hidden=[]
        # 功能：为在线统计特征收集每次前向最后一层、最后位置的 hidden，并记录拷贝耗时。
        # 输入：module、inputs、output 是 forward hook 参数；从 output.hidden_states 读取隐藏状态。
        # 输出：返回 None，不改变模型输出；向外层 hidden 添加 [批大小,D] CPU 浮点副本并累计 feature_seconds。
        def hook(module,inputs,output):
            if output.hidden_states is not None:
                sync();start=time.perf_counter()
                hidden.append(output.hidden_states[-1][:,-1,:].detach().float().cpu().clone())
                state['feature_seconds']+=time.perf_counter()-start
        handle=model.model.register_forward_hook(hook)
        try:result=original(*args,**kwargs)
        finally:handle.remove()
        sync();start=time.perf_counter()
        f=hidden_features(hidden[-kwargs['latent_steps']:])
        if not torch.allclose(f['last_hidden'],result[3][0].float().cpu(),atol=1e-6,rtol=0):raise ValueError('Live hidden mismatch')
        legacy=compressor._as_legacy_tuple(result[0]);mask=kwargs['attention_mask']
        total=int(legacy[-1][0].shape[-2]);history=total-int(mask.shape[1])-kwargs['latent_steps']
        scores,_=compressor._aggregate_prompt_scores_kv(device=legacy[-1][0].device,B=1,
            H_kv=int(legacy[-1][0].shape[1]),L_total=total,L_history=history,L_prompt=int(mask.shape[1]),
            all_steps_attentions=result[1],layer_idx=len(legacy)-1)
        statistics=torch.cat((f['statistics'],attention_features(scores,mask)))
        if statistics.shape!=(18,) or not torch.isfinite(statistics).all():raise ValueError('Invalid live statistics')
        feature=statistics.unsqueeze(0).to(result[3].device)
        sync();state['feature_seconds']+=time.perf_counter()-start
        return (*result[:3],feature)
    model.generate_latent_batch=generate


# 功能：在同一批开发验证题上完整运行两类 Router 与固定 B8/B32，对比答题、KV 和时延。
# 输入：a 指定数据/特征/权重目录、模型、设备和 limit；d 为合并数据，s 为固定来源划分。
# 输出：返回 None；保存 evaluation/manifest.json、各策略逐题 JSON 以及端到端汇总报告。
# 流程：核验候选语义、权重来源和题目身份，安装在线特征，先用训练题热身；逐题随机化四策略执行顺序并支持跳过已有结果。
# 范围：当前生成设置在函数内固定；limit 按本次遍历的验证题数计数，部分完成时汇总会标记不完整。
def evaluate(a,d,s):
    choices=validate_budget_contract(d)
    from budget_router import BudgetRouter
    from collect_router_data import load_task,summarize_result
    from compression_methods import HOBF
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device,set_seed
    a.method='latent_mas';a.task='gsm8k';a.split='train';a.prompt='sequential'
    a.latent_steps=20;a.max_new_tokens=2048;a.pca_rank=8;a.sink_size=4;a.seed=42
    a.think=False;a.latent_space_realign=False;a.text_mas_context_length=-1;a.latent_only=False;a.sequential_info_only=False
    out=a.output/'evaluation';out.mkdir(parents=True,exist_ok=True)
    strategies=['fixed8','fixed32','last_hidden','statistics']
    checkpoints={f:read(a.output/('train_'+f)/'best_router.pt') for f in strategies[2:]}
    for f,c in checkpoints.items():
        if tuple(c['router_config']['budget_choices'])!=choices:
            raise ValueError('Checkpoint budget choices mismatch')
        meta=c['experiment_v2']
        if meta['feature']!=f or meta['data_sha256']!=sha(a.data/'merged_dataset.pt') or meta['source_split']!=s:
            raise ValueError('Checkpoint provenance mismatch')
    source_rows={r['source_id']:r for r in d['records']}
    ids=sorted(source_rows[x]['dataset_index'] for x in s['validation_sources'])
    plan=json.loads((a.features/'plan.json').read_text(encoding='utf-8'))
    if plan['settings']['model_name']!=a.model_name:raise ValueError('Model path mismatch')
    root=Path(__file__).parent
    manifest=dict(data_sha256=sha(a.data/'merged_dataset.pt'),ids=ids,model_name=a.model_name,
                  settings=plan['settings'],checkpoint_sha256={f:sha(a.output/('train_'+f)/'best_router.pt') for f in strategies[2:]},
                  code_sha256={f:sha(root/f) for f in ['run_router_v2.py','stage2_features.py','models.py','methods/latent_mas.py','compression_methods/HOBF.py','budget_router.py']})
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text(encoding='utf-8'))!=manifest:raise ValueError('Evaluation settings changed')
    write(mp,manifest)
    dataset=list(load_task('gsm8k','train',cache_dir=a.cache_dir))
    # Check live question identity against saved screening records.
    locations={c['dataset_index']:c['screen_folder'] for c in plan['sources']}
    for i in ids:
        saved=read(Path(locations[i])/'samples'/f'sample_{i:06d}.pt')
        if saved['question']!=dataset[i]['question'] or saved['gold']!=dataset[i]['gold']:raise ValueError('Question identity changed')
    set_seed(42);device=auto_device(a.device)
    model=ModelWrapper(a.model_name,device,args=a)
    compressor=HOBF(sink_size=4,kv_budget=32,pca_rank=8)
    state={'feature':None,'feature_seconds':0.,'router_seconds':0.}
    install_live_features(model,compressor,state)
    routers={}
    for f,c in checkpoints.items():
        router=BudgetRouter(**c['router_config']);router.load_state_dict(c['state_dict']);router.to(device).eval()
        select=router.select_budget
        # 功能：为当前 Router 的预算选择接口增加 GPU 同步计时。
        # 输入：args、kw 原样传递；默认参数 _select 绑定本轮 Router 的原 select_budget，避免循环闭包绑定错对象。
        # 输出：原 select_budget 的返回结果原样返回；在 inference_mode 下执行并累计 state.router_seconds。
        def timed(*args,_select=select,**kw):
            sync();start=time.perf_counter()
            with torch.inference_mode():result=_select(*args,**kw)
            sync();state['router_seconds']+=time.perf_counter()-start
            return result
        router.select_budget=timed;routers[f]=router
    methods={}
    for strategy in strategies:
        methods[strategy]=LatentMASMethod(model,compressor=compressor,latent_steps=20,judger_max_new_tokens=2048,
            temperature=0.,top_p=1.,generate_bs=1,budget_policy='fixed' if strategy.startswith('fixed') else 'router',
            kv_budget=8 if strategy=='fixed8' else 32,budget_choices=choices,budget_router=routers.get(strategy),args=a)
    # 功能：让指定策略完整解答一题，同时记录三个节点的预算和端到端时间。
    # 输入：strategy 为 fixed8/fixed32/last_hidden/statistics，i 为原数据集题目索引；模型等对象来自外层 evaluate。
    # 输出：一条结果字典，含题目/策略标识、答案及正确性、KV 汇总、三个预算、总耗时和特征/Router 分项耗时。
    # 计时：先清零分项计数并固定该题随机种子；同步后计时完整 run_item，不包含写盘和模型加载。
    def run(strategy,i):
        state.update(feature=strategy,feature_seconds=0.,router_seconds=0.)
        set_seed(42+i*1000);sync();start=time.perf_counter()
        with torch.inference_mode():result=methods[strategy].run_item(dataset[i])
        sync();elapsed=time.perf_counter()-start
        budgets=[int(x['selected_budget']) for x in result['router_records']]
        if len(budgets)!=3:raise ValueError('Expected three handoffs')
        return dict(dataset_index=i,strategy=strategy,**summarize_result(result),budgets=budgets,
                    elapsed_seconds=elapsed,feature_seconds=state['feature_seconds'],router_seconds=state['router_seconds'])
    # Warm up every strategy on a training question, excluding it from reported timing.
    warm=source_rows[s['train_sources'][0]]['dataset_index']
    for st in strategies:run(st,warm)
    finished=0
    for i in ids:
        order=strategies.copy();random.Random(42+i).shuffle(order)
        for st in order:
            folder=out/st;folder.mkdir(exist_ok=True);path=folder/f'sample_{i:06d}.json'
            if path.exists():
                saved=json.loads(path.read_text(encoding='utf-8'))
                if saved['dataset_index']!=i or saved['strategy']!=st:raise ValueError('Resume record mismatch')
                continue
            row=run(st,i);write(path,row)
            print(f"{st} source={i} correct={row['correct']} B={row['budgets']} seconds={row['elapsed_seconds']:.2f}",flush=True)
        finished+=1
        if a.limit and finished>=a.limit:break
    summary(a,ids,strategies)


# 功能：读取四策略逐题结果，汇总完整解题表现和共同答对题上的通信收益。
# 输入：a.output 指定实验目录，ids 为应评估题目索引，strategies 为策略顺序（前两项固定预算，后两项 Router）。
# 输出：返回 None；写 end_to_end.json、end_to_end.md 并打印表格，报告完成度、正确率、KV 均值及延迟分位数。
# 口径：全题均值包含答错题；只有全部完成后才计算共同答对集合的 KV 节省和全题平均时延差，KV 张量大小不等于实测网络流量。
def summary(a,ids,strategies):
    import numpy as np
    out=a.output/'evaluation';all_rows={}
    lines=['# 新版Router端到端开发集评估','','| 策略 | 完成题数 | 答对题数 | 正确率 | 全题平均KV(MiB) | 平均时延(s) | P50/P95(s) | 统计特征计算(s/题) | Router前向(s/题) |',
           '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for st in strategies:
        rows=[json.loads(p.read_text(encoding='utf-8')) for i in ids for p in [out/st/f'sample_{i:06d}.json'] if p.exists()]
        all_rows[st]=rows
        if not rows:continue
        # 功能：对当前策略已经完成题目的某个数值字段求平均。
        # 输入：key 为结果字典字段名；数据取自外层 rows，含答对和答错题。
        # 输出：Python float 均值，供该策略的汇总表使用。
        def avg(key):return float(np.mean([r[key] for r in rows]))
        n=len(rows);correct=sum(r['correct'] for r in rows);p50,p95=np.quantile([r['elapsed_seconds'] for r in rows],[.5,.95])
        lines.append(f"| {st} | {n}/{len(ids)} | {correct} | {correct/n:.2%} | {avg('communication_overhead'):.4f} | {avg('elapsed_seconds'):.3f} | {p50:.3f}/{p95:.3f} | {avg('feature_seconds'):.4f} | {avg('router_seconds'):.4f} |")
    complete=all(len(rows)==len(ids) for rows in all_rows.values())
    lines+=['',f'完整评估：{complete}。不完整时不得将不同题目范围的均值直接比较。',
            'KV为三次传递张量大小之和，并非实测网络流量。全题均值包含答错题。',
            '时延在GPU同步后测量，排除模型加载和逐题结果写盘，包含完整推理、在线特征获取和Router选择。',
            '统计特征分项记录hook拷贝与计算，GPU同步及其余调度开销以端到端计时为准。hidden直接使用原流程返回特征，额外特征计算计0。',
            '每题策略顺序固定随机化，各策略先用训练题热身。单次测量不代表稳定性能基准；需保持机器独占和相同运行设置。',
            '193题为曾用于选模型的开发验证集且均有Oracle成功方案，不是官方test。']
    if complete:
        lines+=['','| Router | 固定对照 | 共同答对题 | 共同答对KV节省 | 全题平均时延差(s) |','|---|---|---:|---:|---:|']
        for st in strategies[2:]:
            dynamic={r['dataset_index']:r for r in all_rows[st]}
            for baseline in strategies[:2]:
                common=[r for r in all_rows[baseline] if r['correct'] and dynamic[r['dataset_index']]['correct']]
                saving=1-sum(dynamic[r['dataset_index']]['communication_overhead'] for r in common)/sum(r['communication_overhead'] for r in common) if common else None
                change=np.mean([r['elapsed_seconds'] for r in all_rows[st]])-np.mean([r['elapsed_seconds'] for r in all_rows[baseline]])
                value='无共同答对题' if saving is None else f'{saving:.2%}'
                lines.append(f'| {st} | {baseline} | {len(common)} | {value} | {change:.3f} |')
    write(out/'end_to_end.json',dict(complete=complete,rows=all_rows))
    (out/'end_to_end.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))


# 功能：提供扩充 Router 的重新训练与完整解题评估入口。
# 输入：无显式参数；从命令行读取 train/evaluate、特征类型、数据/特征/输出目录、模型与设备。
# 输出：返回 None；验证 limit 和目录独立性、加载固定数据后，调用训练或评估阶段并生成对应产物。
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--feature',choices=['last_hidden','statistics'],default='last_hidden')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model_name',required=True)
    p.add_argument('--cache_dir',default='/root/autodl-tmp/hf_cache');p.add_argument('--local_files_only',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.limit is not None and a.limit<=0:p.error('limit must be positive')
    for original in (a.data,a.features):
        x=original.resolve();y=a.output.resolve()
        if x==y or x in y.parents or y in x.parents:p.error('Output must be separate')
    d,s=load_data(a);a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='train':train(a,d,s)
    else:evaluate(a,d,s)


if __name__=='__main__':main()
