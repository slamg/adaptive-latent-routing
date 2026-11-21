# 文件作用：组织新旧合计1000道题的固定预算与 Oracle 对照。
# 主要职责：检查可复用结果，调用公共比较模块补跑缺失B32任务并生成报告。

"""补齐同一1000题的B32；只写新目录，保留全部原始实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import oracle_comparison_core as core

ROOT = Path('/root/autodl-tmp')

# 功能：按块读取文件并计算内容摘要，用于识别数据文件或代码文件的版本。
# 输入：p 为可读取的文件路径；每次读取最多 1 MiB 的二进制内容。
# 输出：返回 SHA-256 十六进制字符串；只读文件，不修改文件内容。
def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

# 功能：核对同一题的两份推理结果，避免把答案或 KV 量不一致的记录混在一起。
# 输入：a、b 为结果字典，i 为报错时显示的题目索引；比较 correct、prediction 和 communication_overhead。
# 输出：一致时返回 None；答案/正确性不同，或 KV 量超出相对 1e-6、绝对 1e-5 的容差时抛出 ValueError。
# 说明：虽报错文字写 B32，本函数也被用于核对复用报告中的 B8 结果；它不会重新执行推理。
def agree(a,b,i):
    import math
    if a['correct'] != b['correct'] or a.get('prediction') != b.get('prediction') or not math.isclose(float(a['communication_overhead']),float(b['communication_overhead']),rel_tol=1e-6,abs_tol=1e-5):
        raise ValueError(f'题目{i}的重复B32结果不一致，停止混合；请先复核')

# 功能：整合原 500 题和新增 500 题，检查并复用完整搜索、旧报告及端到端 B32 结果，只补跑仍缺失的题。
# 输入：命令行提供 plan/run/report、limit、confirm_settings；数据路径及模型/生成配置由函数内的 ROOT 和固定参数确定。
# 输出：正常结束返回 None；写 provenance.json、coverage.json，run 可写逐题补跑记录，覆盖齐全后生成 comparison.md/json。
# 衔接：核对新旧题目互斥、重复结果一致及可选端到端记录的设置/数据/代码摘要；已有 provenance 不一致时停止。
# 控制：plan 也会保存来源和覆盖信息；report 遇到缺题以状态码 1 结束，limit 仅控制本次补跑数量，不缩小最终统计范围。
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['plan','run','report'])
    p.add_argument('--limit',type=int)
    p.add_argument('--confirm_settings',action='store_true')
    a=p.parse_args()
    if a.limit is not None and a.limit<=0: p.error('limit must be positive')
    a.model_name=str(ROOT/'hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c')
    a.cache_dir=str(ROOT/'hf_cache');a.local_files_only=True;a.device='cuda'
    a.prompt='sequential';a.latent_steps=20;a.max_new_tokens=2048;a.pca_rank=8;a.sink_size=4;a.seed=42;a.think=False;a.latent_space_realign=False
    a.output_folder=ROOT/'router_data/gsm8k_oracle_comparison_1000'
    ids=[];screen={};fixed={};oracle={};origins={}
    for name,s,o in [('old','gsm8k_screen_500','gsm8k_joint_selected_50'),('new','gsm8k_expansion_500_v1/screen','gsm8k_expansion_500_v1/exhaustive')]:
        ii,ss,ff,oo,mm=core.audit(ROOT/'router_data'/s,ROOT/'router_data'/o,500)
        if set(ids)&set(ii):raise ValueError('新旧题目重复')
        if any(x['model_name']!=a.model_name for x in mm.values()):raise ValueError('模型不一致')
        ids+=ii;screen.update(ss);fixed.update(ff);oracle.update(oo)
        origins.update({i:'exhaustive' for i in ff})
    # 原500题完整报告包括补跑结果；按题目内容和生成参数核对。
    oldpath=ROOT/'router_data/gsm8k_fixed32_comparison_500/comparison.json'
    old=json.loads(oldpath.read_text(encoding='utf-8'))
    if old['settings']!=core.settings(a) or set(old['question_ids'])!=set(ids[:500]):raise ValueError('旧500题报告参数或范围不一致')
    seen=set()
    for r in old['per_question']:
        i=int(r['dataset_index'])
        if i not in set(ids[:500]) or i in seen:raise ValueError('旧报告逐题索引异常')
        seen.add(i);agree(screen[i]['b8'],r['b8'],i)
        b=core.result_check(r['b32'])
        if i in fixed:agree(fixed[i],b,i)
        else:fixed[i]=dict(b,schedule=[32]*3,cost=96);origins[i]='old_500_report'
    if seen!=set(ids[:500]):raise ValueError('旧报告缺题')
    # 可选复用端到端B32结果。manifest及当前代码必须一致；否则停止而非静默混合。
    ev=ROOT/'router_training/router_v2/evaluation'
    if (ev/'manifest.json').exists():
        m=json.loads((ev/'manifest.json').read_text(encoding='utf-8'))
        if m['settings']!=core.settings(a) or m['model_name']!=a.model_name:raise ValueError('端到端参数不一致')
        if m['data_sha256']!=digest(ROOT/'router_data/gsm8k_merged_v2/merged_dataset.pt'):raise ValueError('端到端数据版本不一致')
        for f,h in m['code_sha256'].items():
            if digest(Path(__file__).parent/f)!=h:raise ValueError(f'端到端代码版本变化: {f}')
        for path in sorted((ev/'fixed32').glob('sample_*.json')):
            r=json.loads(path.read_text(encoding='utf-8'));i=int(r['dataset_index'])
            if i not in screen or i not in m['ids'] or r['strategy']!='fixed32' or r['budgets']!=[32]*3:raise ValueError('端到端记录身份/预算异常')
            if str(r['gold'])!=str(screen[i]['gold']):raise ValueError('端到端标准答案不一致')
            core.result_check(r)
            if i in fixed:agree(fixed[i],r,i)
            else:fixed[i]=dict(r,schedule=[32]*3,cost=96);origins[i]='end_to_end'
    # 冻结来源、原始结果摘要和运行代码，避免断点续跑混用版本。
    data_digest=hashlib.sha256(json.dumps([ids,screen,fixed,oracle],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    files=['complete_oracle_1000.py','oracle_comparison_core.py','models.py','methods/latent_mas.py','compression_methods/HOBF.py']
    plan={'ids':ids,'settings':core.settings(a),'data_digest':data_digest,'code_sha256':{f:digest(Path(__file__).parent/f) for f in files},'reuse_origins':{str(i):v for i,v in origins.items()}}
    a.output_folder.mkdir(parents=True,exist_ok=True)
    pp=a.output_folder/'provenance.json'
    if pp.exists() and json.loads(pp.read_text(encoding='utf-8'))!=plan:raise ValueError('来源或设置已变化，不能混入已有补跑目录')
    if not pp.exists():core.save_json(pp,plan)
    completed=core.read_completed(a,ids,screen,fixed)
    missing=[i for i in ids if i not in completed]
    summary={'questions':len(ids),'b8_correct':sum(r['b8']['correct'] for r in screen.values()),'oracle_correct':sum(r is not None for r in oracle.values()),'reused_b32':len(fixed),'new_completed_b32':len(completed)-len(fixed),'missing_b32':len(missing),'missing_ids':missing}
    core.save_json(a.output_folder/'coverage.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if a.mode=='plan':return
    if missing and a.mode=='run':
        if not a.confirm_settings:p.error('核对设置后加 --confirm_settings；latent_steps=20,max_new_tokens=2048,seed=42,sequential,think=False')
        (a.output_folder/'samples').mkdir(exist_ok=True)
        core.run_missing(a,missing if a.limit is None else missing[:a.limit],screen)
        completed=core.read_completed(a,ids,screen,fixed)
    if len(completed)!=len(ids):
        print(f'尚缺{len(ids)-len(completed)}题；完整统计尚未生成。')
        if a.mode=='report':raise SystemExit(1)
        return
    core.report(a,ids,screen,completed,oracle)
    print('Saved:',a.output_folder/'comparison.md')

if __name__=='__main__':main()
