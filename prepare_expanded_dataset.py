# 文件作用：合并新旧标签数据并建立扩充后的固定训练/验证划分。
# 主要职责：检查来源覆盖、保留旧题归属、分配新增题目，保存合并数据、来源划分及报告。

"""Merge two completed collection batches and freeze a source-level split."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from audit_router_sources import summarize


# 功能：按独立题目为新增数据分配训练/验证归属，并保留每层的题目数量统计。
# 输入：records 为新增状态记录，fraction 为期望验证比例，seed 为分层打乱种子。
# 输出：返回（排序后的训练来源 ID 列表、验证来源 ID 列表、各分层的题数统计列表）。
# 流程：按 easy/hard 和是否含非 B8 状态分层，同题全部状态归属同一侧；单题分层只分给训练。
# 注意：此处以掩码第 0 项代表 B8，依赖调用方固定候选顺序为 (8,16,32,64)；小分层比例会受取整影响。
def assign_new_sources(records, fraction, seed):
    # 按独立题目分层：是否贡献非B8标签，以及原筛选easy/hard类别。
    # 不参考任何Router预测结果。不同状态不能被拆到不同集合。
    groups = defaultdict(list)
    by_source = defaultdict(list)
    for r in records:
        by_source[r['source_id']].append(r)
    for source, rows in sorted(by_source.items()):
        positive = any(not r['optimal_budget_mask'][0] for r in rows)
        groups[(rows[0]['source_kind'], positive)].append(source)
    rng = random.Random(seed)
    train, val, strata = [], [], []
    for key, sources in sorted(groups.items()):
        rng.shuffle(sources)
        nval = min(len(sources)-1, max(1, round(len(sources)*fraction))) if len(sources)>1 else 0
        val.extend(sources[:nval])
        train.extend(sources[nval:])
        strata.append(dict(source_kind=key[0], has_non_b8=key[1], sources=len(sources),
                           train_sources=len(sources)-nval, validation_sources=nval))
    return sorted(train), sorted(val), strata


# 功能：核对新旧有效来源是否互斥、原划分是否完整，以及合并状态是否重复。
# 输入：old_records/new_records 为新旧状态列表，old_split 含 train_sources 与 validation_sources。
# 输出：无返回值（None）；来源重叠、旧划分重复或覆盖不符、相同题目/前缀重复时抛出 ValueError。
# 注意：状态去重键是 (source_id, prefix)，同一道题的不同预算前缀可以保留为不同训练样本。
def validate_ids(old_records, new_records, old_split):
    old_ids = {r['source_id'] for r in old_records}
    new_ids = {r['source_id'] for r in new_records}
    if old_ids & new_ids:
        raise ValueError('Old/new labelled source overlap')
    train, val = old_split['train_sources'], old_split['validation_sources']
    if len(train)!=len(set(train)) or len(val)!=len(set(val)):
        raise ValueError('Duplicate sources in original split')
    if set(train)&set(val) or set(train)|set(val)!=old_ids:
        raise ValueError('Original saved split does not exactly cover original data')
    seen = set()
    for r in old_records + new_records:
        key = (r['source_id'], tuple(r['prefix']))
        if key in seen:
            raise ValueError('Duplicate source/prefix state')
        seen.add(key)


# 功能：读取一批筛选与完整搜索结果，核查题号覆盖后整理训练状态，并计算输入内容指纹。
# 输入：screen 与 exhaustive 为 Path 目录，分别含 selection.json、逐题分片及可选完整搜索题号文件。
# 输出：依次返回（标准化状态记录、构建统计、该批全部已选题号集合、SHA-256 指纹字符串）。
# 流程：检查全部筛选题已保存、失败题及计划搜索题已有完整搜索分片，再调用 build_router_examples。
# 注意：已选题号集合包括没有成功标签的题；本函数不重新执行模型，也不重新推导每个分片中的 Oracle。
def read_complete_batch(screen, exhaustive):
    import torch
    from router_training_data import build_router_examples
    selection = json.loads((screen/'selection.json').read_text(encoding='utf-8'))
    if selection['task']!='gsm8k' or selection['split']!='train':
        raise ValueError('Expected GSM8K train')
    ids = set(selection['selected_indices'])
    if len(ids)!=len(selection['selected_indices']):
        raise ValueError('Duplicate selection indices')
    screen_ids, failed = set(), set()
    full_ids = set()
    digest = hashlib.sha256()
    for folder, full in ((screen, False), (exhaustive, True)):
        for path in sorted((folder/'samples').glob('sample_*.pt')):
            # 保留输入内容指纹；相同输出目录不能悄悄换一批数据。
            digest.update(str(path.resolve()).encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            item = torch.load(path, map_location='cpu')
            index = int(item['sample_id' if full else 'dataset_index'])
            target = full_ids if full else screen_ids
            if index in target:
                raise ValueError('Duplicate shard id')
            target.add(index)
            if not full and not item['minimum_budget_correct']:
                failed.add(index)
    if screen_ids!=ids or not failed<=full_ids or not full_ids<=ids:
        raise ValueError('Incomplete screen/exhaustive batch or unexpected source IDs')
    index_file = screen/'full_search_indices.txt'
    if index_file.exists():
        planned = {int(x) for x in index_file.read_text().split()}
        # 旧批次后来由50题扩到76题，允许包含原始计划外的补齐题。
        if not planned<=full_ids:
            raise ValueError('Missing planned full-search sources')
    records, report = build_router_examples(screen, exhaustive, view='primary_reserve',
                                            budget_choices=(8,16,32,64))
    return records, report, ids, digest.hexdigest()


# 功能：合并完成采集的新旧批次，保留旧题归属，为新增来源固定划分并生成合并报告。
# 输入：命令行指定两批筛选/搜索目录、旧 source_coverage.json、输出路径以及新增划分比例和种子。
# 输出：返回 None；保存 merged_dataset.pt、source_split.json、merge_report.json 与 merge_report.md。
# 流程：读取与校验两批数据 → 复核原来源划分及统计 → 划分新题 → 合并记录和来源归属 → 输出统计。
# 注意：以题目为单位保持训练/验证互斥，不能把同题的多个状态随机拆到两侧。
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old_screen', required=True)
    p.add_argument('--old_exhaustive', required=True)
    p.add_argument('--new_screen', required=True)
    p.add_argument('--new_exhaustive', required=True)
    p.add_argument('--old_coverage', required=True, help='Previously generated source_coverage.json')
    p.add_argument('--output_folder', required=True)
    p.add_argument('--new_split_seed', type=int, default=20261110)
    p.add_argument('--validation_fraction', type=float, default=0.2)
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError('Invalid validation fraction')
    out = Path(args.output_folder).resolve()
    inputs = [Path(getattr(args,k)).resolve() for k in ('old_screen','old_exhaustive','new_screen','new_exhaustive')]
    for root in inputs:
        if out==root or out in root.parents or root in out.parents:
            raise ValueError('Output must be separate from original data')
    print('Reading and checking old batch...', flush=True)
    old, old_report, old_ids, old_hash = read_complete_batch(*inputs[:2])
    print('Reading and checking new batch...', flush=True)
    new, new_report, new_ids, new_hash = read_complete_batch(*inputs[2:])
    if old_ids & new_ids:
        raise ValueError('Old and new screening questions overlap')
    coverage = json.loads(Path(args.old_coverage).read_text(encoding='utf-8-sig'))
    old_split = coverage['source_split']
    validate_ids(old, new, old_split)
    for part, key in [('train','train_sources'),('validation','validation_sources')]:
        s = summarize([r for r in old if r['source_id'] in set(old_split[key])])
        if s != coverage['partitions'][part]:
            raise ValueError('Original data no longer matches saved coverage: '+part)
    nt, nv, strata = assign_new_sources(new, args.validation_fraction, args.new_split_seed)
    split = dict(train_sources=sorted(old_split['train_sources']+nt),
                 validation_sources=sorted(old_split['validation_sources']+nv),
                 old_train_sources=old_split['train_sources'], old_validation_sources=old_split['validation_sources'],
                 new_train_sources=nt, new_validation_sources=nv,
                 new_split_seed=args.new_split_seed, validation_fraction=args.validation_fraction,
                 policy='Preserve old source assignments; stratify new sources by screening kind and presence of non-B8 optimal labels.')
    records = old+new
    train_ids, val_ids = set(split['train_sources']), set(split['validation_sources'])
    if train_ids & val_ids or train_ids|val_ids != {r['source_id'] for r in records}:
        raise ValueError('Merged split leakage or missing source')
    manifest = dict(arguments=vars(args), old_input_sha256=old_hash, new_input_sha256=new_hash,
                    old_split_sha256=hashlib.sha256(json.dumps(old_split,sort_keys=True).encode()).hexdigest())
    report_path = out/'merge_report.json'
    if out.exists() and any(out.iterdir()):
        if not report_path.exists() or json.loads(report_path.read_text(encoding='utf-8'))['manifest']!=manifest:
            raise ValueError('Output exists with different/unfinished settings; use a new output folder')
    groups = {'old_all':old, 'new_all':new,
              'old_train':[r for r in old if r['source_id'] in train_ids],
              'old_validation':[r for r in old if r['source_id'] in val_ids],
              'new_train':[r for r in new if r['source_id'] in train_ids],
              'new_validation':[r for r in new if r['source_id'] in val_ids],
              'merged_train':[r for r in records if r['source_id'] in train_ids],
              'merged_validation':[r for r in records if r['source_id'] in val_ids]}
    stats = {name:summarize(rows) for name,rows in groups.items()}
    report = dict(manifest=manifest, old_build=old_report, new_build=new_report,
                  new_source_strata=strata, coverage=stats, source_split=split)
    labels = {'old_all':'旧批次合计','new_all':'新增批次合计','old_train':'旧训练','old_validation':'旧验证',
              'new_train':'新增训练','new_validation':'新增验证','merged_train':'合并训练','merged_validation':'合并验证'}
    lines = ['# 新旧数据整理与固定划分', '',
             '| 范围 | 独立题目 | 状态 | 非B8独立题目 | 非B8状态 |', '|---|---:|---:|---:|---:|']
    for name,s in stats.items():
        lines.append(f"| {labels[name]} | {s['total_sources']} | {s['total_states']} | {s['non_b8_sources']} | {s['non_b8_states']} |")
    for name in ('merged_train','merged_validation'):
        s = stats[name]
        lines += ['', '## '+labels[name], '', '| 通信位置(0/1/2) | 非B8来源题 | 非B8状态 |','|---|---:|---:|']
        for h,v in s['handoffs'].items():
            lines.append(f"| {h} | {v['non_b8_sources']} | {v['non_b8_states']} |")
        lines += ['', '| 代表标签 | 独立来源题 | 状态 |','|---|---:|---:|']
        for b,v in s['budgets'].items():
            lines.append(f"| {b} | {v['canonical_sources']} | {v['canonical_states']} |")
        lines += ['', '非B8状态前1/3/5道来源题占比：'+ ' / '.join('无' if s['top_source_shares'][k] is None else f"{s['top_source_shares'][k]:.2%}" for k in ('1','3','5'))]
    lines += ['', '旧题归属完全保留；新增题按筛选easy/hard、是否有非B8状态分层，固定种子抽取约20%用于验证。单题分层组仅分配到训练。',
              '此划分是GSM8K train内部的开发验证划分，不是官方test。各预算来源题可重叠。',
              '非B8指最优集合不包含B8。主轨迹与备用状态复用原数据构建函数去重，无成功后续状态排除。',
              'merged_dataset.pt保存原最后一步hidden及标签；新题的最近5步均值、18维特征仍需后续补采。',
              '后续实验必须读取本次source_split.json，不能对合并数据重新随机划分。']
    out.mkdir(parents=True, exist_ok=True)
    import torch
    destination = out/'merged_dataset.pt'
    temp = out/'merged_dataset.pt.tmp'
    torch.save(dict(schema_version=1,budget_choices=(8,16,32,64),view='primary_reserve',
                    records=records,source_split=split,manifest=manifest),temp)
    temp.replace(destination)
    for name,obj in [('source_split.json',split),('merge_report.json',report)]:
        path=out/name;tmp=out/(name+'.tmp')
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    (out/'merge_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print('Saved:',out)


if __name__=='__main__':
    main()
