# 文件作用：盘点 Router 数据中的独立题目来源及历史状态覆盖情况。
# 主要职责：统计非B8来源、通信阶段分布和状态集中程度，输出覆盖报告。

"""Read-only source coverage audit. Run beside router_training_data.py."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# 功能：分别按独立题目和历史状态统计标签覆盖，检查非 B8 状态来自多少题及其集中程度。
# 输入：records 为统一状态记录，choices 为候选预算顺序，用于定位最优掩码中的 B8 项。
# 输出：统计字典，含题数、状态数、非 B8 数量、通信阶段/预算分布、逐题明细与来源集中度。
# 注意：非 B8 指最优预算集合不包含 B8，不等于整题使用 (8,8,8) 必然答错；重复题目/前缀或无效掩码会报错。
def summarize(records, choices=(8, 16, 32, 64)):
    """Non-B8 means B8 is absent from the entire optimal action set."""
    b8 = choices.index(8)
    groups = defaultdict(list)
    seen = set()
    for r in records:
        key = (r['source_id'], tuple(r['prefix']))
        if key in seen:
            raise ValueError('Duplicate state: %r' % (key,))
        seen.add(key)
        mask = r['optimal_budget_mask']
        if len(mask) != len(choices) or not any(mask):
            raise ValueError('Invalid optimal mask')
        groups[r['source_id']].append(r)
    rows = []
    for source, items in sorted(groups.items()):
        positive = [r for r in items if not r['optimal_budget_mask'][b8]]
        rows.append(dict(source_id=source, source_kind=items[0]['source_kind'],
                         total_states=len(items), non_b8_states=len(positive),
                         non_b8_by_handoff={str(h): sum(r['handoff_index'] == h for r in positive) for h in range(3)},
                         canonical_budget_counts={str(b): sum(r['target_budget'] == b for r in items) for b in choices},
                         non_b8_prefixes=[list(r['prefix']) for r in positive]))
    positive_rows = sorted((r for r in rows if r['non_b8_states']),
                           key=lambda r: (-r['non_b8_states'], r['source_id']))
    total = sum(r['non_b8_states'] for r in rows)
    handoffs = {}
    for h in range(3):
        subset = [r for r in records if r['handoff_index'] == h]
        pos = [r for r in subset if not r['optimal_budget_mask'][b8]]
        handoffs[str(h)] = dict(total_states=len(subset), non_b8_states=len(pos),
                               non_b8_sources=len({r['source_id'] for r in pos}))
    budgets = {}
    for i, b in enumerate(choices):
        canonical = [r for r in records if r['target_budget'] == b]
        optimal = [r for r in records if r['optimal_budget_mask'][i]]
        budgets[str(b)] = dict(canonical_states=len(canonical),
                              canonical_sources=len({r['source_id'] for r in canonical}),
                              optimal_set_states=len(optimal),
                              optimal_set_sources=len({r['source_id'] for r in optimal}))
    return dict(total_sources=len(rows), total_states=len(records), non_b8_states=total,
                non_b8_sources=len(positive_rows),
                non_b8_source_kinds=dict(Counter(r['source_kind'] for r in positive_rows)),
                top_source_shares={str(k): sum(r['non_b8_states'] for r in positive_rows[:k]) / total if total else None for k in (1, 3, 5, 10)},
                handoffs=handoffs, budgets=budgets, sources=rows,
                ranked_non_b8_sources=positive_rows)


# 功能：将来源覆盖统计转换为可阅读的 Markdown 报告。
# 输入：report 为报告字典，其中 partitions 含训练和验证两部分的 summarize 统计结果。
# 输出：包含来源覆盖、通信阶段、预算标签和来源集中度等表格的 Markdown 字符串。
# 注意：此函数只组织文本，不写文件，也不根据数量自动判断数据是否足够。
def markdown(report):
    lines = ['# 非B8独立题目覆盖核查', '',
             '非B8：Oracle并列最优动作集合不包含B=8；不是“选8一定答错”。',
             '统计基于有效、去重后的状态。来源题可以同时贡献B8和非B8状态。', '',
             '| 划分 | 独立题数 | 状态数 | 非B8状态数 | 非B8来源题数 |',
             '|---|---:|---:|---:|---:|']
    for part, s in report['partitions'].items():
        lines.append(f"| {part} | {s['total_sources']} | {s['total_states']} | {s['non_b8_states']} | {s['non_b8_sources']} |")
    for part, s in report['partitions'].items():
        lines += ['', f'## {part}', '', '### 非B8状态集中程度', '',
                  '| 贡献最多的题目 | 占该划分全部非B8状态 |', '|---|---:|']
        for k, value in s['top_source_shares'].items():
            share = '无非B8状态' if value is None else f'{value:.2%}'
            lines.append(f'| 前{k}题（不足则取全部） | {share} |')
        lines += ['', '### 按通信位置', '', '| handoff（从0计） | 全部状态 | 非B8状态 | 非B8独立来源题 |', '|---|---:|---:|---:|']
        for h, v in s['handoffs'].items():
            lines.append(f"| {h} | {v['total_states']} | {v['non_b8_states']} | {v['non_b8_sources']} |")
        lines += ['', '### 按预算', '',
                  '代表标签是并列最优中选定的标签；最优集合覆盖保留所有并列动作，各预算间可能重叠。', '',
                  '| B | 代表标签状态 | 代表标签来源题 | 最优集合含B的状态 | 最优集合含B的来源题 |', '|---|---:|---:|---:|---:|']
        for b, v in s['budgets'].items():
            lines.append(f"| {b} | {v['canonical_states']} | {v['canonical_sources']} | {v['optimal_set_states']} | {v['optimal_set_sources']} |")
        lines += ['', '### 所有非B8来源题（按状态数排序）', '',
                  '| 题目ID | 筛选类别 | 全部状态 | 非B8状态 | handoff 0 / 1 / 2 |', '|---|---|---:|---:|---|']
        for r in s['ranked_non_b8_sources']:
            hs = ' / '.join(str(r['non_b8_by_handoff'][str(h)]) for h in range(3))
            lines.append(f"| {r['source_id']} | {r['source_kind']} | {r['total_states']} | {r['non_b8_states']} | {hs} |")
    lines += ['', 'easy/hard沿用固定(8,8,8)筛选结果，不等同于某个状态的B8/非B8标签。',
              '状态条数受已采集前缀范围影响：完整搜索题最多21个前缀，未完整搜索的简单题通常只有3个。',
              '本报告不自动判断样本是否足够，也不训练模型、补采题目或改变原始划分。']
    return '\n'.join(lines) + '\n'


# 功能：读取生成数据及来源划分，审计训练/验证的独立来源与状态覆盖并输出报告。
# 输入：命令行给出筛选/搜索目录、输出路径及可选 source_split；未指定划分文件时按比例和种子生成来源划分。
# 输出：返回 None；保存 source_coverage.json 与 source_coverage.md，并打印报告内容。
# 流程：整理 primary_reserve 状态 → 验证划分无重复、互斥且完整覆盖 → 分别统计训练/验证 → 保存结果。
# 注意：脚本只审计数据，不训练 Router；与预期旧批次数量不符时会在报告中记录警告。
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen_folder', required=True)
    p.add_argument('--exhaustive_folder', required=True)
    p.add_argument('--output_folder', required=True)
    p.add_argument('--source_split', help='原训练source_split.json；提供时严格使用该划分')
    p.add_argument('--split_seed', type=int, default=20261001)
    p.add_argument('--validation_fraction', type=float, default=0.2)
    args = p.parse_args()
    from router_training_data import build_router_examples, split_source_ids
    choices = (8, 16, 32, 64)
    records, data = build_router_examples(args.screen_folder, args.exhaustive_folder,
                                          view='primary_reserve', budget_choices=choices)
    if args.source_split:
        split = json.loads(Path(args.source_split).read_text(encoding='utf-8-sig'))
        train, val = split['train_sources'], split['validation_sources']
        method = 'saved_source_split'
    else:
        train, val = split_source_ids(records, validation_fraction=args.validation_fraction,
                                     seed=args.split_seed)
        method = 'reconstructed_using_original_function'
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError('Duplicate sources in split')
    train, val = set(train), set(val)
    if train & val or train | val != {r['source_id'] for r in records}:
        raise ValueError('Split overlaps or does not exactly cover current data')
    report = dict(arguments=vars(args), split_method=method, data=data,
                  source_split=dict(train_sources=sorted(train), validation_sources=sorted(val)),
                  partitions={name: summarize([r for r in records if r['source_id'] in ids], choices)
                              for name, ids in [('train', train), ('validation', val)]})
    actual = [(s['total_sources'], s['total_states']) for s in report['partitions'].values()]
    report['matches_previous_sizes_and_validation_non_b8'] = (
        actual == [(386, 1781), (97, 499)] and report['partitions']['validation']['non_b8_states'] == 48)
    out = Path(args.output_folder)
    # Do not put reports in either original dataset directory.
    for original in (args.screen_folder, args.exhaustive_folder):
        root = Path(original).resolve()
        if out.resolve() == root or root in out.resolve().parents:
            raise ValueError('Choose an output directory outside original data folders')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source_coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    body = markdown(report)
    if not report['matches_previous_sizes_and_validation_non_b8']:
        body = '**注意：当前规模或验证非B8数与历史386/1781、97/499、48不一致，请先核查输入及划分。**\n\n' + body
    (out / 'source_coverage.md').write_text(body, encoding='utf-8')
    print(body)
    print('Saved:', out / 'source_coverage.json')


if __name__ == '__main__':
    main()
