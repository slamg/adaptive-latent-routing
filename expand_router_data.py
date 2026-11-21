# 文件作用：调度新增题目的筛选与完整预算搜索。
# 主要职责：排除旧题、保存扩充计划并调用已有数据生成程序，组织新一批题目采集。

"""Collect a separate batch of unseen GSM8K train sources using existing scripts."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


# 功能：从原数据集索引范围中排除旧题，再无放回抽取新增题目。
# 输入：size 为数据集题数，excluded 为旧题号集合，count 为新增数量，seed 为抽样种子。
# 输出：随机抽取的新题号列表；旧题越界或新增数量不合理时会报错。
# 注意：排除的是原始题号，抽样使用局部随机数生成器，不运行模型。
def choose_new(size, excluded, count, seed):
    excluded = set(excluded)
    if any(i < 0 or i >= size for i in excluded):
        raise ValueError('Excluded index outside dataset')
    pool = [i for i in range(size) if i not in excluded]
    if not 0 < count <= len(pool):
        raise ValueError('Invalid new source count')
    return random.Random(seed).sample(pool, count)


# 功能：通过临时文件写入并替换目标文件，保存 JSON 配置或计划。
# 输入：path 为 Path 目标路径，payload 为能够 JSON 序列化的数据。
# 输出：无返回值（None）；先写同目录 .tmp 文件，再替换 path。
# 注意：父目录应由调用方提前创建；替换后的内容是此次 payload 的完整序列化结果。
def save(path, payload):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


# 功能：组织新增题目的计划、最小预算筛选和完整搜索，复用已有数据生成入口。
# 输入：命令行给出 stage、旧筛选目录、新题数量与种子、模型和输出根目录；读取旧 selection.json。
# 输出：返回 None；保存 expansion_plan.json 和新筛选 selection.json，并由子程序产生筛选或搜索数据。
# 流程：排除旧题并固定新题计划 → 按 stage 调用 screen_router_candidates.py 和 collect_router_data.py。
# 注意：扩充流程设置失败题全部搜索，成功题按 easy_full_count 抽查；plan 阶段只生成计划。
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['plan', 'screen', 'full', 'all'], nargs='?', default='all')
    p.add_argument('--previous_screen', nargs='+', required=True,
                   help='All previous screening folders, including earlier expansion batches')
    p.add_argument('--output_root', required=True)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--sample_seed', type=int, default=20261108)
    p.add_argument('--easy_full_count', type=int, default=20)
    p.add_argument('--model_name', required=True)
    p.add_argument('--cache_dir', default='/root/autodl-tmp/hf_cache')
    args = p.parse_args()
    if args.easy_full_count < 0:
        raise ValueError('easy_full_count must be nonnegative')
    project = Path(__file__).resolve().parent
    out = Path(args.output_root).resolve()
    excluded = set()
    size = None
    for directory in args.previous_screen:
        folder = Path(directory).resolve()
        if out == folder or folder in out.parents or out in folder.parents:
            raise ValueError('Output must be separate from previous data')
        selection = json.loads((folder / 'selection.json').read_text(encoding='utf-8'))
        if selection['task'] != 'gsm8k' or selection['split'] != 'train':
            raise ValueError('Only GSM8K train expansion is allowed')
        if size is not None and size != selection['dataset_size']:
            raise ValueError('Dataset size mismatch')
        size = selection['dataset_size']
        ids = selection['selected_indices']
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate previous source indices')
        excluded.update(ids)
    selected = choose_new(size, excluded, args.count, args.sample_seed)
    screen = out / 'screen'
    full = out / 'exhaustive'
    common = ['--model_name', args.model_name, '--cache_dir', args.cache_dir,
              '--local_files_only', '--task', 'gsm8k', '--split', 'train',
              '--candidate_budgets', '8', '16', '32', '64', '--reference_budget', '32',
              '--prompt', 'sequential', '--latent_steps', '20', '--max_new_tokens', '2048',
              '--pca_rank', '8', '--sink_size', '4', '--seed', '42', '--device', 'cuda']
    # Freeze code and settings: resuming must never silently combine different runs.
    names = ['screen_router_candidates.py', 'collect_router_data.py', 'router_data_selection.py',
             'models.py', 'methods/latent_mas.py', 'compression_methods/HOBF.py', 'budget_router.py']
    hashes = {n: hashlib.sha256((project / n).read_bytes()).hexdigest() for n in names}
    plan = dict(schema_version=1, task='gsm8k', split='train', dataset_size=size,
                excluded_indices=sorted(excluded), selected_indices=selected,
                sample_seed=args.sample_seed, easy_full_count=args.easy_full_count,
                inference_arguments=common, code_sha256=hashes)
    plan_path = out / 'expansion_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/config/code changed. Restore original settings or use a new output root.')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Refusing to reuse nonempty output without an expansion plan')
        out.mkdir(parents=True, exist_ok=True)
        save(plan_path, plan)
    screen.mkdir(exist_ok=True)
    selection = dict(schema_version=1, task='gsm8k', split='train', dataset_size=size,
                     sample_seed=args.sample_seed, selected_indices=selected)
    selection_path = screen / 'selection.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding='utf-8')) != selection:
            raise ValueError('Screen selection differs from frozen plan')
    else:
        save(selection_path, selection)
    print(f'Excluded {len(excluded)} previous sources; selected {len(selected)} new train sources.', flush=True)
    print('Plan:', plan_path, flush=True)
    if args.stage == 'plan':
        return
    if args.stage in ('screen', 'all'):
        subprocess.run([sys.executable, '-u', str(project / 'screen_router_candidates.py'),
                        *common, '--max_samples', str(args.count), '--sample_seed', str(args.sample_seed),
                        '--hard_full_count', '-1', '--easy_full_count', str(args.easy_full_count),
                        '--output_folder', str(screen), '--resume'], check=True, cwd=project)
    if args.stage in ('full', 'all'):
        summary = json.loads((screen / 'summary.json').read_text(encoding='utf-8'))
        if summary['screened_samples'] != args.count:
            raise ValueError('Screening incomplete')
        print('Screening summary:', json.dumps(summary, ensure_ascii=False), flush=True)
        indices = [int(x) for x in (screen / 'full_search_indices.txt').read_text().split()]
        if len(indices) != len(set(indices)) or not set(indices) <= set(selected):
            raise ValueError('Full-search indices do not match frozen source pool')
        if len(indices) != summary['full_search_total']:
            raise ValueError('Full-search count mismatch')
        if not indices:
            print('No sources selected for full search.')
            return
        subprocess.run([sys.executable, '-u', str(project / 'collect_router_data.py'),
                        *common, '--sample_indices_file', str(screen / 'full_search_indices.txt'),
                        '--max_samples', '-1', '--output_folder', str(full), '--resume'],
                       check=True, cwd=project)
        print('Expansion completed. Original data and original train/validation split were not changed.', flush=True)


if __name__ == '__main__':
    main()
