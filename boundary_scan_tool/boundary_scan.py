from __future__ import annotations

import argparse
import csv
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extract import canonical_dataset_name, extract_sample_features, load_sample_for_plot
from ranking import build_category_topk, build_dataset_summary, build_dataset_topk, diverse_rerank
from report_builder import add_reason_column, build_reports
from scoring import CANDIDATE_LABELS, compute_scores
from seed_match import SeedBoundaryIndex
from visualization import build_plot_path, save_sample_plot


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def collect_sample_paths(input_dir: str | Path, dataset_dirs: Iterable[str], recursive: bool = True) -> List[Tuple[str, Path]]:
    input_root = Path(input_dir)
    collected: List[Tuple[str, Path]] = []
    for dataset_dir in dataset_dirs:
        dataset_root = input_root / dataset_dir
        if not dataset_root.exists():
            print(f'[warn] dataset directory not found: {dataset_root}')
            continue
        pattern = '**/*.npy' if recursive else '*.npy'
        for path in sorted(dataset_root.glob(pattern)):
            if path.is_file():
                collected.append((dataset_dir, path))
    return collected


def save_errors(errors: List[Dict[str, Any]], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / 'errors.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['file', 'dataset_name', 'error']
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    return output_path


def reset_output_dir(output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        return
    for child in output_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _attach_plot_paths(df: pd.DataFrame, plot_map: dict[str, Path], output_dir: Path) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        result['plot_rel_path'] = []
        result['plot_abs_path'] = []
        return result
    result = df.copy()
    rel_paths = []
    abs_paths = []
    for file_rel in result['file'].astype(str):
        plot_path = plot_map.get(file_rel)
        rel_paths.append(plot_path.relative_to(output_dir).as_posix() if plot_path else '')
        abs_paths.append(str(plot_path) if plot_path else '')
    result['plot_rel_path'] = rel_paths
    result['plot_abs_path'] = abs_paths
    return result


def _merge_plot_paths(all_df: pd.DataFrame, plot_map: dict[str, Path], output_dir: Path) -> pd.DataFrame:
    plot_rel = []
    plot_abs = []
    for file_rel in all_df['file'].astype(str):
        plot_path = plot_map.get(file_rel)
        plot_rel.append(plot_path.relative_to(output_dir).as_posix() if plot_path else '')
        plot_abs.append(str(plot_path) if plot_path else '')
    result = all_df.copy()
    result['plot_rel_path'] = plot_rel
    result['plot_abs_path'] = plot_abs
    return result


def _rows_for_plotting(*frames: pd.DataFrame) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    merged = pd.concat(valid_frames, axis=0, ignore_index=True)
    return merged.drop_duplicates(subset=['file']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='边界样本补全与主动发现工具')
    parser.add_argument('--input_dir', default=str(PROJECT_ROOT / 'npy_jobs'), help='包含 36data/37data/74data 的输入根目录')
    parser.add_argument('--output_dir', default=str(PROJECT_ROOT / 'results' / 'boundary_scan'), help='用于保存 CSV、图和报告的输出目录')
    parser.add_argument('--config', default=str(SCRIPT_DIR / 'config.yaml'), help='配置 YAML 路径')
    parser.add_argument('--max_files', type=int, default=None, help='用于快速 smoke test 的可选样本上限')
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    reset_output_dir(output_dir)

    scan_cfg = config.get('scan', {})
    ranking_cfg = config.get('ranking', {})
    outputs_cfg = config.get('outputs', {})
    dataset_dirs = scan_cfg.get('datasets', ['36data', '37data', '74data'])
    recursive = bool(scan_cfg.get('recursive', True))
    top_k = int(ranking_cfg.get('top_k', 300))
    new_top_k = int(ranking_cfg.get('new_candidate_top_k', top_k))
    log_every = int(scan_cfg.get('log_every', 200))

    samples = collect_sample_paths(input_dir, dataset_dirs, recursive=recursive)
    if args.max_files:
        samples = samples[: args.max_files]

    print(f'[info] output directory reset: {output_dir}')
    print(f'[info] found {len(samples)} sample files under {input_dir}')
    if not samples:
        save_errors([], output_dir)
        print('[warn] no samples found, exiting')
        return

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for index, (dataset_dir, sample_path) in enumerate(samples, start=1):
        try:
            rows.append(extract_sample_features(sample_path, input_dir, dataset_dir, config))
        except Exception as exc:
            short_error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            errors.append({'file': str(sample_path), 'dataset_name': canonical_dataset_name(dataset_dir), 'error': short_error})
        if index == 1 or index % max(log_every, 1) == 0 or index == len(samples):
            print(f'[scan] processed {index}/{len(samples)}')

    save_errors(errors, output_dir)
    if not rows:
        print('[error] all samples failed to parse, only errors.csv was generated')
        return

    all_df = pd.DataFrame(rows)
    seed_csv_path = resolve_project_path(config.get('seed', {}).get('csv_path', PROJECT_ROOT / 'npy_jobs' / 'boundary_suspicious_samples_index.csv'))
    if seed_csv_path.exists():
        seed_index = SeedBoundaryIndex.from_csv(seed_csv_path, input_dir)
        all_df = seed_index.annotate_dataframe(all_df)
        print(f'[info] loaded seed boundary CSV: {seed_csv_path} | rows={seed_index.total_rows}')
    else:
        all_df['is_seed_boundary'] = 0
        all_df['seed_match_key'] = all_df['file'].astype(str)
        print(f'[warn] seed boundary CSV not found: {seed_csv_path}')

    all_df = compute_scores(all_df, config)
    all_df = add_reason_column(all_df)
    candidate_pool = all_df[all_df['category_label'].isin(CANDIDATE_LABELS)].copy()
    candidate_pool = candidate_pool.sort_values(['overall_candidate_score', 'seed_similarity_score'], ascending=[False, False]).reset_index(drop=True)
    reranked_candidates = diverse_rerank(candidate_pool, config, top_k=top_k)
    reranked_candidates['boundary_rank'] = range(1, len(reranked_candidates) + 1)

    new_candidates_pool = candidate_pool[candidate_pool['is_seed_boundary'].fillna(0).astype(int) == 0].copy()
    reranked_new = diverse_rerank(new_candidates_pool, config, top_k=new_top_k)

    category_tops = build_category_topk(candidate_pool, config)
    dataset_tops = build_dataset_topk(reranked_new if not reranked_new.empty else new_candidates_pool, config)
    dataset_summary = build_dataset_summary(all_df, reranked_new)

    plot_rows = _rows_for_plotting(
        reranked_candidates.head(top_k),
        reranked_new.head(new_top_k),
        *category_tops.values(),
        *dataset_tops.values(),
    )
    plot_map: dict[str, Path] = {}
    for index, row in plot_rows.iterrows():
        sample_path = input_dir / str(row['file'])
        plot_path = build_plot_path(output_dir, str(row['dataset_name']), str(row['file']))
        try:
            sample = load_sample_for_plot(sample_path, config)
            save_sample_plot(sample, row.to_dict(), plot_path, config)
            plot_map[str(row['file'])] = plot_path
        except Exception as exc:
            short_error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            errors.append({'file': str(sample_path), 'dataset_name': str(row['dataset_name']), 'error': f'plot error: {short_error}'})
        rendered = index + 1
        if rendered == 1 or rendered % max(log_every, 1) == 0 or rendered == len(plot_rows):
            print(f'[plot] rendered {rendered}/{len(plot_rows)} candidate plots')

    all_df = _merge_plot_paths(all_df.sort_values(['overall_candidate_score', 'seed_similarity_score'], ascending=[False, False]).reset_index(drop=True), plot_map, output_dir)
    reranked_candidates = _attach_plot_paths(reranked_candidates, plot_map, output_dir)
    reranked_new = _attach_plot_paths(reranked_new, plot_map, output_dir)
    category_tops = {name: _attach_plot_paths(frame, plot_map, output_dir) for name, frame in category_tops.items()}
    dataset_tops = {name: _attach_plot_paths(frame, plot_map, output_dir) for name, frame in dataset_tops.items()}

    all_csv_path = output_dir / 'all_samples_boundary_scores.csv'
    all_df.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
    reranked_candidates.to_csv(output_dir / 'suspicious_samples_topk.csv', index=False, encoding='utf-8-sig')
    reranked_new.to_csv(output_dir / 'new_boundary_candidates_topk.csv', index=False, encoding='utf-8-sig')
    dataset_summary.to_csv(output_dir / 'per_dataset_candidate_summary.csv', index=False, encoding='utf-8-sig')

    for prefix, frame in category_tops.items():
        frame.to_csv(output_dir / f'{prefix}_boundary_candidates_topk.csv', index=False, encoding='utf-8-sig')

    save_errors(errors, output_dir)

    sections = {
        'overall_top': reranked_candidates.head(int(outputs_cfg.get('report_overall_top_n', 80))),
        'new_candidates': reranked_new.head(int(outputs_cfg.get('report_new_top_n', 120))),
        'stable_side': category_tops.get('stable_side', pd.DataFrame()).head(int(outputs_cfg.get('report_category_top_n', 80))),
        'unstable_side': category_tops.get('unstable_side', pd.DataFrame()).head(int(outputs_cfg.get('report_category_top_n', 80))),
        'central_ambiguous': category_tops.get('central_ambiguous', pd.DataFrame()).head(int(outputs_cfg.get('report_category_top_n', 80))),
        'dataset_tops': {name: frame.head(int(outputs_cfg.get('report_dataset_top_n', 50))) for name, frame in dataset_tops.items()},
    }
    html_path, md_path = build_reports(all_df, sections, output_dir, config)

    print(f'[info] wrote {all_csv_path}')
    print(f'[info] wrote {output_dir / "suspicious_samples_topk.csv"}')
    print(f'[info] wrote {output_dir / "new_boundary_candidates_topk.csv"}')
    print(f'[info] wrote {output_dir / "per_dataset_candidate_summary.csv"}')
    print(f'[info] wrote {html_path}')
    if outputs_cfg.get('write_report_md', True):
        print(f'[info] wrote {md_path}')
    print('[done] boundary scan finished')


if __name__ == '__main__':
    main()
