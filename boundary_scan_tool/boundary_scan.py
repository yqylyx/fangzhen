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
from report_builder import add_reason_column, build_html_report
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Boundary sample scan and visualization tool')
    parser.add_argument('--input_dir', default=str(PROJECT_ROOT / 'npy_jobs'), help='Input root directory containing 36data/37data/74data')
    parser.add_argument('--output_dir', default=str(PROJECT_ROOT / 'results' / 'boundary_scan'), help='Output directory for CSV, plots and report')
    parser.add_argument('--config', default=str(SCRIPT_DIR / 'config.yaml'), help='Path to config YAML')
    parser.add_argument('--max_files', type=int, default=None, help='Optional limit for quick smoke testing')
    args = parser.parse_args()

    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    config = load_config(args.config)
    reset_output_dir(output_dir)

    scan_cfg = config['scan']
    dataset_dirs = scan_cfg.get('datasets', ['36data', '37data', '74data'])
    recursive = bool(scan_cfg.get('recursive', True))
    top_k = int(scan_cfg.get('top_k', 300))
    log_every = int(scan_cfg.get('log_every', 200))

    samples = collect_sample_paths(input_dir, dataset_dirs, recursive=recursive)
    if args.max_files:
        samples = samples[: args.max_files]

    print(f'[info] output directory reset: {output_dir}')
    print(f'[info] found {len(samples)} sample files under {input_dir}')
    if not samples:
        print('[warn] no samples found, exiting')
        save_errors([], output_dir)
        return

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for index, (dataset_dir, sample_path) in enumerate(samples, start=1):
        try:
            row = extract_sample_features(sample_path, input_dir, dataset_dir, config)
            rows.append(row)
        except Exception as exc:
            short_error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            errors.append(
                {
                    'file': str(sample_path),
                    'dataset_name': canonical_dataset_name(dataset_dir),
                    'error': short_error,
                }
            )
        if index == 1 or index % max(log_every, 1) == 0 or index == len(samples):
            print(f'[scan] processed {index}/{len(samples)}')

    save_errors(errors, output_dir)

    if not rows:
        print('[error] all samples failed to parse, only errors.csv was generated')
        return

    all_df = pd.DataFrame(rows)
    all_df = add_reason_column(all_df)
    all_df = all_df.sort_values(['boundary_score', 'raw_boundary_score'], ascending=[False, False]).reset_index(drop=True)

    candidate_mask = all_df.get('boundary_candidate_flag', 1).fillna(1).astype(int) == 1
    all_df['boundary_rank'] = ''
    all_df.loc[candidate_mask, 'boundary_rank'] = range(1, int(candidate_mask.sum()) + 1)

    candidate_df = all_df.loc[candidate_mask].copy().reset_index(drop=True)

    all_csv_path = output_dir / 'all_samples_boundary_scores.csv'
    all_df.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
    print(f'[info] wrote {all_csv_path}')
    print(f'[info] candidate boundary samples kept for Top-K/report: {len(candidate_df)}')
    print(f'[info] excluded non-candidate samples (clear stable / clear unstable): {int((~candidate_mask).sum())}')

    top_df = candidate_df.head(min(top_k, len(candidate_df))).copy()
    plot_rel_paths: List[str] = []

    for index, row in top_df.iterrows():
        sample_path = input_dir / row['file']
        plot_path = build_plot_path(output_dir, str(row['dataset_name']), str(row['file']))
        try:
            sample = load_sample_for_plot(sample_path, config)
            save_sample_plot(sample, row, plot_path, config)
            plot_rel_paths.append(plot_path.relative_to(output_dir).as_posix())
        except Exception as exc:
            short_error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            plot_rel_paths.append('')
            errors.append(
                {
                    'file': str(sample_path),
                    'dataset_name': str(row['dataset_name']),
                    'error': f'plot error: {short_error}',
                }
            )
        rendered = index + 1
        if rendered == 1 or rendered % max(log_every, 1) == 0 or rendered == len(top_df):
            print(f'[plot] rendered {rendered}/{len(top_df)} top suspicious samples')

    top_df['plot_rel_path'] = plot_rel_paths
    top_csv_path = output_dir / 'suspicious_samples_topk.csv'
    top_df.to_csv(top_csv_path, index=False, encoding='utf-8-sig')
    print(f'[info] wrote {top_csv_path}')

    save_errors(errors, output_dir)
    report_path = build_html_report(all_df, top_df, output_dir, config)
    print(f'[info] wrote {report_path}')
    print('[done] boundary scan finished')


if __name__ == '__main__':
    main()
