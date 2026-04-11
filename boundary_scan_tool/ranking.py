from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

CYCLE_DEFAULT = [
    'stable_side_boundary_candidate',
    'central_ambiguous_candidate',
    'unstable_side_boundary_candidate',
]


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(['overall_candidate_score', 'seed_similarity_score', 'tail_amp_top1'], ascending=[False, False, False]).reset_index(drop=True)


def diverse_rerank(df: pd.DataFrame, config: dict[str, Any], top_k: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranking_cfg = config['ranking']
    cycle = list(ranking_cfg.get('category_cycle', CYCLE_DEFAULT))
    limit = int(top_k or ranking_cfg.get('top_k', len(df)))
    dataset_cap = int(ranking_cfg.get('max_per_dataset_in_topk', max(1, limit // 2)))

    grouped = {
        label: deque(_sort_candidates(df[df['category_label'] == label]).to_dict(orient='records'))
        for label in cycle
    }
    fallback = deque(_sort_candidates(df).to_dict(orient='records'))
    chosen: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    dataset_counts: defaultdict[str, int] = defaultdict(int)

    while len(chosen) < min(limit, len(df)):
        progressed = False
        for label in cycle:
            queue = grouped[label]
            while queue:
                row = queue[0]
                if row['file'] in seen_files:
                    queue.popleft()
                    continue
                if dataset_counts[str(row['dataset_name'])] >= dataset_cap and len(chosen) < max(len(cycle), limit // 2):
                    break
                row = queue.popleft()
                chosen.append(row)
                seen_files.add(row['file'])
                dataset_counts[str(row['dataset_name'])] += 1
                progressed = True
                break
            if len(chosen) >= min(limit, len(df)):
                break
        if progressed:
            continue
        while fallback and len(chosen) < min(limit, len(df)):
            row = fallback.popleft()
            if row['file'] in seen_files:
                continue
            chosen.append(row)
            seen_files.add(row['file'])
            break
        else:
            break

    ranked = pd.DataFrame(chosen)
    ranked['overall_rank'] = range(1, len(ranked) + 1)
    return ranked


def build_category_topk(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_k = int(config['ranking'].get('per_category_top_k', 80))
    result: dict[str, pd.DataFrame] = {}
    for label, prefix in [
        ('stable_side_boundary_candidate', 'stable_side'),
        ('unstable_side_boundary_candidate', 'unstable_side'),
        ('central_ambiguous_candidate', 'central_ambiguous'),
    ]:
        subset = _sort_candidates(df[df['category_label'] == label]).head(top_k).copy()
        subset['category_rank'] = range(1, len(subset) + 1)
        result[prefix] = subset
    return result


def build_dataset_topk(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    top_k = int(config['ranking'].get('per_dataset_top_k', 60))
    result: dict[str, pd.DataFrame] = {}
    for dataset_name, subset in df.groupby('dataset_name', sort=True):
        ranked = _sort_candidates(subset).head(top_k).copy()
        ranked['dataset_rank'] = range(1, len(ranked) + 1)
        result[str(dataset_name)] = ranked
    return result


def build_dataset_summary(all_df: pd.DataFrame, new_candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, subset in all_df.groupby('dataset_name', sort=True):
        rows.append(
            {
                'dataset_name': dataset_name,
                'total_samples': int(len(subset)),
                'seed_boundary_samples': int(subset['is_seed_boundary'].sum()),
                'new_candidate_samples': int((new_candidates_df['dataset_name'] == dataset_name).sum()),
                'stable_side_candidates': int((subset['category_label'] == 'stable_side_boundary_candidate').sum()),
                'unstable_side_candidates': int((subset['category_label'] == 'unstable_side_boundary_candidate').sum()),
                'central_ambiguous_candidates': int((subset['category_label'] == 'central_ambiguous_candidate').sum()),
            }
        )
    return pd.DataFrame(rows)
