from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CANDIDATE_LABELS = {
    'stable_side_boundary_candidate',
    'unstable_side_boundary_candidate',
    'central_ambiguous_candidate',
}

SIMILARITY_FEATURE_COLUMNS = [
    'tail_voltage_min',
    'final_recovered_ratio_0_9',
    'tail_spread_mean',
    'tail_spread_slope',
    'angle_speed_median',
    'spread_peak',
    'tail_amp_top1',
    'tail_amp_top2',
    'tail_amp_top1_ratio',
    'amp_std_across_channels',
    'top1_minus_median_amp',
    'decay_ratio_top1',
    'decay_ratio_mean',
    'voltage_rebound_instability_score',
    'tail_low_voltage_reentry_count',
    'spread_reentry_count',
    'tail_sign_change_density',
    'oscillation_persistence_score',
]


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def gaussian_closeness(value: Any, center: float, sigma: float) -> float:
    value = _safe_float(value)
    if not np.isfinite(value):
        return 0.0
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-((value - float(center)) ** 2) / (2.0 * sigma * sigma)))


def _branch_score_from_terms(df: pd.DataFrame, terms: dict[str, dict[str, Any]]) -> pd.Series:
    if df.empty or not terms:
        return pd.Series(0.0, index=df.index, dtype=float)
    numerator = pd.Series(0.0, index=df.index, dtype=float)
    denominator = 0.0
    for feature_name, term_cfg in terms.items():
        if feature_name not in df.columns:
            continue
        weight = float(term_cfg.get('weight', 0.0))
        if weight <= 0.0:
            continue
        center = float(term_cfg.get('center', term_cfg.get('tau', 0.0)))
        sigma = float(term_cfg.get('sigma', 1.0))
        values = df[feature_name].astype(float)
        closeness = values.apply(lambda x: gaussian_closeness(x, center, sigma))
        numerator = numerator + weight * closeness
        denominator += weight
    if denominator <= 0.0:
        return pd.Series(0.0, index=df.index, dtype=float)
    return numerator / denominator


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vec_norm = float(np.linalg.norm(vector))
    if vec_norm <= 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    mat_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(mat_norm * vec_norm, 1e-8)
    return np.clip((matrix @ vector) / denom, -1.0, 1.0)


def compute_seed_similarity(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    if df.empty or 'is_seed_boundary' not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)

    scoring_cfg = config.get('scoring_branches', {})
    seed_cfg = config.get('seed', {})
    same_dataset_bias = float(seed_cfg.get('same_dataset_bias', scoring_cfg.get('same_dataset_seed_bias', 0.15)))
    top_k = int(seed_cfg.get('nearest_k', 5))

    feature_cols = [col for col in scoring_cfg.get('similarity_features', SIMILARITY_FEATURE_COLUMNS) if col in df.columns]
    if not feature_cols:
        return pd.Series(0.0, index=df.index, dtype=float)

    feature_frame = df[feature_cols].astype(float).copy()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    medians = feature_frame.median(axis=0, skipna=True).fillna(0.0)
    feature_frame = feature_frame.fillna(medians)
    means = feature_frame.mean(axis=0)
    stds = feature_frame.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    z = ((feature_frame - means) / stds).to_numpy(dtype=np.float64)

    seed_mask = df['is_seed_boundary'].fillna(0).astype(int) == 1
    if int(seed_mask.sum()) == 0:
        return pd.Series(0.0, index=df.index, dtype=float)

    all_seed = z[seed_mask.to_numpy()]
    dataset_names = df['dataset_name'].astype(str).tolist() if 'dataset_name' in df.columns else [''] * len(df)
    similarities = []

    for idx in range(len(df)):
        current = z[idx]
        global_cos = _cosine_similarity(all_seed, current)
        global_score = float(np.mean(np.sort(global_cos)[-min(top_k, len(global_cos)):])) if global_cos.size else 0.0
        same_dataset_mask = seed_mask & (df['dataset_name'].astype(str) == dataset_names[idx]) if 'dataset_name' in df.columns else seed_mask
        local_seed = z[same_dataset_mask.to_numpy()]
        local_score = 0.0
        if local_seed.size:
            local_cos = _cosine_similarity(local_seed, current)
            local_score = float(np.mean(np.sort(local_cos)[-min(top_k, len(local_cos)):]))
        similarities.append(np.clip(0.5 * (global_score + 1.0) + same_dataset_bias * 0.5 * (local_score + 1.0), 0.0, 1.0))
    return pd.Series(similarities, index=df.index, dtype=float)


def _count_conditions(*conditions: pd.Series) -> pd.Series:
    total = pd.Series(0, index=conditions[0].index if conditions else None, dtype=int)
    for condition in conditions:
        total = total + condition.fillna(False).astype(int)
    return total


def compute_scores(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    scoring_cfg = config.get('scoring_branches', {})
    obvious_stable = scoring_cfg.get('obvious_stable', {})
    obvious_unstable = scoring_cfg.get('obvious_unstable', {})
    branch_weights = scoring_cfg.get('branch_mix', {})
    candidate_threshold = float(scoring_cfg.get('candidate_threshold', 0.42))
    low_conf_threshold = float(scoring_cfg.get('low_confidence_threshold', 0.33))

    result['seed_similarity_score'] = compute_seed_similarity(result, config)
    result['g_v'] = result['tail_voltage_min'].apply(lambda x: gaussian_closeness(x, 0.9, 0.04))
    result['g_r'] = result['final_recovered_ratio_0_9'].apply(lambda x: gaussian_closeness(x, 0.8, 0.1))
    result['g_s'] = result['tail_spread_mean'].apply(lambda x: gaussian_closeness(x, 120.0, 30.0))
    result['g_w'] = result['angle_speed_median'].apply(lambda x: gaussian_closeness(x, 16.0, 8.0))
    result['raw_boundary_score'] = _branch_score_from_terms(result, scoring_cfg.get('base_terms', {}))

    result['stable_side_score'] = _branch_score_from_terms(result, scoring_cfg.get('stable_side_terms', {}))
    result['unstable_side_score'] = _branch_score_from_terms(result, scoring_cfg.get('unstable_side_terms', {}))
    result['central_ambiguous_score'] = _branch_score_from_terms(result, scoring_cfg.get('central_ambiguous_terms', {}))

    result['stable_clear_flag'] = (
        (result['tail_voltage_min'] >= float(obvious_stable.get('tail_voltage_min_gte', 0.96)))
        & (result['final_recovered_ratio_0_9'] >= float(obvious_stable.get('final_recovered_ratio_0_9_gte', 0.98)))
        & (result['tail_spread_mean'] <= float(obvious_stable.get('tail_spread_mean_lte', 70.0)))
        & (result['angle_speed_median'] <= float(obvious_stable.get('angle_speed_median_lte', 6.0)))
        & (result['tail_amp_top1'] <= float(obvious_stable.get('tail_amp_top1_lte', 12.0)))
    ).astype(int)
    stable_hard_condition_count = _count_conditions(
        result['tail_voltage_min'] >= float(obvious_stable.get('hard_tail_voltage_min_gte', 0.95)),
        result['final_recovered_ratio_0_9'] >= float(obvious_stable.get('hard_final_recovered_ratio_0_9_gte', 0.99)),
        result['tail_spread_mean'] <= float(obvious_stable.get('hard_tail_spread_mean_lte', 75.0)),
        result['angle_speed_median'] <= float(obvious_stable.get('hard_angle_speed_median_lte', 14.0)),
        result['tail_amp_top1'] <= float(obvious_stable.get('hard_tail_amp_top1_lte', 10.0)),
        result['large_amp_channel_count_20'] <= float(obvious_stable.get('hard_large_amp_channel_count_20_lte', 0.0)),
        result['tail_low_voltage_reentry_count'] <= float(obvious_stable.get('hard_tail_low_voltage_reentry_count_lte', 0.0)),
        result['spread_reentry_count'] <= float(obvious_stable.get('hard_spread_reentry_count_lte', 0.0)),
        result['oscillation_persistence_score'] <= float(obvious_stable.get('hard_oscillation_persistence_score_lte', 0.50)),
    )
    result['hard_obvious_stable_flag'] = (
        stable_hard_condition_count >= int(obvious_stable.get('hard_min_condition_count', 7))
    ).astype(int)
    result['very_stable_recovered_flag'] = (
        (result['tail_voltage_min'] >= float(obvious_stable.get('very_stable_tail_voltage_min_gte', 0.96)))
        & (result['final_recovered_ratio_0_9'] >= float(obvious_stable.get('very_stable_final_recovered_ratio_0_9_gte', 0.995)))
        & (result['tail_low_voltage_reentry_count'] <= float(obvious_stable.get('very_stable_tail_low_voltage_reentry_count_lte', 0.0)))
        & (result['spread_reentry_count'] <= float(obvious_stable.get('very_stable_spread_reentry_count_lte', 0.0)))
        & (result['tail_spread_mean'] <= float(obvious_stable.get('very_stable_tail_spread_mean_lte', 90.0)))
        & (result['angle_speed_median'] <= float(obvious_stable.get('very_stable_angle_speed_median_lte', 16.0)))
        & (result['large_amp_channel_count_20'] <= float(obvious_stable.get('very_stable_large_amp_channel_count_20_lte', 5.0)))
    ).astype(int)
    stable_candidate_dynamic_count = _count_conditions(
        result['tail_spread_mean'] >= float(obvious_stable.get('candidate_tail_spread_mean_gte', 95.0)),
        result['angle_speed_median'] >= float(obvious_stable.get('candidate_angle_speed_median_gte', 12.0)),
        result['tail_amp_top1'] >= float(obvious_stable.get('candidate_tail_amp_top1_gte', 24.0)),
        result['large_amp_channel_count_20'] >= float(obvious_stable.get('candidate_large_amp_channel_count_20_gte', 2.0)),
        result['oscillation_persistence_score'] >= float(obvious_stable.get('candidate_oscillation_persistence_score_gte', 0.58)),
    )
    stable_candidate_risk_count = _count_conditions(
        result['tail_voltage_min'] <= float(obvious_stable.get('candidate_tail_voltage_min_lte', 0.94)),
        result['final_recovered_ratio_0_9'] <= float(obvious_stable.get('candidate_final_recovered_ratio_0_9_lte', 0.99)),
        result['tail_low_voltage_reentry_count'] >= float(obvious_stable.get('candidate_tail_low_voltage_reentry_count_gte', 1.0)),
        result['spread_reentry_count'] >= float(obvious_stable.get('candidate_spread_reentry_count_gte', 1.0)),
        result['voltage_rebound_instability_score'] >= float(obvious_stable.get('candidate_voltage_rebound_instability_score_gte', 0.42)),
    )
    result['stable_candidate_min_risk_flag'] = (
        (stable_candidate_dynamic_count >= int(obvious_stable.get('candidate_min_dynamic_condition_count', 2)))
        & (stable_candidate_risk_count >= int(obvious_stable.get('candidate_min_risk_condition_count', 1)))
    ).astype(int)

    unstable_condition_count = _count_conditions(
        result['tail_voltage_min'] <= float(obvious_unstable.get('tail_voltage_min_lte', 0.74)),
        result['final_recovered_ratio_0_9'] <= float(obvious_unstable.get('final_recovered_ratio_0_9_lte', 0.3)),
        result['tail_spread_mean'] >= float(obvious_unstable.get('tail_spread_mean_gte', 180.0)),
        result['duration_below_0_8'] >= float(obvious_unstable.get('duration_below_0_8_gte', 2.0)),
        result['non_recovery_ratio'] >= float(obvious_unstable.get('non_recovery_ratio_gte', 0.6)),
    )
    result['dynamic_unstable_flag'] = (
        (result['spread_peak'] >= float(obvious_unstable.get('spread_peak_gte', 180.0)))
        | (result['angle_speed_p90'] >= float(obvious_unstable.get('angle_speed_p90_gte', 50.0)))
        | (result['invalid_angle_ratio'] >= float(obvious_unstable.get('invalid_angle_ratio_gte', 0.08)))
    ).astype(int)
    severe_dynamic_condition_count = _count_conditions(
        result['tail_spread_mean'] >= float(obvious_unstable.get('severe_tail_spread_mean_gte', 145.0)),
        result['angle_speed_median'] >= float(obvious_unstable.get('severe_angle_speed_median_gte', 35.0)),
        result['angle_speed_p90'] >= float(obvious_unstable.get('severe_angle_speed_p90_gte', 60.0)),
        result['tail_amp_top1'] >= float(obvious_unstable.get('severe_tail_amp_top1_gte', 80.0)),
        result['large_amp_channel_count_20'] >= float(obvious_unstable.get('severe_large_amp_channel_count_20_gte', 6.0)),
        result['spread_peak'] >= float(obvious_unstable.get('severe_spread_peak_gte', 200.0)),
    )
    result['severe_dynamic_unstable_flag'] = (
        severe_dynamic_condition_count >= int(obvious_unstable.get('severe_min_condition_count', 3))
    ).astype(int)
    result['hard_obvious_unstable_flag'] = (
        (result['tail_voltage_min'] <= float(obvious_unstable.get('hard_tail_voltage_min_lte', 0.20)))
        | ((result['tail_amp_top1'] >= float(obvious_unstable.get('hard_tail_amp_top1_gte', 120.0)))
           & (result['tail_amp_top2'] >= float(obvious_unstable.get('hard_tail_amp_top2_gte', 100.0))))
        | ((result['large_amp_channel_count_20'] >= float(obvious_unstable.get('hard_large_amp_channel_count_20_gte', 5.0)))
           & (result['tail_spread_mean'] >= float(obvious_unstable.get('hard_tail_spread_mean_gte', 130.0))))
    ).astype(int)
    unstable_candidate_dynamic_count = _count_conditions(
        result['tail_spread_mean'] >= float(obvious_unstable.get('candidate_tail_spread_mean_gte', 95.0)),
        result['angle_speed_median'] >= float(obvious_unstable.get('candidate_angle_speed_median_gte', 8.0)),
        result['tail_amp_top1'] >= float(obvious_unstable.get('candidate_tail_amp_top1_gte', 18.0)),
        result['large_amp_channel_count_20'] >= float(obvious_unstable.get('candidate_large_amp_channel_count_20_gte', 1.0)),
        result['spread_reentry_count'] >= float(obvious_unstable.get('candidate_spread_reentry_count_gte', 1.0)),
        result['oscillation_persistence_score'] >= float(obvious_unstable.get('candidate_oscillation_persistence_score_gte', 0.48)),
    )
    unstable_candidate_core_dynamic_count = _count_conditions(
        result['tail_spread_mean'] >= float(obvious_unstable.get('candidate_core_tail_spread_mean_gte', 85.0)),
        result['angle_speed_median'] >= float(obvious_unstable.get('candidate_core_angle_speed_median_gte', 6.0)),
        result['tail_amp_top1'] >= float(obvious_unstable.get('candidate_core_tail_amp_top1_gte', 15.0)),
        result['large_amp_channel_count_20'] >= float(obvious_unstable.get('candidate_core_large_amp_channel_count_20_gte', 1.0)),
    )
    result['unstable_candidate_core_dynamic_flag'] = (
        unstable_candidate_core_dynamic_count >= int(obvious_unstable.get('candidate_core_min_dynamic_condition_count', 1))
    ).astype(int)
    result['unstable_candidate_min_dynamic_flag'] = (
        (unstable_candidate_dynamic_count >= int(obvious_unstable.get('candidate_min_dynamic_condition_count', 2)))
        & (result['unstable_candidate_core_dynamic_flag'] == 1)
    ).astype(int)
    result['unstable_clear_flag'] = ((unstable_condition_count >= int(obvious_unstable.get('min_condition_count', 2))) | (result['dynamic_unstable_flag'] == 1) | (result['severe_dynamic_unstable_flag'] == 1) | (result['hard_obvious_unstable_flag'] == 1)).astype(int)

    label_series = result.get('original_label', pd.Series('', index=result.index)).astype(str).str.lower()
    stable_mix = branch_weights.get('stable_label', {'stable_side_score': 0.5, 'central_ambiguous_score': 0.2, 'seed_similarity_score': 0.3})
    unstable_mix = branch_weights.get('unstable_label', {'unstable_side_score': 0.5, 'central_ambiguous_score': 0.2, 'seed_similarity_score': 0.3})
    default_mix = branch_weights.get('default', {'stable_side_score': 0.3, 'unstable_side_score': 0.3, 'central_ambiguous_score': 0.2, 'seed_similarity_score': 0.2})

    def _mix_score(row: pd.Series, weights: dict[str, float]) -> float:
        return float(sum(float(weights.get(name, 0.0)) * float(row.get(name, 0.0)) for name in weights))

    overall_scores = []
    for idx, row in result.iterrows():
        if label_series.iloc[idx] == 'stable':
            score = _mix_score(row, stable_mix)
        elif label_series.iloc[idx] == 'unstable':
            score = _mix_score(row, unstable_mix)
        else:
            score = _mix_score(row, default_mix)
        if int(row.get('stable_clear_flag', 0)) == 1 or int(row.get('hard_obvious_stable_flag', 0)) == 1 or int(row.get('very_stable_recovered_flag', 0)) == 1:
            score *= float(scoring_cfg.get('stable_clear_penalty', 0.35))
        if int(row.get('unstable_clear_flag', 0)) == 1:
            score *= float(scoring_cfg.get('unstable_clear_penalty', 0.2))
        overall_scores.append(float(np.clip(score, 0.0, 1.0)))
    result['overall_candidate_score'] = overall_scores
    result['penalized_boundary_score'] = result['overall_candidate_score']
    result['boundary_score'] = result[['stable_side_score', 'unstable_side_score', 'central_ambiguous_score']].max(axis=1)

    result['stable_side_signal'] = result['stable_side_score']
    result['unstable_side_signal'] = result['unstable_side_score']
    result['high_voltage_fast_signal'] = np.clip(0.5 * result['stable_side_score'] + 0.3 * result['tail_amp_top1_ratio'].fillna(0.0) + 0.2 * result['tail_spread_mean'].apply(lambda x: gaussian_closeness(x, 135.0, 25.0)), 0.0, 1.0)
    result['high_voltage_fast_flag'] = ((label_series == 'stable') & (result['tail_voltage_min'] >= 0.94) & (result['tail_amp_top1'] >= 20.0) & (result['tail_spread_mean'] >= 95.0)).astype(int)

    category_labels = []
    boundary_sides = []
    stable_flags = []
    unstable_flags = []
    central_flags = []
    candidate_flags = []

    for idx, row in result.iterrows():
        branch_scores = {
            'stable_side_boundary_candidate': float(row['stable_side_score']),
            'unstable_side_boundary_candidate': float(row['unstable_side_score']),
            'central_ambiguous_candidate': float(row['central_ambiguous_score']),
        }
        label_name = label_series.iloc[idx]
        top_label = max(branch_scores, key=branch_scores.get)
        top_value = branch_scores[top_label]

        category = top_label
        if label_name == 'stable' and (int(row['hard_obvious_stable_flag']) == 1 or int(row['very_stable_recovered_flag']) == 1):
            category = 'obvious_stable'
        elif int(row['hard_obvious_unstable_flag']) == 1 or int(row['severe_dynamic_unstable_flag']) == 1:
            category = 'obvious_unstable'
        elif top_value < low_conf_threshold:
            if (int(row['stable_clear_flag']) == 1 or int(row['hard_obvious_stable_flag']) == 1) and label_name != 'unstable':
                category = 'obvious_stable'
            elif int(row['unstable_clear_flag']) == 1 or label_name == 'unstable':
                category = 'obvious_unstable'
            else:
                category = 'obvious_stable'
        else:
            if int(row['unstable_clear_flag']) == 1:
                category = 'obvious_unstable'
            elif label_name == 'stable' and int(row['hard_obvious_stable_flag']) == 1:
                category = 'obvious_stable'
            elif label_name == 'stable' and row['stable_side_score'] >= row['central_ambiguous_score'] - 0.03 and int(row['stable_candidate_min_risk_flag']) == 1:
                category = 'stable_side_boundary_candidate'
            elif label_name == 'stable' and int(row['stable_candidate_min_risk_flag']) == 0:
                category = 'obvious_stable'
            elif label_name == 'stable' and top_label == 'unstable_side_boundary_candidate':
                category = 'obvious_stable'
            elif label_name == 'unstable' and row['unstable_side_score'] >= row['central_ambiguous_score'] - 0.03 and int(row['unstable_candidate_min_dynamic_flag']) == 1:
                category = 'unstable_side_boundary_candidate'
            elif label_name == 'unstable' and int(row['unstable_candidate_min_dynamic_flag']) == 0:
                category = 'obvious_unstable'
            elif top_label == 'central_ambiguous_candidate':
                category = 'central_ambiguous_candidate'
            elif int(row['stable_clear_flag']) == 1 and row['stable_side_score'] < candidate_threshold:
                category = 'obvious_stable'

        if category in CANDIDATE_LABELS and float(row['overall_candidate_score']) < candidate_threshold:
            category = 'obvious_unstable' if (label_name == 'unstable' or int(row['unstable_clear_flag']) == 1) else ('central_ambiguous_candidate' if row['central_ambiguous_score'] >= low_conf_threshold else 'obvious_stable')
        if category == 'central_ambiguous_candidate' and label_name == 'unstable' and int(row['unstable_clear_flag']) == 1:
            category = 'obvious_unstable'

        category_labels.append(category)
        boundary_sides.append(category.replace('_candidate', ''))
        stable_flags.append(int(category == 'stable_side_boundary_candidate'))
        unstable_flags.append(int(category == 'unstable_side_boundary_candidate'))
        central_flags.append(int(category == 'central_ambiguous_candidate'))
        candidate_flags.append(int(category in CANDIDATE_LABELS))

    result['category_label'] = category_labels
    result['boundary_side'] = boundary_sides
    result['stable_side_boundary_flag'] = stable_flags
    result['unstable_side_boundary_flag'] = unstable_flags
    result['central_ambiguous_flag'] = central_flags
    result['boundary_candidate_flag'] = candidate_flags
    result['side_bonus'] = np.clip(result['overall_candidate_score'] - result['raw_boundary_score'], -1.0, 1.0)
    result['strong_high_voltage_override'] = ((result['high_voltage_fast_flag'] == 1) & (result['stable_side_score'] >= 0.6)).astype(int)
    result['obvious_stable_due_to_hard_rule'] = result['hard_obvious_stable_flag']
    result['obvious_stable_due_to_clean_recovery'] = result['very_stable_recovered_flag']
    result['obvious_stable_due_to_weak_boundary_risk'] = ((label_series == 'stable') & (result['stable_candidate_min_risk_flag'] == 0)).astype(int)
    result['obvious_unstable_due_to_dynamics'] = result['severe_dynamic_unstable_flag']
    result['obvious_unstable_due_to_hard_rule'] = result['hard_obvious_unstable_flag']
    result['obvious_unstable_due_to_weak_boundary_dynamics'] = ((label_series == 'unstable') & (result['unstable_candidate_min_dynamic_flag'] == 0)).astype(int)
    result['obvious_unstable_due_to_missing_core_dynamics'] = ((label_series == 'unstable') & (result['unstable_candidate_core_dynamic_flag'] == 0)).astype(int)
    return result
