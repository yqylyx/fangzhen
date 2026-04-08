from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


KEY_ALIASES = {
    'times': ('times', 't', 'time'),
    'voltages': ('voltages', 'V', 'voltage'),
    'angles': ('angles', 'delta', 'angle'),
    'label': ('label', 'y', 'target'),
}


def find_first(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f'Missing keys {keys} in payload')


def canonical_dataset_name(dataset_dir_name: str) -> str:
    lowered = dataset_dir_name.lower()
    if lowered.startswith('36'):
        return '36'
    if lowered.startswith('37'):
        return '37'
    if lowered.startswith('74'):
        return '74'
    return dataset_dir_name


def normalize_label(value: Any) -> Tuple[int | None, str]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'unstable', 'unsteady'}:
            return 1, 'unstable'
        if lowered in {'0', 'stable'}:
            return 0, 'stable'
        return None, value.strip()
    if value is None:
        return None, ''
    try:
        ivalue = int(value)
    except Exception:
        return None, str(value)
    if ivalue == 0:
        return 0, 'stable'
    if ivalue == 1:
        return 1, 'unstable'
    return ivalue, str(ivalue)


def _clean_numeric_array(values: Any, invalid_value: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array[~np.isfinite(array)] = np.nan
    if np.isfinite(invalid_value):
        array[np.isclose(array, invalid_value, atol=1e-9)] = np.nan
    return array


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError(f'Expected 2D array, got shape {values.shape}')
    return values


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f'{path} does not contain a dict payload')
    return payload


def load_sample(path: str | Path, invalid_value: float) -> Dict[str, Any]:
    sample_path = Path(path)
    payload = _load_payload(sample_path)

    label_value, label_name = normalize_label(find_first(payload, *KEY_ALIASES['label']))

    times = _clean_numeric_array(find_first(payload, *KEY_ALIASES['times']), invalid_value)
    voltages = _ensure_2d(_clean_numeric_array(find_first(payload, *KEY_ALIASES['voltages']), invalid_value))
    angles = _ensure_2d(_clean_numeric_array(find_first(payload, *KEY_ALIASES['angles']), invalid_value))

    min_len = min(times.shape[0], voltages.shape[0], angles.shape[0])
    if min_len <= 1:
        raise ValueError(f'{sample_path} has too few valid time steps')

    times = times[:min_len]
    voltages = voltages[:min_len]
    angles = angles[:min_len]

    finite_times = times[np.isfinite(times)]
    if np.isnan(times).any() or finite_times.size < 2 or np.any(np.diff(finite_times) <= 0):
        times = np.arange(min_len, dtype=np.float64)

    return {
        'path': sample_path,
        'payload': payload,
        'times': times,
        'voltages': voltages,
        'angles': angles,
        'label_value': label_value,
        'label_name': label_name,
    }


def build_relative_angles(angles: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode or 'per_time_median').strip().lower()
    result = angles.astype(np.float64, copy=True)

    if mode in {'per_time_median', 'time_median', 'median', 'per-time-median'}:
        reference = np.nanmedian(result, axis=1, keepdims=True)
        return result - reference

    if mode in {'subtract_initial', 'initial', 'per_channel_initial'}:
        initial = np.full((1, result.shape[1]), np.nan, dtype=np.float64)
        for channel in range(result.shape[1]):
            valid_index = np.flatnonzero(np.isfinite(result[:, channel]))
            if valid_index.size:
                initial[0, channel] = result[valid_index[0], channel]
        return result - initial

    raise ValueError(f'Unsupported relative angle mode: {mode}')


def _window_mask(times: np.ndarray, window_sec: float) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    end_time = float(times[-1])
    start_time = end_time - float(window_sec)
    return np.isfinite(times) & (times >= start_time)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float('nan')
    return float(np.nanmean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float('nan')
    return float(np.nanstd(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float('nan')
    return float(np.nanmin(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).any():
        return float('nan')
    return float(np.nanmax(values))


def _integrate_series(series: np.ndarray, times: np.ndarray) -> float:
    mask = np.isfinite(series) & np.isfinite(times)
    if mask.sum() < 2:
        if mask.sum() == 1:
            return float(series[mask][0])
        return 0.0
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(series[mask], times[mask]))
    return float(np.trapz(series[mask], times[mask]))


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() < 2:
        return float('nan')
    x = times[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.dot(x_centered, y - y.mean()))
    return numerator / denominator


def gaussian_closeness(value: float, tau: float, sigma: float) -> float:
    if not np.isfinite(value):
        return 0.0
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-((float(value) - float(tau)) ** 2) / (2.0 * sigma * sigma)))


def compute_voltage_features(times: np.ndarray, voltages: np.ndarray, tail_window_sec: float, final_window_sec: float) -> Dict[str, float]:
    tail_mask = _window_mask(times, tail_window_sec)
    final_mask = _window_mask(times, final_window_sec)

    min_voltage_t = np.nanmin(voltages, axis=1)
    mean_voltage_t = np.nanmean(voltages, axis=1)

    final_voltages = voltages[final_mask]
    final_channel_mean = np.nanmean(final_voltages, axis=0) if final_voltages.size else np.array([], dtype=np.float64)

    return {
        'min_voltage_global': _safe_min(voltages),
        'tail_voltage_mean': _safe_mean(voltages[tail_mask]),
        'tail_voltage_min': _safe_mean(min_voltage_t[tail_mask]),
        'tail_voltage_std': _safe_std(mean_voltage_t[tail_mask]),
        'final_recovered_ratio_0_9': _safe_mean((final_voltages >= 0.9).astype(np.float64)),
        'final_recovered_ratio_0_95': _safe_mean((final_voltages >= 0.95).astype(np.float64)),
        'duration_below_0_9': _integrate_series((min_voltage_t < 0.9).astype(np.float64), times),
        'duration_below_0_85': _integrate_series((min_voltage_t < 0.85).astype(np.float64), times),
        'duration_below_0_8': _integrate_series((min_voltage_t < 0.8).astype(np.float64), times),
        'duration_below_0_7': _integrate_series((min_voltage_t < 0.7).astype(np.float64), times),
        'undervoltage_area': _integrate_series(np.maximum(0.0, 0.9 - min_voltage_t), times),
        'non_recovery_ratio': _safe_mean((final_channel_mean < 0.9).astype(np.float64)),
    }


def compute_angle_features(times: np.ndarray, angles: np.ndarray, tail_window_sec: float, relative_angle_mode: str) -> Dict[str, float]:
    angles_rel = build_relative_angles(angles, relative_angle_mode)
    invalid_angle_mask = ~np.isfinite(angles)
    invalid_angle_ratio = float(invalid_angle_mask.mean())

    invalid_rows = np.flatnonzero(np.any(invalid_angle_mask, axis=1))
    first_invalid_time = float(times[invalid_rows[0]]) if invalid_rows.size else float('nan')

    spread_t = np.nanmax(angles_rel, axis=1) - np.nanmin(angles_rel, axis=1)
    tail_mask = _window_mask(times, tail_window_sec)
    tail_times = times[tail_mask]
    tail_spread = spread_t[tail_mask]

    tail_angles = angles_rel[tail_mask]
    angle_speed_abs = np.array([], dtype=np.float64)
    if tail_angles.shape[0] >= 2:
        delta_t = np.diff(tail_times)
        delta_angles = np.diff(tail_angles, axis=0)
        safe_dt = np.where(delta_t > 0, delta_t, np.nan)
        angle_speed = delta_angles / safe_dt[:, None]
        angle_speed_abs = np.abs(angle_speed[np.isfinite(angle_speed)])

    return {
        'spread_peak': _safe_max(spread_t),
        'tail_spread_mean': _safe_mean(tail_spread),
        'tail_spread_std': _safe_std(tail_spread),
        'tail_spread_slope': _linear_slope(tail_times, tail_spread),
        'tail_spread_slope_abs': abs(_linear_slope(tail_times, tail_spread)) if np.isfinite(_linear_slope(tail_times, tail_spread)) else float('nan'),
        'angle_speed_median': float(np.nanmedian(angle_speed_abs)) if angle_speed_abs.size else float('nan'),
        'angle_speed_p90': float(np.nanpercentile(angle_speed_abs, 90)) if angle_speed_abs.size else float('nan'),
        'invalid_angle_ratio': invalid_angle_ratio,
        'first_invalid_time': first_invalid_time,
    }


def _feature_value(features: Dict[str, Any], name: str) -> float:
    if name.endswith('_abs') and name[:-4] in features:
        value = features.get(name[:-4])
        return abs(float(value)) if value is not None and np.isfinite(value) else float('nan')
    value = features.get(name)
    try:
        value = float(value)
    except Exception:
        return float('nan')
    return value if np.isfinite(value) else float('nan')


def _weighted_gaussian_score(features: Dict[str, Any], score_terms: Dict[str, Dict[str, float]]) -> float:
    total = 0.0
    total_weight = 0.0
    for feature_name, term_cfg in score_terms.items():
        weight = float(term_cfg.get('weight', 0.0))
        if weight <= 0.0:
            continue
        value = _feature_value(features, feature_name)
        total += weight * gaussian_closeness(value, term_cfg['tau'], term_cfg['sigma'])
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return total / total_weight


def _in_range(value: float, lower: float | None = None, upper: float | None = None) -> bool:
    if not np.isfinite(value):
        return False
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def _count_true(values: Tuple[bool, ...]) -> int:
    return sum(1 for value in values if value)


def compute_side_boundary_features(features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    side_cfg = config['score'].get('side_boundary', {})
    label_name = str(features.get('original_label', '')).lower()

    stable_cfg = side_cfg.get('stable_side', {})
    unstable_cfg = side_cfg.get('unstable_side', {})
    high_voltage_cfg = side_cfg.get('high_voltage_fast_instability', {})

    stable_conditions = (
        _in_range(features['tail_spread_mean'], stable_cfg.get('tail_spread_mean_gte'), stable_cfg.get('tail_spread_mean_lte')),
        _in_range(features['angle_speed_median'], stable_cfg.get('angle_speed_median_gte'), stable_cfg.get('angle_speed_median_lte')),
        _in_range(features['tail_voltage_min'], stable_cfg.get('tail_voltage_min_gte'), stable_cfg.get('tail_voltage_min_lte')),
        _in_range(features['final_recovered_ratio_0_9'], stable_cfg.get('final_recovered_ratio_0_9_gte'), stable_cfg.get('final_recovered_ratio_0_9_lte')),
        _in_range(features['tail_spread_slope_abs'], None, stable_cfg.get('tail_spread_slope_abs_lte')),
    )
    stable_side_signal = _weighted_gaussian_score(features, stable_cfg.get('score_terms', {}))

    high_voltage_conditions = (
        _in_range(features['tail_voltage_min'], high_voltage_cfg.get('tail_voltage_min_gte'), None),
        _in_range(features['final_recovered_ratio_0_9'], high_voltage_cfg.get('final_recovered_ratio_0_9_gte'), None),
        _in_range(features['tail_spread_mean'], high_voltage_cfg.get('tail_spread_mean_gte'), high_voltage_cfg.get('tail_spread_mean_lte')),
        _in_range(features['angle_speed_median'], high_voltage_cfg.get('angle_speed_median_gte'), None),
        _in_range(features['spread_peak'], high_voltage_cfg.get('spread_peak_gte'), None),
    )
    high_voltage_fast_signal = _weighted_gaussian_score(features, high_voltage_cfg.get('score_terms', {}))
    high_voltage_fast_flag = int(
        label_name == 'stable'
        and _count_true(high_voltage_conditions) >= int(high_voltage_cfg.get('min_condition_count', 3))
    )

    stable_side_boundary_flag = int(
        label_name == 'stable'
        and (
            _count_true(stable_conditions) >= int(stable_cfg.get('min_condition_count', 2))
            or high_voltage_fast_flag == 1
        )
    )
    stable_side_bonus = 0.0
    if stable_side_boundary_flag:
        stable_side_bonus += float(stable_cfg.get('bonus_weight', 0.0)) * stable_side_signal
    if high_voltage_fast_flag:
        stable_side_bonus += float(high_voltage_cfg.get('bonus_weight', 0.0)) * high_voltage_fast_signal

    unstable_conditions = (
        _in_range(features['tail_spread_mean'], unstable_cfg.get('tail_spread_mean_gte'), unstable_cfg.get('tail_spread_mean_lte')),
        _in_range(features['angle_speed_median'], unstable_cfg.get('angle_speed_median_gte'), unstable_cfg.get('angle_speed_median_lte')),
        _in_range(features['tail_voltage_min'], unstable_cfg.get('tail_voltage_min_gte'), unstable_cfg.get('tail_voltage_min_lte')),
        _in_range(features['final_recovered_ratio_0_9'], unstable_cfg.get('final_recovered_ratio_0_9_gte'), unstable_cfg.get('final_recovered_ratio_0_9_lte')),
        _in_range(features['tail_spread_slope_abs'], None, unstable_cfg.get('tail_spread_slope_abs_lte')),
    )
    unstable_side_signal = _weighted_gaussian_score(features, unstable_cfg.get('score_terms', {}))
    unstable_side_boundary_flag = int(
        label_name == 'unstable'
        and _count_true(unstable_conditions) >= int(unstable_cfg.get('min_condition_count', 2))
    )
    unstable_side_bonus = float(unstable_cfg.get('bonus_weight', 0.0)) * unstable_side_signal if unstable_side_boundary_flag else 0.0

    boundary_side = 'general'
    if high_voltage_fast_flag:
        boundary_side = 'stable_side_high_voltage'
    elif stable_side_boundary_flag:
        boundary_side = 'stable_side'
    elif unstable_side_boundary_flag:
        boundary_side = 'unstable_side'

    return {
        'stable_side_signal': stable_side_signal,
        'unstable_side_signal': unstable_side_signal,
        'high_voltage_fast_signal': high_voltage_fast_signal,
        'stable_side_boundary_flag': stable_side_boundary_flag,
        'unstable_side_boundary_flag': unstable_side_boundary_flag,
        'high_voltage_fast_flag': high_voltage_fast_flag,
        'stable_side_bonus': stable_side_bonus,
        'unstable_side_bonus': unstable_side_bonus,
        'boundary_side': boundary_side,
    }


def apply_boundary_score(features: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    score_cfg = config['score']
    gaussian_terms = score_cfg['gaussian_terms']

    g_v = gaussian_closeness(features['tail_voltage_min'], gaussian_terms['tail_voltage_min']['tau'], gaussian_terms['tail_voltage_min']['sigma'])
    g_r = gaussian_closeness(features['final_recovered_ratio_0_9'], gaussian_terms['final_recovered_ratio_0_9']['tau'], gaussian_terms['final_recovered_ratio_0_9']['sigma'])
    g_s = gaussian_closeness(features['tail_spread_mean'], gaussian_terms['tail_spread_mean']['tau'], gaussian_terms['tail_spread_mean']['sigma'])
    g_w = gaussian_closeness(features['angle_speed_median'], gaussian_terms['angle_speed_median']['tau'], gaussian_terms['angle_speed_median']['sigma'])

    raw_score = (
        gaussian_terms['tail_voltage_min']['weight'] * g_v
        + gaussian_terms['final_recovered_ratio_0_9']['weight'] * g_r
        + gaussian_terms['tail_spread_mean']['weight'] * g_s
        + gaussian_terms['angle_speed_median']['weight'] * g_w
    )

    stable_cfg = score_cfg['stable_clear']
    unstable_cfg = score_cfg['unstable_clear']
    unstable_dynamic_cfg = score_cfg.get('unstable_dynamic_clear', {})

    stable_clear_flag = int(
        (features['tail_voltage_min'] >= stable_cfg['tail_voltage_min_gte'])
        or (features['final_recovered_ratio_0_9'] >= stable_cfg['final_recovered_ratio_0_9_gte'])
        or (features['tail_spread_mean'] <= stable_cfg['tail_spread_mean_lte'])
        or (features['angle_speed_median'] <= stable_cfg['angle_speed_median_lte'])
    )

    dynamic_unstable_conditions = (
        _feature_value(features, 'tail_spread_mean') >= float(unstable_dynamic_cfg.get('tail_spread_mean_gte', 120.0)),
        _feature_value(features, 'angle_speed_median') >= float(unstable_dynamic_cfg.get('angle_speed_median_gte', 45.0)),
        _feature_value(features, 'tail_spread_slope_abs') >= float(unstable_dynamic_cfg.get('tail_spread_slope_abs_gte', 20.0)),
        _feature_value(features, 'spread_peak') >= float(unstable_dynamic_cfg.get('spread_peak_gte', 110.0)),
    )
    dynamic_unstable_flag = int(
        _count_true(dynamic_unstable_conditions) >= int(unstable_dynamic_cfg.get('min_condition_count', 2))
    )

    unstable_clear_flag = int(
        (features['tail_voltage_min'] <= unstable_cfg['tail_voltage_min_lte'])
        or (features['final_recovered_ratio_0_9'] <= unstable_cfg['final_recovered_ratio_0_9_lte'])
        or (features['tail_spread_mean'] >= unstable_cfg['tail_spread_mean_gte'])
        or (features['duration_below_0_8'] >= unstable_cfg['duration_below_0_8_gte'])
        or dynamic_unstable_flag
    )

    penalized_score = raw_score
    if stable_clear_flag:
        penalized_score *= float(score_cfg.get('stable_penalty_multiplier', 0.35))
    if unstable_clear_flag:
        penalized_score *= float(score_cfg.get('unstable_penalty_multiplier', 0.35))

    side_features = compute_side_boundary_features(features, config)
    side_bonus = side_features['stable_side_bonus'] + side_features['unstable_side_bonus']
    max_boundary_score = float(score_cfg.get('max_boundary_score', 1.5))
    boundary_score = min(max_boundary_score, penalized_score + side_bonus)

    stable_override_cfg = score_cfg.get('stable_candidate_override', {})
    strong_high_voltage_override = False
    if stable_clear_flag and side_features.get('high_voltage_fast_flag', 0) == 1:
        override_conditions = (
            side_features.get('high_voltage_fast_signal', 0.0) >= float(stable_override_cfg.get('high_voltage_fast_signal_gte', 0.9)),
            _feature_value(features, 'tail_spread_mean') >= float(stable_override_cfg.get('tail_spread_mean_gte', 115.0)),
            _feature_value(features, 'angle_speed_median') >= float(stable_override_cfg.get('angle_speed_median_gte', 18.0)),
            _feature_value(features, 'spread_peak') >= float(stable_override_cfg.get('spread_peak_gte', 140.0)),
        )
        strong_high_voltage_override = _count_true(override_conditions) >= int(stable_override_cfg.get('min_condition_count', 3))

    boundary_candidate_flag = 1
    if stable_clear_flag and not strong_high_voltage_override:
        boundary_candidate_flag = 0
        boundary_score = 0.0
        side_bonus = 0.0
        if side_features.get('boundary_side') in {'stable_side', 'stable_side_high_voltage'}:
            side_features['boundary_side'] = 'clear_stable'
    if unstable_clear_flag:
        boundary_candidate_flag = 0
        boundary_score = 0.0
        side_bonus = 0.0
        side_features['unstable_side_boundary_flag'] = 0
        side_features['unstable_side_bonus'] = 0.0
        side_features['boundary_side'] = 'clear_unstable'

    return {
        'g_v': g_v,
        'g_r': g_r,
        'g_s': g_s,
        'g_w': g_w,
        'raw_boundary_score': raw_score,
        'penalized_boundary_score': penalized_score,
        'side_bonus': side_bonus,
        'stable_clear_flag': stable_clear_flag,
        'unstable_clear_flag': unstable_clear_flag,
        'boundary_score': boundary_score,
        'boundary_candidate_flag': boundary_candidate_flag,
        'dynamic_unstable_flag': dynamic_unstable_flag,
        'strong_high_voltage_override': int(strong_high_voltage_override),
        **side_features,
    }


def extract_sample_features(path: str | Path, input_root: str | Path, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    feature_cfg = config['features']
    sample = load_sample(path, invalid_value=float(feature_cfg['invalid_value']))

    times = sample['times']
    voltages = sample['voltages']
    angles = sample['angles']

    row: Dict[str, Any] = {
        'file': Path(path).resolve().relative_to(Path(input_root).resolve()).as_posix(),
        'dataset_name': canonical_dataset_name(dataset_name),
        'original_label': sample['label_name'],
        'original_label_value': sample['label_value'],
        'T': int(times.shape[0]),
        'Nv': int(voltages.shape[1]),
        'Ng': int(angles.shape[1]),
    }
    row.update(
        compute_voltage_features(
            times=times,
            voltages=voltages,
            tail_window_sec=float(feature_cfg['tail_window_sec']),
            final_window_sec=float(feature_cfg['final_window_sec']),
        )
    )
    row.update(
        compute_angle_features(
            times=times,
            angles=angles,
            tail_window_sec=float(feature_cfg['tail_window_sec']),
            relative_angle_mode=str(feature_cfg['relative_angle_mode']),
        )
    )
    row.update(apply_boundary_score(row, config))
    return row


def load_sample_for_plot(path: str | Path, config: Dict[str, Any]) -> Dict[str, Any]:
    feature_cfg = config['features']
    sample = load_sample(path, invalid_value=float(feature_cfg['invalid_value']))
    sample['angles_rel'] = build_relative_angles(sample['angles'], str(feature_cfg['relative_angle_mode']))
    return sample
