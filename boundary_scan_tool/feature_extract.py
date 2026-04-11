from __future__ import annotations

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
    lowered = str(dataset_dir_name).lower()
    if lowered.startswith('36'):
        return '36'
    if lowered.startswith('37'):
        return '37'
    if lowered.startswith('74'):
        return '74'
    return str(dataset_dir_name)


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


def _middle_window_mask(times: np.ndarray, tail_window_sec: float, middle_window_sec: float) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0, dtype=bool)
    end_time = float(times[-1]) - float(tail_window_sec)
    start_time = end_time - float(middle_window_sec)
    return np.isfinite(times) & (times >= start_time) & (times < end_time)


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


def _count_threshold_reentry(series: np.ndarray, threshold: float, mode: str = 'below') -> int:
    values = np.asarray(series, dtype=np.float64)
    if mode == 'below':
        state = np.isfinite(values) & (values < threshold)
    else:
        state = np.isfinite(values) & (values > threshold)
    if state.size <= 1:
        return 0
    transitions = (~state[:-1]) & state[1:]
    return int(np.count_nonzero(transitions))


def _sign_change_density(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0.0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    return float(np.count_nonzero(signs[1:] * signs[:-1] < 0) / max(signs.size - 1, 1))


def _gini_like(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    mean_value = float(np.mean(values))
    if mean_value <= 1e-8:
        return 0.0
    diff_sum = np.abs(values[:, None] - values[None, :]).sum()
    return float(diff_sum / (2.0 * values.size * values.size * mean_value))


def _channel_peak_to_peak(window: np.ndarray) -> np.ndarray:
    if window.size == 0:
        return np.array([], dtype=np.float64)
    return np.nanmax(window, axis=0) - np.nanmin(window, axis=0)


def compute_voltage_features(times: np.ndarray, voltages: np.ndarray, tail_window_sec: float, final_window_sec: float, stable_cfg: Dict[str, Any]) -> Dict[str, float]:
    tail_mask = _window_mask(times, tail_window_sec)
    final_mask = _window_mask(times, final_window_sec)
    min_voltage_t = np.nanmin(voltages, axis=1)
    mean_voltage_t = np.nanmean(voltages, axis=1)
    final_voltages = voltages[final_mask]
    final_channel_mean = np.nanmean(final_voltages, axis=0) if final_voltages.size else np.array([], dtype=np.float64)
    tail_low_th = float(stable_cfg.get('low_voltage_threshold', 0.9))
    tail_low_voltage_reentry_count = _count_threshold_reentry(min_voltage_t[tail_mask], tail_low_th, mode='below')
    recover_ratio = _safe_mean((final_voltages >= tail_low_th).astype(np.float64))
    voltage_rebound_instability_score = float(np.clip(
        0.45 * min(1.0, tail_low_voltage_reentry_count / max(float(stable_cfg.get('reentry_norm', 3.0)), 1.0))
        + 0.30 * min(1.0, _safe_std(min_voltage_t[tail_mask]) / max(float(stable_cfg.get('tail_min_voltage_std_norm', 0.03)), 1e-6))
        + 0.25 * min(1.0, (1.0 - (recover_ratio if np.isfinite(recover_ratio) else 0.0)) / max(float(stable_cfg.get('non_recovery_norm', 0.5)), 1e-6)),
        0.0,
        1.0,
    ))
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
        'tail_low_voltage_reentry_count': float(tail_low_voltage_reentry_count),
        'voltage_rebound_instability_score': voltage_rebound_instability_score,
    }


def compute_angle_features(times: np.ndarray, angles: np.ndarray, tail_window_sec: float, relative_angle_mode: str, stable_cfg: Dict[str, Any]) -> Dict[str, Any]:
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
    spread_reentry_count = _count_threshold_reentry(tail_spread, float(stable_cfg.get('spread_danger_threshold', 120.0)), mode='above')
    slope = _linear_slope(tail_times, tail_spread)
    return {
        'angles_rel': angles_rel,
        'spread_t': spread_t,
        'spread_peak': _safe_max(spread_t),
        'tail_spread_mean': _safe_mean(tail_spread),
        'tail_spread_std': _safe_std(tail_spread),
        'tail_spread_slope': slope,
        'tail_spread_slope_abs': abs(slope) if np.isfinite(slope) else float('nan'),
        'angle_speed_median': float(np.nanmedian(angle_speed_abs)) if angle_speed_abs.size else float('nan'),
        'angle_speed_p90': float(np.nanpercentile(angle_speed_abs, 90)) if angle_speed_abs.size else float('nan'),
        'invalid_angle_ratio': invalid_angle_ratio,
        'first_invalid_time': first_invalid_time,
        'spread_reentry_count': float(spread_reentry_count),
    }


def compute_local_oscillation_features(times: np.ndarray, angles_rel: np.ndarray, osc_cfg: Dict[str, Any], tail_window_sec: float) -> Dict[str, float]:
    tail_mask = _window_mask(times, tail_window_sec)
    middle_mask = _middle_window_mask(times, tail_window_sec, float(osc_cfg.get('middle_window_sec', 2.0)))
    tail_angles = angles_rel[tail_mask]
    middle_angles = angles_rel[middle_mask]
    tail_amp_per_channel = _channel_peak_to_peak(tail_angles)
    middle_amp_per_channel = _channel_peak_to_peak(middle_angles)
    if middle_amp_per_channel.size == 0 and tail_amp_per_channel.size:
        middle_amp_per_channel = np.full_like(tail_amp_per_channel, np.nan)

    finite_tail = tail_amp_per_channel[np.isfinite(tail_amp_per_channel)]
    if finite_tail.size == 0:
        finite_tail = np.array([0.0], dtype=np.float64)
    sorted_tail = np.sort(finite_tail)
    tail_amp_top1 = float(sorted_tail[-1]) if sorted_tail.size >= 1 else 0.0
    tail_amp_top2 = float(sorted_tail[-2]) if sorted_tail.size >= 2 else tail_amp_top1
    tail_amp_mean = float(np.mean(finite_tail)) if finite_tail.size else 0.0
    tail_amp_median = float(np.median(finite_tail)) if finite_tail.size else 0.0
    top1_idx = int(np.nanargmax(tail_amp_per_channel)) if tail_amp_per_channel.size and np.isfinite(tail_amp_per_channel).any() else -1
    mid_top1 = float(middle_amp_per_channel[top1_idx]) if 0 <= top1_idx < middle_amp_per_channel.size and np.isfinite(middle_amp_per_channel[top1_idx]) else float('nan')
    mid_mean = _safe_mean(middle_amp_per_channel)
    tail_sign_change_density = float(np.mean([_sign_change_density(channel_values) for channel_values in tail_angles.T])) if tail_angles.size else 0.0
    decay_ratio_top1 = tail_amp_top1 / max(mid_top1 if np.isfinite(mid_top1) else np.nan, 1e-6) if np.isfinite(mid_top1) else float('nan')
    decay_ratio_mean = tail_amp_mean / max(mid_mean if np.isfinite(mid_mean) else np.nan, 1e-6) if np.isfinite(mid_mean) else float('nan')
    oscillation_persistence_score = float(np.clip(
        0.45 * min(1.0, (decay_ratio_mean if np.isfinite(decay_ratio_mean) else 0.0) / max(float(osc_cfg.get('decay_ratio_norm', 1.2)), 1e-6))
        + 0.30 * min(1.0, tail_sign_change_density / max(float(osc_cfg.get('sign_change_density_norm', 0.35)), 1e-6))
        + 0.25 * min(1.0, tail_amp_top1 / max(float(osc_cfg.get('tail_amp_norm', 30.0)), 1e-6)),
        0.0,
        1.0,
    ))
    return {
        'tail_amp_top1': tail_amp_top1,
        'tail_amp_top2': tail_amp_top2,
        'tail_amp_top1_ratio': tail_amp_top1 / max(tail_amp_mean, 1e-6),
        'tail_amp_top2_ratio': tail_amp_top2 / max(tail_amp_mean, 1e-6),
        'large_amp_channel_count_20': float(np.count_nonzero(finite_tail >= float(osc_cfg.get('large_amp_threshold_1', 20.0)))),
        'large_amp_channel_count_30': float(np.count_nonzero(finite_tail >= float(osc_cfg.get('large_amp_threshold_2', 30.0)))),
        'amp_std_across_channels': _safe_std(finite_tail),
        'amp_gini_like': _gini_like(finite_tail),
        'top1_minus_median_amp': tail_amp_top1 - tail_amp_median,
        'decay_ratio_top1': decay_ratio_top1,
        'decay_ratio_mean': decay_ratio_mean,
        'tail_sign_change_density': tail_sign_change_density,
        'oscillation_persistence_score': oscillation_persistence_score,
    }


def extract_sample_features(path: str | Path, input_root: str | Path, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    feature_cfg = config['features']
    stable_cfg = config.get('features_stable_side', {})
    osc_cfg = config.get('features_local_oscillation', {})
    sample = load_sample(path, invalid_value=float(feature_cfg['invalid_value']))
    times = sample['times']
    voltages = sample['voltages']
    angles = sample['angles']
    sample_path = Path(path).resolve()
    input_root = Path(input_root).resolve()
    angle_features = compute_angle_features(times, angles, float(feature_cfg['tail_window_sec']), str(feature_cfg['relative_angle_mode']), stable_cfg)
    row: Dict[str, Any] = {
        'file': sample_path.relative_to(input_root).as_posix(),
        'sample_name': sample_path.name,
        'sample_stem': sample_path.stem,
        'source_parent_dir_name': sample_path.parent.name,
        'source_relative_dir': sample_path.parent.relative_to(input_root).as_posix(),
        'source_abs_path': str(sample_path),
        'dataset_name': canonical_dataset_name(dataset_name),
        'dataset_dir_name': str(dataset_name),
        'original_label': sample['label_name'],
        'original_label_value': sample['label_value'],
        'T': int(times.shape[0]),
        'Nv': int(voltages.shape[1]),
        'Ng': int(angles.shape[1]),
    }
    row.update(compute_voltage_features(times, voltages, float(feature_cfg['tail_window_sec']), float(feature_cfg['final_window_sec']), stable_cfg))
    row.update({key: value for key, value in angle_features.items() if key not in {'angles_rel', 'spread_t'}})
    row.update(compute_local_oscillation_features(times, angle_features['angles_rel'], osc_cfg, float(feature_cfg['tail_window_sec'])))
    return row


def load_sample_for_plot(path: str | Path, config: Dict[str, Any]) -> Dict[str, Any]:
    feature_cfg = config['features']
    sample = load_sample(path, invalid_value=float(feature_cfg['invalid_value']))
    angle_features = compute_angle_features(sample['times'], sample['angles'], float(feature_cfg['tail_window_sec']), str(feature_cfg['relative_angle_mode']), config.get('features_stable_side', {}))
    return {
        'times': sample['times'],
        'voltages': sample['voltages'],
        'angles_rel': angle_features['angles_rel'],
        'spread_t': angle_features['spread_t'],
        'min_voltage_t': np.nanmin(sample['voltages'], axis=1),
        'label_name': sample['label_name'],
    }
