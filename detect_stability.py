from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import AutoLocator, FixedLocator

matplotlib.use("Agg")

LOGGER = logging.getLogger("detect_stability")


@dataclass
class StabilityConfig:
    tail_window_sec: float = 5.0
    final_window_sec: float = 1.0
    pre_invalid_window_sec: float = 2.0
    spread_stable_threshold: float = 120.0
    spread_unstable_threshold: float = 360.0
    spread_slope_threshold: float = 15.0
    angle_speed_threshold: float = 20.0
    slip_threshold: float = 360.0
    large_relative_angle_threshold: float = 120.0
    extreme_relative_angle_threshold: float = 180.0
    large_relative_angle_duration_threshold_sec: float = 3.0
    single_machine_reseparation_tail_spread_threshold: float = 100.0
    single_machine_reseparation_amp_max_threshold: float = 40.0
    single_machine_extreme_peak_threshold: float = 250.0
    single_machine_extreme_tail_spread_threshold: float = 120.0
    single_machine_extreme_tail_spread_peak_threshold: float = 150.0
    single_machine_extreme_tail_peak_relative_threshold: float = 100.0
    single_machine_extreme_spread_std_threshold: float = 20.0
    single_machine_extreme_amp_max_threshold: float = 45.0
    oscillation_tail_amplitude_threshold: float = 12.0
    oscillation_tail_amplitude_max_threshold: float = 60.0
    oscillation_decay_ratio_threshold: float = 0.85
    oscillation_angle_speed_threshold: float = 30.0
    oscillation_ampmax_tail_spread_threshold: float = 120.0
    oscillation_ampmax_peak_relative_threshold: float = 110.0
    oscillation_ampmax_angle_speed_threshold: float = 45.0
    oscillation_ampmax_spread_std_threshold: float = 30.0
    oscillation_spread_std_threshold: float = 25.0
    oscillation_spread_tail_spread_threshold: float = 120.0
    oscillation_spread_peak_relative_threshold: float = 110.0
    oscillation_spread_amp_max_threshold: float = 60.0
    oscillation_speed_tail_spread_threshold: float = 110.0
    oscillation_speed_peak_relative_threshold: float = 90.0
    oscillation_speed_amp_max_threshold: float = 30.0
    high_amp_oscillation_tail_amplitude_mean_threshold: float = 35.0
    high_amp_oscillation_tail_spread_threshold: float = 110.0
    high_amp_oscillation_angle_speed_threshold: float = 40.0
    high_amp_oscillation_decay_ratio_threshold: float = 0.9
    high_amp_oscillation_spread_slope_threshold: float = 2.5
    high_amp_oscillation_peak_relative_threshold: float = 100.0
    high_amp_oscillation_amp_max_threshold: float = 55.0
    moderate_oscillation_tail_amplitude_max_threshold: float = 10.0
    moderate_oscillation_tail_amplitude_mean_upper_threshold: float = 10.0
    moderate_oscillation_angle_speed_threshold: float = 15.0
    moderate_oscillation_tail_spread_threshold: float = 80.0
    moderate_oscillation_peak_relative_threshold: float = 80.0
    large_tail_spread_strong_threshold: float = 125.0
    large_tail_peak_relative_strong_threshold: float = 110.0
    large_tail_angle_speed_strong_threshold: float = 15.0
    reseparating_tail_spread_threshold: float = 80.0
    reseparating_tail_spread_slope_threshold: float = 8.0
    reseparating_peak_relative_threshold: float = 150.0
    reseparating_tail_amp_max_threshold: float = 50.0
    reseparating_decay_ratio_threshold: float = 0.9
    growing_tail_spread_strong_threshold: float = 125.0
    growing_tail_spread_slope_threshold: float = 15.0
    growing_tail_peak_relative_threshold: float = 150.0
    growing_tail_duration_large_angle_threshold_sec: float = 2.0
    invalid_ratio_threshold: float = 0.25
    voltage_recovery_threshold_0_9: float = 0.85
    voltage_recovery_threshold_0_95: float = 0.65
    severe_low_voltage_threshold_0_8: float = 0.8
    extreme_low_voltage_threshold_0_7: float = 0.7
    long_term_low_voltage_threshold_0_85: float = 0.85
    persistent_local_low_voltage_threshold: float = 0.85
    overvoltage_threshold_1_12: float = 1.12
    persistent_high_voltage_bus_count_threshold: float = 2.0
    mixed_voltage_split_duration_below_0_85_threshold_sec: float = 2.5
    mixed_voltage_split_duration_above_1_12_threshold_sec: float = 8.0
    mixed_voltage_split_recovered_0_95_threshold: float = 0.92
    mixed_voltage_split_low_bus_threshold: float = 0.5
    mixed_voltage_split_high_bus_threshold: float = 0.5
    voltage_oscillation_std_threshold: float = 0.012
    voltage_oscillation_low_bus_threshold: float = 0.5
    voltage_oscillation_tail_min_threshold: float = 0.93
    voltage_oscillation_duration_below_0_9_threshold_sec: float = 5.0
    voltage_oscillation_strict_std_threshold: float = 0.025
    voltage_oscillation_strict_low_bus_threshold: float = 1.0
    voltage_oscillation_strict_tail_min_threshold: float = 0.94
    severe_voltage_oscillation_std_threshold: float = 0.02
    severe_voltage_oscillation_duration_below_0_85_threshold_sec: float = 3.0
    severe_voltage_oscillation_duration_below_0_8_threshold_sec: float = 2.0
    repeated_voltage_dip_std_threshold: float = 0.04
    repeated_voltage_dip_duration_below_0_85_threshold_sec: float = 4.0
    repeated_voltage_dip_recovered_0_95_threshold: float = 0.96
    tail_repeated_voltage_dip_std_threshold: float = 0.025
    tail_repeated_voltage_dip_min_threshold: float = 0.85
    tail_repeated_voltage_dip_duration_threshold_sec: float = 0.4
    tail_repeated_voltage_dip_recovered_0_95_threshold: float = 0.8
    tail_worst_bus_oscillation_std_threshold: float = 0.03
    tail_worst_bus_dip_threshold: float = 0.88
    tail_worst_bus_dip_duration_threshold_sec: float = 0.25
    persistent_single_bus_low_voltage_tail_min_threshold: float = 0.87
    persistent_single_bus_low_voltage_tail_duration_threshold_sec: float = 0.8
    persistent_single_bus_low_voltage_low_bus_threshold: float = 1.0
    moderate_voltage_oscillation_std_threshold: float = 0.02
    moderate_voltage_oscillation_duration_below_0_85_threshold_sec: float = 2.5
    moderate_voltage_oscillation_duration_below_0_8_threshold_sec: float = 1.5
    moderate_voltage_oscillation_recovered_0_95_threshold: float = 0.97
    moderate_overvoltage_duration_threshold_sec: float = 12.0
    moderate_overvoltage_high_bus_threshold: float = 1.2
    moderate_overvoltage_recovered_0_95_threshold: float = 0.90
    moderate_overvoltage_low_bus_threshold: float = 1.0
    coupled_angle_voltage_oscillation_amp_threshold: float = 35.0
    coupled_angle_voltage_oscillation_speed_threshold: float = 50.0
    coupled_angle_voltage_oscillation_peak_threshold: float = 90.0
    coupled_angle_voltage_oscillation_voltage_std_threshold: float = 0.022
    slow_recovery_duration_below_0_85_threshold_sec: float = 4.0
    slow_recovery_duration_below_0_8_threshold_sec: float = 2.0
    slow_recovery_last_below_0_9_threshold_sec: float = 12.0
    persistent_multi_bus_low_voltage_duration_below_0_85_threshold_sec: float = 4.0
    persistent_multi_bus_low_voltage_recovered_0_95_threshold: float = 0.75
    persistent_multi_bus_low_voltage_low_bus_threshold: float = 3.0
    persistent_multi_bus_low_voltage_tail_min_threshold: float = 0.88
    undervoltage_area_threshold: float = 0.35
    non_recovery_ratio_threshold: float = 0.20
    duration_below_0_9_threshold_sec: float = 6.0
    duration_below_0_85_threshold_sec: float = 3.0
    duration_below_0_8_threshold_sec: float = 2.0
    duration_below_0_7_threshold_sec: float = 0.5
    duration_above_1_12_threshold_sec: float = 3.0
    angle_weight: float = 0.60
    voltage_weight: float = 0.40
    total_instability_threshold: float = 0.55
    invalid_value: float = -99999.0
    unwrap_angles: bool = True
    ref_mode: str = "median"
    recursive: bool = True
    label_unstable_value: int = 1
    label_stable_value: int = 0
    plot_max_angle_curves: int = 9999
    plot_max_voltage_curves: int = 9999
    figure_width: float = 14.0
    figure_height: float = 10.0
    plot_only_unstable: bool = True
    mismatch_export_enabled: bool = True
    mismatch_export_plot: bool = True
    workers: int = 0


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def configure_matplotlib_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_batch_outputs(output_dir: Path) -> None:
    managed_paths = [
        output_dir / "mismatch_results.csv",
        output_dir / "abnormal_results.csv",
        output_dir / "mismatches_plots",
        output_dir / "plots",
        output_dir / "mismatches",
        output_dir / "mismatches_json",
        output_dir / "summary.csv",
        output_dir / "detailed_results.jsonl",
        output_dir / "metrics.json",
        output_dir / "confusion_matrix.png",
    ]
    for path in managed_paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def load_config(config_path: Path | None) -> StabilityConfig:
    config = StabilityConfig()
    if config_path is None:
        return config
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    valid_keys = {field.name for field in fields(StabilityConfig)}
    for key, value in raw.items():
        if key in valid_keys:
            setattr(config, key, value)
        else:
            LOGGER.warning("忽略未知配置项: %s", key)
    return config


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return None


def round_or_none(value: Any, digits: int = 6) -> float | None:
    number = safe_float(value)
    return round(number, digits) if number is not None else None


def serialize_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: serialize_data(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [serialize_data(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def label_num_to_en(label_value: int | None, config: StabilityConfig) -> str | None:
    if label_value is None:
        return None
    return "unstable" if int(label_value) == config.label_unstable_value else "stable"


def label_en_to_cn(label: str | None) -> str:
    if label == "unstable":
        return "失稳"
    if label == "stable":
        return "稳定"
    return "未知"


def label_en_to_num(label: str, config: StabilityConfig) -> int:
    return config.label_unstable_value if label == "unstable" else config.label_stable_value


def load_npy_sample(path: Path) -> dict[str, Any]:
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"无法加载样本: {exc}") from exc

    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        data = obj.item()
    elif isinstance(obj, dict):
        data = obj
    else:
        raise ValueError(f"样本不是标量 object dict，实际类型: {type(obj)!r}")

    if not isinstance(data, dict):
        raise ValueError(f"样本内容不是 dict，实际类型: {type(data)!r}")
    return data


def unwrap_1d_ignore_nan(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    valid = np.isfinite(values)
    if valid.sum() <= 1:
        return values
    indices = np.where(valid)[0]
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    result = values.copy()
    for group in groups:
        if len(group) <= 1:
            continue
        radians = np.deg2rad(values[group])
        result[group] = np.rad2deg(np.unwrap(radians))
    return result


def unwrap_matrix_ignore_nan(values: np.ndarray) -> np.ndarray:
    return np.column_stack([unwrap_1d_ignore_nan(values[:, idx]) for idx in range(values.shape[1])])


def rowwise_reference(values: np.ndarray, mode: str = "median") -> np.ndarray:
    reference = np.full((values.shape[0], 1), np.nan, dtype=float)
    valid_rows = ~np.all(np.isnan(values), axis=1)
    if not np.any(valid_rows):
        return reference
    if mode == "median":
        reference[valid_rows, 0] = np.nanmedian(values[valid_rows], axis=1)
    elif mode == "mean":
        reference[valid_rows, 0] = np.nanmean(values[valid_rows], axis=1)
    else:
        raise ValueError(f"不支持的参考模式: {mode}")
    return reference


def nanmax_minus_nanmin(values: np.ndarray) -> np.ndarray:
    spread = np.full(values.shape[0], np.nan, dtype=float)
    valid_rows = ~np.all(np.isnan(values), axis=1)
    if not np.any(valid_rows):
        return spread
    vmax = np.nanmax(values[valid_rows], axis=1)
    vmin = np.nanmin(values[valid_rows], axis=1)
    spread[valid_rows] = vmax - vmin
    return spread


def compute_tail_mask(times: np.ndarray, window_sec: float) -> np.ndarray:
    if times.size == 0:
        return np.array([], dtype=bool)
    start = max(float(times[-1]) - window_sec, float(times[0]))
    return times >= start


def linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() < 2:
        return 0.0
    x = times[mask].astype(float)
    y = values[mask].astype(float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    if denom <= 0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def line_fit(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope = linear_slope(times, values)
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() == 0:
        return slope, 0.0
    intercept = float(np.nanmean(values[mask]) - slope * np.nanmean(times[mask]))
    return slope, intercept


def focused_tick_values(start: float, end: float, step: float) -> np.ndarray:
    if not np.isfinite(start) or not np.isfinite(end) or not np.isfinite(step) or step <= 0 or end <= start:
        return np.array([], dtype=float)
    first = math.ceil(start / step) * step
    last = math.floor(end / step) * step
    if last < first:
        return np.array([], dtype=float)
    count = int(round((last - first) / step)) + 1
    return np.round(first + np.arange(count, dtype=float) * step, 6)


def resolve_worker_count(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def array_nanmax_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def rowwise_nanmax_abs(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=float)
    valid_rows = ~np.all(np.isnan(values), axis=1)
    if not np.any(valid_rows):
        return result
    result[valid_rows] = np.nanmax(np.abs(values[valid_rows]), axis=1)
    return result


def per_machine_window_amplitude(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or mask.size != values.shape[0] or np.sum(mask) < 2:
        return np.full(values.shape[1], np.nan, dtype=float)
    window = values[mask]
    valid_cols = ~np.all(np.isnan(window), axis=0)
    amplitude = np.full(values.shape[1], np.nan, dtype=float)
    if not np.any(valid_cols):
        return amplitude
    amplitude[valid_cols] = 0.5 * (np.nanmax(window[:, valid_cols], axis=0) - np.nanmin(window[:, valid_cols], axis=0))
    return amplitude


def duration_condition(times: np.ndarray, mask: np.ndarray) -> float:
    if times.size <= 1 or mask.size != times.size:
        return 0.0
    durations = np.diff(times)
    active = mask[:-1] | mask[1:]
    return float(np.sum(durations[active]))


def normalize_value(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 1.0
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, value in components.items():
        weight = weights.get(key, 1.0)
        numerator += weight * float(np.clip(value, 0.0, 1.0))
        denominator += weight
    return float(numerator / denominator) if denominator else 0.0


def safe_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} 为空")
    return array.astype(float)


def preprocess_angles(
    angles: np.ndarray,
    times: np.ndarray,
    invalid_value: float = -99999.0,
    unwrap: bool = True,
    ref_mode: str = "median",
) -> dict[str, Any]:
    raw_angles = safe_array(angles, "angles")
    times = safe_array(times, "times")
    if raw_angles.ndim != 2:
        raise ValueError(f"angles 维度异常: {raw_angles.shape}")
    if times.ndim != 1:
        raise ValueError(f"times 维度异常: {times.shape}")
    if raw_angles.shape[0] != times.shape[0]:
        raise ValueError(f"angles 与 times 长度不一致: {raw_angles.shape[0]} vs {times.shape[0]}")

    cleaned_angles = raw_angles.copy()
    invalid_mask = cleaned_angles == invalid_value
    cleaned_angles[invalid_mask] = np.nan

    monotonic = bool(np.all(np.diff(times) > 0))
    effective_times = times.copy()
    if not monotonic:
        LOGGER.warning("times 非严格单调，使用索引序列进行趋势计算")
        effective_times = np.arange(times.shape[0], dtype=float)

    if unwrap:
        unwrapped_angles = unwrap_matrix_ignore_nan(cleaned_angles)
    else:
        unwrapped_angles = cleaned_angles.copy()

    reference = rowwise_reference(unwrapped_angles, ref_mode)

    relative_angles = unwrapped_angles - reference
    has_invalid_angles = bool(np.isnan(cleaned_angles).any())
    first_invalid_time = None
    if has_invalid_angles:
        invalid_rows = np.where(np.isnan(cleaned_angles).any(axis=1))[0]
        if invalid_rows.size:
            first_invalid_time = float(times[invalid_rows[0]])

    return {
        "raw_angles": raw_angles,
        "cleaned_angles": cleaned_angles,
        "unwrapped_angles": unwrapped_angles,
        "relative_angles": relative_angles,
        "times": times,
        "effective_times": effective_times,
        "invalid_mask": invalid_mask,
        "has_invalid_angles": has_invalid_angles,
        "first_invalid_time": first_invalid_time,
        "times_monotonic": monotonic,
    }


def preprocess_voltages(voltages: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    voltages = safe_array(voltages, "voltages")
    times = safe_array(times, "times")
    if voltages.ndim != 2:
        raise ValueError(f"voltages 维度异常: {voltages.shape}")
    if times.ndim != 1:
        raise ValueError(f"times 维度异常: {times.shape}")
    if voltages.shape[0] != times.shape[0]:
        raise ValueError(f"voltages 与 times 长度不一致: {voltages.shape[0]} vs {times.shape[0]}")

    return {
        "voltages": voltages,
        "times": times,
        "voltage_mean_t": np.nanmean(voltages, axis=1),
        "voltage_min_t": np.nanmin(voltages, axis=1),
    }


def extract_angle_features(relative_angles: np.ndarray, times: np.ndarray, config: StabilityConfig) -> dict[str, Any]:
    spread_t = nanmax_minus_nanmin(relative_angles)
    max_pairwise_sep_t = spread_t.copy()
    max_abs_relative_t = rowwise_nanmax_abs(relative_angles)
    tail_mask = compute_tail_mask(times, config.tail_window_sec)
    tail_start = float(times[tail_mask][0]) if tail_mask.any() else float(times[0])
    prev_tail_mask = (times >= max(tail_start - config.tail_window_sec, float(times[0]))) & (times < tail_start)
    tail_spread = spread_t[tail_mask]
    tail_times = times[tail_mask]

    diffs = np.diff(relative_angles, axis=0)
    time_diffs = np.diff(times)
    finite_dt = np.where(time_diffs > 0, time_diffs, np.nan)
    angle_speed_matrix = np.abs(diffs / finite_dt[:, None])
    tail_speed_mask = tail_mask[1:] if tail_mask.size > 1 else np.array([], dtype=bool)
    tail_speed_values = angle_speed_matrix[tail_speed_mask]
    tail_angle_speed = float(np.nanmedian(tail_speed_values)) if tail_speed_values.size else 0.0

    invalid_t = np.isnan(relative_angles).mean(axis=1)
    invalid_tail_ratio = float(np.nanmean(invalid_t[tail_mask])) if tail_mask.any() else float(np.nanmean(invalid_t))

    tail_spread_mean = float(np.nanmean(tail_spread)) if tail_spread.size else float("nan")
    tail_spread_std = float(np.nanstd(tail_spread)) if tail_spread.size else float("nan")
    tail_spread_slope = linear_slope(tail_times, tail_spread) if tail_spread.size else 0.0
    tail_spread_peak = array_nanmax_or_nan(tail_spread)

    max_sep = float(np.nanmax(max_pairwise_sep_t)) if np.isfinite(max_pairwise_sep_t).any() else float("nan")
    peak_max_abs_relative = array_nanmax_or_nan(max_abs_relative_t)
    duration_large_relative_angle = duration_condition(times, max_abs_relative_t >= config.large_relative_angle_threshold)
    duration_extreme_relative_angle = duration_condition(times, max_abs_relative_t >= config.extreme_relative_angle_threshold)
    tail_peak_max_abs_relative = array_nanmax_or_nan(max_abs_relative_t[tail_mask])
    tail_duration_large_relative_angle = duration_condition(times[tail_mask], max_abs_relative_t[tail_mask] >= config.large_relative_angle_threshold) if tail_mask.any() else 0.0
    tail_duration_extreme_relative_angle = duration_condition(times[tail_mask], max_abs_relative_t[tail_mask] >= config.extreme_relative_angle_threshold) if tail_mask.any() else 0.0
    tail_amp_per_machine = per_machine_window_amplitude(relative_angles, tail_mask)
    prev_tail_amp_per_machine = per_machine_window_amplitude(relative_angles, prev_tail_mask)
    tail_oscillation_amplitude_mean = float(np.nanmean(tail_amp_per_machine)) if np.isfinite(tail_amp_per_machine).any() else 0.0
    tail_oscillation_amplitude_max = float(np.nanmax(tail_amp_per_machine)) if np.isfinite(tail_amp_per_machine).any() else 0.0
    prev_tail_oscillation_amplitude_mean = float(np.nanmean(prev_tail_amp_per_machine)) if np.isfinite(prev_tail_amp_per_machine).any() else 0.0
    oscillation_decay_ratio = float(
        tail_oscillation_amplitude_mean / max(prev_tail_oscillation_amplitude_mean, 1e-6)
    ) if tail_oscillation_amplitude_mean > 0 else 0.0
    slip_score = float(
        np.clip(
            max(
                normalize_value(tail_spread_peak, config.slip_threshold * 0.75, config.slip_threshold),
                normalize_value(tail_spread_mean, config.spread_unstable_threshold * 0.8, config.spread_unstable_threshold * 1.3),
            ),
            0.0,
            1.0,
        )
    )

    divergence_score = float(
        np.clip(
            0.45 * normalize_value(tail_spread_mean, config.spread_stable_threshold, config.spread_unstable_threshold)
            + 0.35 * normalize_value(tail_spread_slope, config.spread_slope_threshold * 0.2, config.spread_slope_threshold)
            + 0.20 * normalize_value(tail_angle_speed, config.angle_speed_threshold * 0.2, config.angle_speed_threshold),
            0.0,
            1.0,
        )
    )

    pre_invalid_divergence = 0.0
    invalid_rows = np.where(np.isnan(relative_angles).any(axis=1))[0]
    if invalid_rows.size:
        first_invalid_idx = int(invalid_rows[0])
        if first_invalid_idx > 1:
            first_invalid_time = times[first_invalid_idx]
            pre_mask = (times < first_invalid_time) & (times >= first_invalid_time - config.pre_invalid_window_sec)
            pre_spread = spread_t[pre_mask]
            pre_time = times[pre_mask]
            if pre_spread.size:
                pre_invalid_divergence = max(
                    float(np.nanmean(pre_spread)),
                    float(linear_slope(pre_time, pre_spread) * config.pre_invalid_window_sec),
                )

    return {
        "spread_t": spread_t,
        "max_pairwise_sep_t": max_pairwise_sep_t,
        "tail_spread_mean": tail_spread_mean,
        "tail_spread_std": tail_spread_std,
        "tail_spread_slope": float(tail_spread_slope),
        "tail_angle_speed": tail_angle_speed,
        "divergence_score": divergence_score,
        "slip_score": slip_score,
        "peak_max_abs_relative": peak_max_abs_relative,
        "duration_large_relative_angle": duration_large_relative_angle,
        "duration_extreme_relative_angle": duration_extreme_relative_angle,
        "tail_peak_max_abs_relative": tail_peak_max_abs_relative,
        "tail_duration_large_relative_angle": tail_duration_large_relative_angle,
        "tail_duration_extreme_relative_angle": tail_duration_extreme_relative_angle,
        "tail_spread_peak": tail_spread_peak,
        "tail_oscillation_amplitude_mean": tail_oscillation_amplitude_mean,
        "tail_oscillation_amplitude_max": tail_oscillation_amplitude_max,
        "prev_tail_oscillation_amplitude_mean": prev_tail_oscillation_amplitude_mean,
        "oscillation_decay_ratio": oscillation_decay_ratio,
        "invalid_tail_ratio": invalid_tail_ratio,
        "pre_invalid_divergence": float(pre_invalid_divergence),
        "_tail_mask": tail_mask,
        "_tail_trend": line_fit(tail_times, tail_spread) if tail_spread.size else (0.0, 0.0),
    }


def extract_voltage_features(voltages: np.ndarray, times: np.ndarray, config: StabilityConfig) -> dict[str, Any]:
    voltage_mean_t = np.nanmean(voltages, axis=1)
    voltage_min_t = np.nanmin(voltages, axis=1)
    voltage_max_t = np.nanmax(voltages, axis=1)
    tail_mask = compute_tail_mask(times, config.tail_window_sec)
    final_mask = compute_tail_mask(times, config.final_window_sec)

    tail_voltage_mean = float(np.nanmean(voltage_mean_t[tail_mask])) if tail_mask.any() else float(np.nanmean(voltage_mean_t))
    tail_voltage_min = float(np.nanmean(voltage_min_t[tail_mask])) if tail_mask.any() else float(np.nanmean(voltage_min_t))
    tail_voltage_max = float(np.nanmean(voltage_max_t[tail_mask])) if tail_mask.any() else float(np.nanmean(voltage_max_t))
    tail_voltage_std = float(np.nanstd(voltage_mean_t[tail_mask])) if tail_mask.any() else float(np.nanstd(voltage_mean_t))
    tail_voltage_slope = linear_slope(times[tail_mask], voltage_mean_t[tail_mask]) if tail_mask.any() else 0.0
    tail_voltage_min_min = float(np.nanmin(voltage_min_t[tail_mask])) if tail_mask.any() else float(np.nanmin(voltage_min_t))
    tail_voltage_min_std = float(np.nanstd(voltage_min_t[tail_mask])) if tail_mask.any() else float(np.nanstd(voltage_min_t))
    if tail_mask.any() and times.size > 1:
        tail_duration_below_0_85 = duration_condition(times[tail_mask], voltage_min_t[tail_mask] < config.long_term_low_voltage_threshold_0_85)
        tail_duration_below_0_88 = duration_condition(times[tail_mask], voltage_min_t[tail_mask] < config.tail_worst_bus_dip_threshold)
    else:
        tail_duration_below_0_85 = duration_condition(times, voltage_min_t < config.long_term_low_voltage_threshold_0_85)
        tail_duration_below_0_88 = duration_condition(times, voltage_min_t < config.tail_worst_bus_dip_threshold)

    final_recovered_ratio_0_9 = float(np.nanmean(voltages[final_mask] >= 0.9)) if final_mask.any() else float(np.nanmean(voltages[-1:] >= 0.9))
    final_recovered_ratio_0_95 = float(np.nanmean(voltages[final_mask] >= 0.95)) if final_mask.any() else float(np.nanmean(voltages[-1:] >= 0.95))
    final_low_bus_count_mean_0_9 = float(np.nanmean(np.sum(voltages[final_mask] < 0.9, axis=1))) if final_mask.any() else float(np.sum(voltages[-1:] < 0.9, axis=1)[0])
    final_high_bus_count_mean_1_12 = float(np.nanmean(np.sum(voltages[final_mask] > config.overvoltage_threshold_1_12, axis=1))) if final_mask.any() else float(np.sum(voltages[-1:] > config.overvoltage_threshold_1_12, axis=1)[0])
    duration_below_0_9 = duration_condition(times, voltage_min_t < 0.9)
    duration_below_0_85 = duration_condition(times, voltage_min_t < config.long_term_low_voltage_threshold_0_85)
    duration_below_0_8 = duration_condition(times, voltage_min_t < config.severe_low_voltage_threshold_0_8)
    duration_below_0_7 = duration_condition(times, voltage_min_t < config.extreme_low_voltage_threshold_0_7)
    duration_above_1_12 = duration_condition(times, voltage_max_t > config.overvoltage_threshold_1_12)
    below_0_9_mask = np.any(voltages < 0.9, axis=1)
    last_time_below_0_9 = float(times[np.where(below_0_9_mask)[0][-1]]) if np.any(below_0_9_mask) else float(times[0])
    undervoltage_area = float(np.trapezoid(np.nanmean(np.clip(0.9 - voltages, 0.0, None), axis=1), x=times))
    overvoltage_area = float(np.trapezoid(np.nanmean(np.clip(voltages - config.overvoltage_threshold_1_12, 0.0, None), axis=1), x=times))
    non_recovery_ratio = float(1.0 - np.nanmean(np.nanmean(voltages[final_mask], axis=0) >= 0.9)) if final_mask.any() else float(1.0 - np.nanmean(voltages[-1:, :] >= 0.9))

    return {
        "min_voltage_global": float(np.nanmin(voltages)),
        "voltage_mean_t": voltage_mean_t,
        "voltage_min_t": voltage_min_t,
        "voltage_max_t": voltage_max_t,
        "tail_voltage_mean": tail_voltage_mean,
        "tail_voltage_min": tail_voltage_min,
        "tail_voltage_max": tail_voltage_max,
        "tail_voltage_std": tail_voltage_std,
        "tail_voltage_slope": float(tail_voltage_slope),
        "tail_voltage_min_min": tail_voltage_min_min,
        "tail_voltage_min_std": tail_voltage_min_std,
        "tail_duration_below_0_85": tail_duration_below_0_85,
        "tail_duration_below_0_88": tail_duration_below_0_88,
        "final_recovered_ratio_0_9": final_recovered_ratio_0_9,
        "final_recovered_ratio_0_95": final_recovered_ratio_0_95,
        "final_low_bus_count_mean_0_9": final_low_bus_count_mean_0_9,
        "final_high_bus_count_mean_1_12": final_high_bus_count_mean_1_12,
        "duration_below_0_9": duration_below_0_9,
        "duration_below_0_85": duration_below_0_85,
        "duration_below_0_8": duration_below_0_8,
        "duration_below_0_7": duration_below_0_7,
        "duration_above_1_12": duration_above_1_12,
        "last_time_below_0_9": last_time_below_0_9,
        "undervoltage_area": undervoltage_area,
        "overvoltage_area": overvoltage_area,
        "non_recovery_ratio": non_recovery_ratio,
        "_tail_mask": tail_mask,
        "_tail_trend": line_fit(times[tail_mask], voltage_mean_t[tail_mask]) if tail_mask.any() else (0.0, 0.0),
    }


def score_instability(angle_features: dict[str, Any], voltage_features: dict[str, Any], config: StabilityConfig) -> dict[str, Any]:
    angle_speed_gate = float(
        angle_features["tail_spread_mean"] >= config.spread_stable_threshold
        or angle_features["tail_peak_max_abs_relative"] >= config.large_relative_angle_threshold
    )
    angle_components = {
        "tail_spread": normalize_value(angle_features["tail_spread_mean"], config.spread_stable_threshold, config.spread_unstable_threshold),
        "spread_slope": normalize_value(angle_features["tail_spread_slope"], config.spread_slope_threshold * 0.2, config.spread_slope_threshold),
        "angle_speed": angle_speed_gate * normalize_value(angle_features["tail_angle_speed"], config.angle_speed_threshold * 0.2, config.angle_speed_threshold),
        "divergence": angle_features["divergence_score"],
        "slip": angle_features["slip_score"],
        "single_machine_swing": max(
            normalize_value(angle_features["tail_peak_max_abs_relative"], config.large_relative_angle_threshold, config.extreme_relative_angle_threshold),
            normalize_value(
                angle_features["tail_duration_large_relative_angle"],
                config.large_relative_angle_duration_threshold_sec * 0.25,
                config.large_relative_angle_duration_threshold_sec,
            ),
        ),
        "sustained_oscillation": min(
            1.0,
            0.55 * normalize_value(angle_features["tail_oscillation_amplitude_mean"], config.oscillation_tail_amplitude_threshold * 0.5, config.oscillation_tail_amplitude_threshold)
            + 0.25 * normalize_value(angle_features["oscillation_decay_ratio"], config.oscillation_decay_ratio_threshold * 0.7, config.oscillation_decay_ratio_threshold)
            + 0.20 * angle_speed_gate * normalize_value(angle_features["tail_angle_speed"], config.oscillation_angle_speed_threshold * 0.5, config.oscillation_angle_speed_threshold),
        ),
        "invalid_tail": normalize_value(angle_features["invalid_tail_ratio"], config.invalid_ratio_threshold * 0.5, config.invalid_ratio_threshold),
        "pre_invalid_divergence": normalize_value(angle_features["pre_invalid_divergence"], config.spread_unstable_threshold * 0.35, config.spread_unstable_threshold),
    }
    angle_weights = {
        "tail_spread": 0.24,
        "spread_slope": 0.20,
        "angle_speed": 0.12,
        "divergence": 0.16,
        "slip": 0.18,
        "single_machine_swing": 0.10,
        "sustained_oscillation": 0.12,
        "invalid_tail": 0.03,
        "pre_invalid_divergence": 0.04,
    }
    angle_instability_score = weighted_score(angle_components, angle_weights)

    voltage_components = {
        "tail_voltage_mean": 1.0 - normalize_value(voltage_features["tail_voltage_mean"], 0.82, 0.97),
        "tail_voltage_min": 1.0 - normalize_value(voltage_features["tail_voltage_min"], 0.70, 0.92),
        "tail_voltage_slope": normalize_value(-voltage_features["tail_voltage_slope"], 0.002, 0.02),
        "recovered_ratio_0_9": 1.0 - normalize_value(voltage_features["final_recovered_ratio_0_9"], config.voltage_recovery_threshold_0_9 * 0.5, config.voltage_recovery_threshold_0_9),
        "recovered_ratio_0_95": 1.0 - normalize_value(voltage_features["final_recovered_ratio_0_95"], config.voltage_recovery_threshold_0_95 * 0.5, config.voltage_recovery_threshold_0_95),
        "duration_below_0_9": normalize_value(voltage_features["duration_below_0_9"], config.duration_below_0_9_threshold_sec * 0.5, config.duration_below_0_9_threshold_sec),
        "duration_below_0_85": normalize_value(voltage_features["duration_below_0_85"], config.duration_below_0_85_threshold_sec * 0.5, config.duration_below_0_85_threshold_sec),
        "duration_below_0_8": normalize_value(voltage_features["duration_below_0_8"], config.duration_below_0_8_threshold_sec * 0.5, config.duration_below_0_8_threshold_sec),
        "duration_below_0_7": normalize_value(voltage_features["duration_below_0_7"], config.duration_below_0_7_threshold_sec * 0.5, config.duration_below_0_7_threshold_sec),
        "duration_above_1_12": normalize_value(voltage_features["duration_above_1_12"], config.duration_above_1_12_threshold_sec * 0.5, config.duration_above_1_12_threshold_sec),
        "undervoltage_area": normalize_value(voltage_features["undervoltage_area"], config.undervoltage_area_threshold * 0.4, config.undervoltage_area_threshold),
        "non_recovery_ratio": normalize_value(voltage_features["non_recovery_ratio"], config.non_recovery_ratio_threshold * 0.4, config.non_recovery_ratio_threshold),
    }
    voltage_weights = {
        "tail_voltage_mean": 0.12,
        "tail_voltage_min": 0.15,
        "tail_voltage_slope": 0.08,
        "recovered_ratio_0_9": 0.18,
        "recovered_ratio_0_95": 0.10,
        "duration_below_0_9": 0.09,
        "duration_below_0_85": 0.12,
        "duration_below_0_8": 0.10,
        "duration_below_0_7": 0.05,
        "duration_above_1_12": 0.10,
        "undervoltage_area": 0.08,
        "non_recovery_ratio": 0.05,
    }
    voltage_instability_score = weighted_score(voltage_components, voltage_weights)

    total_instability_score = float(np.clip(config.angle_weight * angle_instability_score + config.voltage_weight * voltage_instability_score, 0.0, 1.0))

    strong_rules: list[str] = []
    reason: list[str] = []
    if angle_features["slip_score"] >= 0.85:
        strong_rules.append("功角出现明显滑极迹象")
    if angle_features["tail_spread_mean"] >= config.spread_unstable_threshold and angle_features["tail_spread_slope"] >= config.spread_slope_threshold:
        strong_rules.append("功角后期仍持续发散")

    single_machine_long_excursion = (
        angle_features["peak_max_abs_relative"] >= config.extreme_relative_angle_threshold
        and (
            (
                angle_features["tail_peak_max_abs_relative"] >= config.extreme_relative_angle_threshold
                and angle_features["tail_duration_large_relative_angle"] >= config.large_relative_angle_duration_threshold_sec * 0.25
            )
            or (
                angle_features["tail_spread_slope"] >= config.reseparating_tail_spread_slope_threshold
                and angle_features["tail_spread_mean"] >= config.reseparating_tail_spread_threshold
                and angle_features["tail_spread_peak"] >= config.single_machine_reseparation_tail_spread_threshold
                and angle_features["tail_oscillation_amplitude_max"] >= config.single_machine_reseparation_amp_max_threshold
                and angle_features["oscillation_decay_ratio"] >= config.oscillation_decay_ratio_threshold
            )
        )
    )
    single_machine_extreme_excursion = (
        angle_features["peak_max_abs_relative"] >= config.single_machine_extreme_peak_threshold
        and (
            (
                angle_features["tail_spread_peak"] >= config.single_machine_extreme_tail_spread_peak_threshold
                and (
                    angle_features["tail_spread_slope"] > 0.0
                    or angle_features["tail_duration_large_relative_angle"] >= config.large_relative_angle_duration_threshold_sec * 0.25
                )
            )
            or (
                angle_features["tail_peak_max_abs_relative"] >= config.single_machine_extreme_tail_peak_relative_threshold
                and angle_features["tail_oscillation_amplitude_max"] >= config.single_machine_extreme_amp_max_threshold
                and (
                    angle_features["tail_spread_mean"] >= config.single_machine_extreme_tail_spread_threshold
                    or angle_features["tail_spread_std"] >= config.single_machine_extreme_spread_std_threshold
                )
            )
        )
    )
    if single_machine_long_excursion or single_machine_extreme_excursion:
        strong_rules.append("单机相对功角长时间大幅偏离同步群")

    if (
        angle_features["tail_oscillation_amplitude_mean"] >= config.oscillation_tail_amplitude_threshold
        and angle_features["oscillation_decay_ratio"] >= config.oscillation_decay_ratio_threshold
        and (
            angle_features["tail_spread_slope"] >= 0.0
            or angle_features["tail_peak_max_abs_relative"] >= config.oscillation_ampmax_peak_relative_threshold
        )
        and (
            (
                angle_features["tail_oscillation_amplitude_max"] >= config.oscillation_tail_amplitude_max_threshold
                and (
                    angle_features["tail_spread_mean"] >= config.oscillation_ampmax_tail_spread_threshold
                    or angle_features["tail_peak_max_abs_relative"] >= config.oscillation_ampmax_peak_relative_threshold
                    or angle_features["tail_spread_std"] >= config.oscillation_ampmax_spread_std_threshold
                )
            )
            or (
                angle_features["tail_spread_std"] >= config.oscillation_spread_std_threshold
                and (
                    angle_features["tail_spread_mean"] >= config.oscillation_spread_tail_spread_threshold
                    or angle_features["tail_peak_max_abs_relative"] >= config.oscillation_spread_peak_relative_threshold
                    or angle_features["tail_oscillation_amplitude_max"] >= config.oscillation_spread_amp_max_threshold
                )
            )
        )
    ):
        strong_rules.append("功角持续大幅振荡且阻尼不足")
    if (
        angle_features["tail_oscillation_amplitude_mean"] >= config.high_amp_oscillation_tail_amplitude_mean_threshold
        and angle_features["tail_spread_mean"] >= config.high_amp_oscillation_tail_spread_threshold
        and angle_features["tail_angle_speed"] >= config.high_amp_oscillation_angle_speed_threshold
        and angle_features["oscillation_decay_ratio"] >= config.high_amp_oscillation_decay_ratio_threshold
        and (
            angle_features["tail_spread_slope"] >= config.high_amp_oscillation_spread_slope_threshold
            or angle_features["peak_max_abs_relative"] >= config.high_amp_oscillation_peak_relative_threshold
            or angle_features["tail_oscillation_amplitude_max"] >= config.high_amp_oscillation_amp_max_threshold
        )
    ):
        strong_rules.append("功角高振幅长期振荡且尾段展宽偏大")

    if (
        angle_features["tail_oscillation_amplitude_max"] >= config.moderate_oscillation_tail_amplitude_max_threshold
        and angle_features["tail_oscillation_amplitude_mean"] <= config.moderate_oscillation_tail_amplitude_mean_upper_threshold
        and angle_features["oscillation_decay_ratio"] >= config.oscillation_decay_ratio_threshold
        and angle_features["tail_angle_speed"] >= config.moderate_oscillation_angle_speed_threshold
        and (
            angle_features["tail_spread_mean"] >= config.moderate_oscillation_tail_spread_threshold
            or angle_features["tail_peak_max_abs_relative"] >= config.moderate_oscillation_peak_relative_threshold
        )
    ):
        strong_rules.append("功角中等幅值振荡持续不衰减")

    if (
        angle_features["tail_spread_mean"] >= config.large_tail_spread_strong_threshold
        and angle_features["tail_peak_max_abs_relative"] >= config.large_tail_peak_relative_strong_threshold
        and angle_features["tail_angle_speed"] >= config.large_tail_angle_speed_strong_threshold
    ):
        strong_rules.append("功角尾段展宽过大且未回到同步群")

    if (
        angle_features["tail_spread_mean"] >= config.growing_tail_spread_strong_threshold
        and angle_features["tail_spread_slope"] >= config.growing_tail_spread_slope_threshold
        and angle_features["tail_peak_max_abs_relative"] >= config.growing_tail_peak_relative_threshold
        and angle_features["tail_duration_large_relative_angle"] >= config.growing_tail_duration_large_angle_threshold_sec
    ):
        strong_rules.append("功角尾段展宽较大且仍在继续扩大")

    if (
        angle_features["tail_spread_mean"] >= config.reseparating_tail_spread_threshold
        and angle_features["tail_spread_slope"] >= config.reseparating_tail_spread_slope_threshold
        and angle_features["peak_max_abs_relative"] >= config.reseparating_peak_relative_threshold
        and angle_features["tail_oscillation_amplitude_max"] >= config.reseparating_tail_amp_max_threshold
        and angle_features["oscillation_decay_ratio"] >= config.reseparating_decay_ratio_threshold
    ):
        strong_rules.append("功角尾段仍在重新拉开且单机摆幅较大")

    if angle_features["invalid_tail_ratio"] >= config.invalid_ratio_threshold and angle_features["pre_invalid_divergence"] >= config.spread_unstable_threshold * 0.5:
        strong_rules.append("功角先发散后大量截断")

    severe_voltage_nonrecovery = (
        voltage_features["duration_below_0_8"] >= config.duration_below_0_8_threshold_sec
        and (
            voltage_features["final_recovered_ratio_0_9"] < config.voltage_recovery_threshold_0_9 * 0.5
            or voltage_features["non_recovery_ratio"] >= config.non_recovery_ratio_threshold
            or voltage_features["tail_voltage_min"] < config.severe_low_voltage_threshold_0_8
        )
    )
    long_term_low_voltage_nonrecovery = (
        voltage_features["duration_below_0_85"] >= config.duration_below_0_85_threshold_sec
        and (
            voltage_features["tail_voltage_min"] < config.long_term_low_voltage_threshold_0_85
            or voltage_features["final_recovered_ratio_0_9"] < config.voltage_recovery_threshold_0_9
            or voltage_features["non_recovery_ratio"] >= config.non_recovery_ratio_threshold
        )
    )
    persistent_local_low_voltage = (
        voltage_features["final_low_bus_count_mean_0_9"] >= 1.0
        and voltage_features["tail_voltage_min"] < config.persistent_local_low_voltage_threshold
        and voltage_features["final_recovered_ratio_0_9"] < 1.0
    )
    extreme_voltage_nonrecovery = (
        voltage_features["duration_below_0_7"] >= config.duration_below_0_7_threshold_sec
        and (
            voltage_features["final_recovered_ratio_0_9"] < config.voltage_recovery_threshold_0_9 * 0.9
            or voltage_features["non_recovery_ratio"] >= config.non_recovery_ratio_threshold
            or voltage_features["tail_voltage_min"] < config.long_term_low_voltage_threshold_0_85
        )
    )
    long_term_overvoltage = (
        voltage_features["duration_above_1_12"] >= config.duration_above_1_12_threshold_sec
        and voltage_features["tail_voltage_max"] > config.overvoltage_threshold_1_12
        and voltage_features["final_high_bus_count_mean_1_12"] >= config.persistent_high_voltage_bus_count_threshold
    )
    mixed_voltage_split = (
        voltage_features["duration_below_0_85"] >= config.mixed_voltage_split_duration_below_0_85_threshold_sec
        and voltage_features["duration_above_1_12"] >= config.mixed_voltage_split_duration_above_1_12_threshold_sec
        and (
            voltage_features["final_recovered_ratio_0_95"] < config.mixed_voltage_split_recovered_0_95_threshold
            or voltage_features["final_low_bus_count_mean_0_9"] >= config.mixed_voltage_split_low_bus_threshold
            or voltage_features["final_high_bus_count_mean_1_12"] >= config.mixed_voltage_split_high_bus_threshold
        )
    )
    voltage_oscillation_nonrecovery = (
        voltage_features["tail_voltage_std"] >= config.voltage_oscillation_std_threshold
        and voltage_features["duration_below_0_9"] >= config.voltage_oscillation_duration_below_0_9_threshold_sec
        and voltage_features["final_low_bus_count_mean_0_9"] >= config.voltage_oscillation_low_bus_threshold
        and voltage_features["tail_voltage_min"] < config.voltage_oscillation_tail_min_threshold
    )
    voltage_oscillation_strict_nonrecovery = (
        voltage_features["tail_voltage_std"] >= config.voltage_oscillation_strict_std_threshold
        and voltage_features["duration_below_0_9"] >= config.voltage_oscillation_duration_below_0_9_threshold_sec
        and voltage_features["final_low_bus_count_mean_0_9"] >= config.voltage_oscillation_strict_low_bus_threshold
        and voltage_features["tail_voltage_min"] < config.voltage_oscillation_strict_tail_min_threshold
    )
    severe_voltage_oscillation = (
        voltage_features["tail_voltage_std"] >= config.severe_voltage_oscillation_std_threshold
        and voltage_features["duration_below_0_85"] >= config.severe_voltage_oscillation_duration_below_0_85_threshold_sec
        and voltage_features["duration_below_0_8"] >= config.severe_voltage_oscillation_duration_below_0_8_threshold_sec
    )
    repeated_voltage_dip_instability = (
        voltage_features["tail_voltage_std"] >= config.repeated_voltage_dip_std_threshold
        and voltage_features["duration_below_0_85"] >= config.repeated_voltage_dip_duration_below_0_85_threshold_sec
        and voltage_features["final_recovered_ratio_0_95"] < config.repeated_voltage_dip_recovered_0_95_threshold
    )
    tail_repeated_voltage_dip_instability = (
        voltage_features["tail_voltage_std"] >= config.tail_repeated_voltage_dip_std_threshold
        and voltage_features["tail_voltage_min_min"] < config.tail_repeated_voltage_dip_min_threshold
        and voltage_features["tail_duration_below_0_85"] >= config.tail_repeated_voltage_dip_duration_threshold_sec
        and voltage_features["final_recovered_ratio_0_95"] < config.tail_repeated_voltage_dip_recovered_0_95_threshold
    )
    tail_worst_bus_oscillation = (
        voltage_features["tail_voltage_min_std"] >= config.tail_worst_bus_oscillation_std_threshold
        and voltage_features["tail_voltage_min_min"] < config.tail_worst_bus_dip_threshold
        and voltage_features["tail_duration_below_0_88"] >= config.tail_worst_bus_dip_duration_threshold_sec
    )
    persistent_single_bus_low_voltage = (
        voltage_features["tail_voltage_min"] < config.persistent_single_bus_low_voltage_tail_min_threshold
        and voltage_features["tail_duration_below_0_88"] >= config.persistent_single_bus_low_voltage_tail_duration_threshold_sec
        and voltage_features["final_low_bus_count_mean_0_9"] >= config.persistent_single_bus_low_voltage_low_bus_threshold
    )
    moderate_voltage_oscillation = (
        voltage_features["tail_voltage_std"] >= config.moderate_voltage_oscillation_std_threshold
        and voltage_features["duration_below_0_85"] >= config.moderate_voltage_oscillation_duration_below_0_85_threshold_sec
        and voltage_features["duration_below_0_8"] >= config.moderate_voltage_oscillation_duration_below_0_8_threshold_sec
        and voltage_features["final_recovered_ratio_0_95"] < config.moderate_voltage_oscillation_recovered_0_95_threshold
    )
    moderate_overvoltage_nonrecovery = (
        voltage_features["duration_above_1_12"] >= config.moderate_overvoltage_duration_threshold_sec
        and voltage_features["final_high_bus_count_mean_1_12"] >= config.moderate_overvoltage_high_bus_threshold
        and voltage_features["final_recovered_ratio_0_95"] < config.moderate_overvoltage_recovered_0_95_threshold
        and voltage_features["final_low_bus_count_mean_0_9"] >= config.moderate_overvoltage_low_bus_threshold
    )
    coupled_angle_voltage_oscillation = (
        angle_features["tail_oscillation_amplitude_mean"] >= config.coupled_angle_voltage_oscillation_amp_threshold
        and angle_features["tail_angle_speed"] >= config.coupled_angle_voltage_oscillation_speed_threshold
        and angle_features["peak_max_abs_relative"] >= config.coupled_angle_voltage_oscillation_peak_threshold
        and voltage_features["tail_voltage_std"] >= config.coupled_angle_voltage_oscillation_voltage_std_threshold
    )
    slow_voltage_recovery = (
        voltage_features["duration_below_0_85"] >= config.slow_recovery_duration_below_0_85_threshold_sec
        and voltage_features["duration_below_0_8"] >= config.slow_recovery_duration_below_0_8_threshold_sec
        and voltage_features["last_time_below_0_9"] >= config.slow_recovery_last_below_0_9_threshold_sec
    )
    persistent_multi_bus_low_voltage = (
        voltage_features["duration_below_0_85"] >= config.persistent_multi_bus_low_voltage_duration_below_0_85_threshold_sec
        and voltage_features["final_recovered_ratio_0_95"] < config.persistent_multi_bus_low_voltage_recovered_0_95_threshold
        and voltage_features["final_low_bus_count_mean_0_9"] >= config.persistent_multi_bus_low_voltage_low_bus_threshold
        and voltage_features["tail_voltage_min"] < config.persistent_multi_bus_low_voltage_tail_min_threshold
    )

    if severe_voltage_nonrecovery:
        strong_rules.append("电压长期严重不恢复")
    if long_term_low_voltage_nonrecovery:
        strong_rules.append("末段长期低于 0.85 p.u.，判定为低压失稳")
    if persistent_local_low_voltage:
        strong_rules.append("存在局部节点末段持续低电压")
    if extreme_voltage_nonrecovery:
        strong_rules.append("存在较长时间极端低压且后期恢复不足")
    if long_term_overvoltage:
        strong_rules.append("末段长期高于 1.12 p.u.，判定为过电压失稳")
    if mixed_voltage_split:
        strong_rules.append("电压长期低压与过电压并存，判定为电压失稳")
    if voltage_oscillation_nonrecovery:
        strong_rules.append("电压长时间振荡且谷值反复下探，判定为电压失稳")
    if voltage_oscillation_strict_nonrecovery:
        strong_rules.append("电压长时间振荡且局部低压持续不恢复，判定为电压失稳")
    if severe_voltage_oscillation:
        strong_rules.append("电压强烈振荡且深跌持续出现，判定为电压失稳")
    if repeated_voltage_dip_instability:
        strong_rules.append("电压反复跌入 0.85 p.u. 以下且后期恢复不足，判定为电压失稳")
    if tail_repeated_voltage_dip_instability:
        strong_rules.append("尾段电压反复跌入 0.85 p.u. 以下且后期恢复不足，判定为电压失稳")
    if tail_worst_bus_oscillation:
        strong_rules.append("尾段最差节点电压反复深跌，判定为电压失稳")
    if persistent_single_bus_low_voltage:
        strong_rules.append("尾段存在单节点持续低电压，判定为电压失稳")
    if moderate_voltage_oscillation:
        strong_rules.append("电压振荡明显且低压持续时间偏长，判定为电压失稳")
    if moderate_overvoltage_nonrecovery:
        strong_rules.append("长期过电压且恢复质量不足，判定为电压失稳")
    if coupled_angle_voltage_oscillation:
        strong_rules.append("功角大幅振荡且电压末段振荡明显，判定为失稳")
    if slow_voltage_recovery:
        strong_rules.append("电压恢复达到稳定范围过慢，判定为电压失稳")
    if persistent_multi_bus_low_voltage:
        strong_rules.append("末段多节点长期低压恢复失败，判定为电压失稳")

    if angle_components["tail_spread"] >= 0.6:
        reason.append(f"末段相对功角展宽偏大，尾段平均 spread={angle_features['tail_spread_mean']:.1f}°")
    if angle_components["spread_slope"] >= 0.6:
        reason.append(f"末段功角 spread 仍在扩大，斜率={angle_features['tail_spread_slope']:.2f}°/s")
    if angle_components["angle_speed"] >= 0.6:
        reason.append(f"末段相对角速度偏大，中位数={angle_features['tail_angle_speed']:.2f}°/s")
    if angle_components["single_machine_swing"] >= 0.6:
        reason.append(
            f"尾段仍存在单机较大相对摆角，tail_peak_abs_relative={angle_features['tail_peak_max_abs_relative']:.1f}°，"
            f"tail_duration_over_{int(config.large_relative_angle_threshold)}={angle_features['tail_duration_large_relative_angle']:.2f}s"
        )
    if angle_components["sustained_oscillation"] >= 0.6:
        reason.append(
            f"功角振荡长时间不衰减，tail_amp_mean={angle_features['tail_oscillation_amplitude_mean']:.2f}°，"
            f"decay_ratio={angle_features['oscillation_decay_ratio']:.2f}"
        )
    if angle_components["invalid_tail"] >= 0.6:
        reason.append(f"末段功角无效值占比偏高，invalid_tail_ratio={angle_features['invalid_tail_ratio']:.2%}")
    if voltage_components["recovered_ratio_0_9"] >= 0.6:
        reason.append(f"末段恢复到 0.9 p.u. 以上的比例偏低，final_recovered_ratio_0_9={voltage_features['final_recovered_ratio_0_9']:.2%}")
    if voltage_features["final_low_bus_count_mean_0_9"] >= 1.0 and voltage_features["tail_voltage_min"] < config.persistent_local_low_voltage_threshold:
        reason.append(
            f"末段仍有局部节点低于 0.9 p.u.，最低节点均值={voltage_features['tail_voltage_min']:.3f} p.u."
        )
    if voltage_components["duration_below_0_85"] >= 0.6:
        reason.append(f"低于 0.85 p.u. 的持续时间偏长，duration_below_0_85={voltage_features['duration_below_0_85']:.2f}s")
    if voltage_components["duration_below_0_8"] >= 0.6:
        reason.append(f"低于 0.8 p.u. 的持续时间偏长，duration_below_0_8={voltage_features['duration_below_0_8']:.2f}s")
    if voltage_components["duration_above_1_12"] >= 0.6:
        reason.append(f"高于 1.12 p.u. 的持续时间偏长，duration_above_1_12={voltage_features['duration_above_1_12']:.2f}s")
    if voltage_features["tail_voltage_std"] >= config.voltage_oscillation_std_threshold:
        reason.append(f"末段电压振荡偏强，tail_voltage_std={voltage_features['tail_voltage_std']:.4f}")
    if voltage_components["undervoltage_area"] >= 0.6:
        reason.append(f"电压恢复面积不足，undervoltage_area={voltage_features['undervoltage_area']:.3f}")
    if not reason:
        reason.append("功角和电压尾段整体恢复较好")

    return {
        "angle_instability_score": angle_instability_score,
        "voltage_instability_score": voltage_instability_score,
        "total_instability_score": total_instability_score,
        "angle_components": angle_components,
        "voltage_components": voltage_components,
        "strong_rules": strong_rules,
        "reason": reason[:5],
    }


def classify_binary(score_dict: dict[str, Any], config: StabilityConfig) -> dict[str, Any]:
    total_score = score_dict["total_instability_score"]
    strong_rules = score_dict["strong_rules"]
    if strong_rules:
        pred_label = "unstable"
        confidence = 0.95
        reason = strong_rules + score_dict["reason"]
    elif total_score >= config.total_instability_threshold:
        pred_label = "unstable"
        margin = min(1.0, (total_score - config.total_instability_threshold) / max(1e-6, 1.0 - config.total_instability_threshold))
        confidence = float(np.clip(0.55 + 0.40 * margin, 0.55, 0.95))
        reason = score_dict["reason"]
    else:
        pred_label = "stable"
        margin = min(1.0, (config.total_instability_threshold - total_score) / max(1e-6, config.total_instability_threshold))
        confidence = float(np.clip(0.55 + 0.35 * margin, 0.55, 0.90))
        reason = score_dict["reason"]

    return {
        "pred_label": pred_label,
        "pred_label_cn": label_en_to_cn(pred_label),
        "confidence": round(confidence, 4),
        "reason": reason[:5],
    }


def build_result(
    path: Path,
    sample: dict[str, Any],
    pre_angles: dict[str, Any],
    pre_voltages: dict[str, Any],
    angle_features: dict[str, Any],
    voltage_features: dict[str, Any],
    score_dict: dict[str, Any],
    classification: dict[str, Any],
    config: StabilityConfig,
) -> dict[str, Any]:
    label_value = int(sample["label"]) if "label" in sample and sample["label"] is not None else None
    original_label = label_num_to_en(label_value, config)
    pred_label = classification["pred_label"]
    result = {
        "file": str(path),
        "source_path": sample.get("source_path"),
        "original_label_value": label_value,
        "original_label": original_label,
        "original_label_cn": label_en_to_cn(original_label),
        "pred_label": pred_label,
        "pred_label_cn": classification["pred_label_cn"],
        "confidence": classification["confidence"],
        "reason": classification["reason"],
        "angle_metrics": {
            "tail_spread_mean": round_or_none(angle_features["tail_spread_mean"]),
            "tail_spread_std": round_or_none(angle_features["tail_spread_std"]),
            "tail_spread_slope": round_or_none(angle_features["tail_spread_slope"]),
            "tail_angle_speed": round_or_none(angle_features["tail_angle_speed"]),
            "divergence_score": round_or_none(angle_features["divergence_score"]),
            "slip_score": round_or_none(angle_features["slip_score"]),
            "peak_max_abs_relative": round_or_none(angle_features["peak_max_abs_relative"]),
            "duration_large_relative_angle": round_or_none(angle_features["duration_large_relative_angle"]),
            "duration_extreme_relative_angle": round_or_none(angle_features["duration_extreme_relative_angle"]),
            "tail_oscillation_amplitude_mean": round_or_none(angle_features["tail_oscillation_amplitude_mean"]),
            "tail_oscillation_amplitude_max": round_or_none(angle_features["tail_oscillation_amplitude_max"]),
            "prev_tail_oscillation_amplitude_mean": round_or_none(angle_features["prev_tail_oscillation_amplitude_mean"]),
            "oscillation_decay_ratio": round_or_none(angle_features["oscillation_decay_ratio"]),
            "invalid_tail_ratio": round_or_none(angle_features["invalid_tail_ratio"]),
            "pre_invalid_divergence": round_or_none(angle_features["pre_invalid_divergence"]),
            "spread_peak": round_or_none(array_nanmax_or_nan(angle_features["spread_t"])),
            "max_pairwise_sep_peak": round_or_none(array_nanmax_or_nan(angle_features["max_pairwise_sep_t"])),
        },
        "voltage_metrics": {
            "min_voltage_global": round_or_none(voltage_features["min_voltage_global"]),
            "tail_voltage_mean": round_or_none(voltage_features["tail_voltage_mean"]),
            "tail_voltage_min": round_or_none(voltage_features["tail_voltage_min"]),
            "tail_voltage_min_min": round_or_none(voltage_features["tail_voltage_min_min"]),
            "tail_voltage_min_std": round_or_none(voltage_features["tail_voltage_min_std"]),
            "tail_voltage_max": round_or_none(voltage_features["tail_voltage_max"]),
            "tail_voltage_std": round_or_none(voltage_features["tail_voltage_std"]),
            "tail_voltage_slope": round_or_none(voltage_features["tail_voltage_slope"]),
            "tail_duration_below_0_85": round_or_none(voltage_features["tail_duration_below_0_85"]),
            "tail_duration_below_0_88": round_or_none(voltage_features["tail_duration_below_0_88"]),
            "final_recovered_ratio_0_9": round_or_none(voltage_features["final_recovered_ratio_0_9"]),
            "final_recovered_ratio_0_95": round_or_none(voltage_features["final_recovered_ratio_0_95"]),
            "final_low_bus_count_mean_0_9": round_or_none(voltage_features["final_low_bus_count_mean_0_9"]),
            "final_high_bus_count_mean_1_12": round_or_none(voltage_features["final_high_bus_count_mean_1_12"]),
            "duration_below_0_9": round_or_none(voltage_features["duration_below_0_9"]),
            "duration_below_0_85": round_or_none(voltage_features["duration_below_0_85"]),
            "duration_below_0_8": round_or_none(voltage_features["duration_below_0_8"]),
            "duration_below_0_7": round_or_none(voltage_features["duration_below_0_7"]),
            "duration_above_1_12": round_or_none(voltage_features["duration_above_1_12"]),
            "undervoltage_area": round_or_none(voltage_features["undervoltage_area"]),
            "overvoltage_area": round_or_none(voltage_features["overvoltage_area"]),
            "non_recovery_ratio": round_or_none(voltage_features["non_recovery_ratio"]),
        },
        "score_breakdown": {
            "angle_instability_score": round_or_none(score_dict["angle_instability_score"]),
            "voltage_instability_score": round_or_none(score_dict["voltage_instability_score"]),
            "total_instability_score": round_or_none(score_dict["total_instability_score"]),
            "angle_components": {k: round_or_none(v) for k, v in score_dict["angle_components"].items()},
            "voltage_components": {k: round_or_none(v) for k, v in score_dict["voltage_components"].items()},
            "strong_rules": score_dict["strong_rules"],
        },
        "has_invalid_angles": pre_angles["has_invalid_angles"],
        "first_invalid_time": round_or_none(pre_angles["first_invalid_time"]),
        "times_monotonic": pre_angles["times_monotonic"],
    }
    result["is_consistent"] = bool(label_en_to_num(pred_label, config) == label_value) if label_value is not None else None
    return result


def strip_plot_data(result: dict[str, Any]) -> dict[str, Any]:
    clean = dict(result)
    clean.pop("_plot_data", None)
    return serialize_data(clean)


def build_plot_payload(pre_angles: dict[str, Any], pre_voltages: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "times": np.asarray(pre_angles["times"], dtype=float),
        "cleaned_angles": np.asarray(pre_angles["cleaned_angles"], dtype=float),
        "voltages": np.asarray(pre_voltages["voltages"], dtype=float),
    }


def load_plot_payload(path: Path, config: StabilityConfig) -> dict[str, np.ndarray]:
    sample = load_npy_sample(path)
    missing = [name for name in ["angles", "voltages", "times"] if name not in sample]
    if missing:
        raise ValueError(f"样本缺少关键字段: {missing}")
    pre_angles = preprocess_angles(sample["angles"], sample["times"], invalid_value=config.invalid_value, unwrap=config.unwrap_angles, ref_mode=config.ref_mode)
    pre_voltages = preprocess_voltages(sample["voltages"], sample["times"])
    return build_plot_payload(pre_angles, pre_voltages)


def plot_diagnostics(result: dict[str, Any], plot_data: dict[str, np.ndarray], save_path: Path) -> None:
    configure_matplotlib_fonts()
    times = np.asarray(plot_data["times"], dtype=float)
    cleaned_angles = np.asarray(plot_data["cleaned_angles"], dtype=float)
    voltages = np.asarray(plot_data["voltages"], dtype=float)

    voltage_count = voltages.shape[1]
    angle_count = cleaned_angles.shape[1]
    fig_width = max(16.0, 14.0 + max(voltage_count, angle_count) * 0.08)
    fig_height = max(12.0, 10.0 + max(voltage_count, angle_count) * 0.04)
    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, constrained_layout=True)

    voltage_colors = plt.cm.turbo(np.linspace(0.0, 1.0, voltage_count))
    for idx in range(voltage_count):
        ax_v.plot(times, voltages[:, idx], color=voltage_colors[idx], alpha=0.95, linewidth=1.2)
    ax_v.set_ylabel("?? / p.u.")
    ax_v.set_title(f"电压曲线总览（共 {voltage_count} 条，全部显示）")
    ax_v.grid(alpha=0.18, which="major")
    ax_v.grid(alpha=0.10, which="minor", linestyle=":")
    ax_v.yaxis.set_major_locator(AutoLocator())
    ax_v.yaxis.set_minor_locator(FixedLocator(focused_tick_values(0.8, 1.15, 0.02)))

    angle_colors = plt.cm.nipy_spectral(np.linspace(0.0, 1.0, angle_count))
    for idx in range(angle_count):
        ax_a.plot(times, cleaned_angles[:, idx], color=angle_colors[idx], alpha=0.95, linewidth=1.25)
    ax_a.set_ylabel("?? / ?")
    ax_a.set_xlabel("?? / s")
    ax_a.set_title(f"功角曲线总览（共 {angle_count} 条，全部显示）")
    ax_a.grid(alpha=0.18, which="major")
    ax_a.grid(alpha=0.10, which="minor", linestyle=":")
    ax_a.yaxis.set_major_locator(AutoLocator())
    finite_angles = cleaned_angles[np.isfinite(cleaned_angles)]
    if finite_angles.size:
        focus_low = float(np.nanpercentile(finite_angles, 5))
        focus_high = float(np.nanpercentile(finite_angles, 95))
        focus_span = max(focus_high - focus_low, 1.0)
        if focus_span <= 60:
            minor_step = 5.0
        elif focus_span <= 120:
            minor_step = 10.0
        elif focus_span <= 240:
            minor_step = 15.0
        else:
            minor_step = 20.0
        ax_a.yaxis.set_minor_locator(FixedLocator(focused_tick_values(focus_low, focus_high, minor_step)))

    title = f"{Path(result['file']).name} | 原始标签: {result['original_label_cn']} | 检测结果: {result['pred_label_cn']} | 置信度: {result['confidence']:.2f}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    reason_text = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(result["reason"][:5]))
    fig.text(
        0.02,
        0.02,
        "主要原因:\n" + reason_text,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#AAAAAA"},
    )

    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def analyze_one_file(path: Path, config: StabilityConfig, plot: bool = False, save_dir: Path | None = None) -> dict[str, Any]:
    sample = load_npy_sample(path)
    missing = [name for name in ["angles", "voltages", "times"] if name not in sample]
    if missing:
        raise ValueError(f"样本缺少关键字段: {missing}")

    pre_angles = preprocess_angles(sample["angles"], sample["times"], invalid_value=config.invalid_value, unwrap=config.unwrap_angles, ref_mode=config.ref_mode)
    pre_voltages = preprocess_voltages(sample["voltages"], sample["times"])
    angle_features = extract_angle_features(pre_angles["relative_angles"], pre_angles["effective_times"], config)
    voltage_features = extract_voltage_features(pre_voltages["voltages"], pre_voltages["times"], config)
    score_dict = score_instability(angle_features, voltage_features, config)
    classification = classify_binary(score_dict, config)
    result = build_result(path, sample, pre_angles, pre_voltages, angle_features, voltage_features, score_dict, classification, config)
    if plot and save_dir is not None:
        plot_diagnostics(result, build_plot_payload(pre_angles, pre_voltages), save_dir)
    return result


def evaluate_against_labels(results_df: pd.DataFrame) -> dict[str, Any]:
    labeled = results_df.dropna(subset=["original_label_value"]).copy()
    if labeled.empty:
        return {}
    y_true = labeled["original_label_value"].astype(int).to_numpy()
    y_pred = labeled["pred_label_num"].astype(int).to_numpy()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def save_confusion_matrix(metrics: dict[str, Any], save_path: Path) -> None:
    configure_matplotlib_fonts()
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["预测稳定", "预测失稳"])
    ax.set_yticks([0, 1], labels=["真实稳定", "真实失稳"])
    ax.set_title(f"混淆矩阵\nAcc={metrics['accuracy']:.3f}  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def result_to_summary_row(result: dict[str, Any], config: StabilityConfig) -> dict[str, Any]:
    return {
        "file": result["file"],
        "原始标签值": result["original_label_value"],
        "原始标签": result["original_label_cn"],
        "检测结果": result["pred_label_cn"],
        "是否一致": result["is_consistent"],
        "confidence": result["confidence"],
        "主要原因": "；".join(result["reason"]),
        "is_consistent": result["is_consistent"],
        "pred_label": result["pred_label"],
        "pred_label_num": label_en_to_num(result["pred_label"], config),
        "original_label_value": result["original_label_value"],
        "total_instability_score": result["score_breakdown"]["total_instability_score"],
        "angle_instability_score": result["score_breakdown"]["angle_instability_score"],
        "voltage_instability_score": result["score_breakdown"]["voltage_instability_score"],
        "has_invalid_angles": result["has_invalid_angles"],
        "first_invalid_time": result["first_invalid_time"],
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serialize_data(payload), fh, ensure_ascii=False, indent=2)


def build_error_result(file_path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "file": str(file_path),
        "source_path": None,
        "original_label_value": None,
        "original_label": None,
        "original_label_cn": "未知",
        "pred_label": "unstable",
        "pred_label_cn": "失稳",
        "confidence": 0.51,
        "reason": [f"样本处理失败: {exc}"],
        "angle_metrics": {},
        "voltage_metrics": {},
        "score_breakdown": {
            "angle_instability_score": None,
            "voltage_instability_score": None,
            "total_instability_score": None,
            "angle_components": {},
            "voltage_components": {},
            "strong_rules": ["样本处理失败，按保守策略标记为失稳"],
        },
        "has_invalid_angles": None,
        "first_invalid_time": None,
        "times_monotonic": None,
        "is_consistent": None,
    }


def analyze_file_task(file_path_str: str, config: StabilityConfig) -> dict[str, Any]:
    file_path = Path(file_path_str)
    try:
        return analyze_one_file(file_path, config, plot=False, save_dir=None)
    except Exception as exc:
        return build_error_result(file_path, exc)


def plot_file_task(file_path_str: str, save_path_str: str, result: dict[str, Any], config: StabilityConfig) -> dict[str, Any]:
    file_path = Path(file_path_str)
    save_path = Path(save_path_str)
    try:
        plot_payload = load_plot_payload(file_path, config)
        plot_diagnostics(result, plot_payload, save_path)
        return {"ok": True, "save_path": str(save_path), "error": None}
    except Exception as exc:
        return {"ok": False, "save_path": str(save_path), "error": str(exc)}


def apply_default_run_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_dir or args.input_file:
        return args

    workspace = Path.cwd()
    default_input_dir = workspace / "npy_jobs"
    default_config = workspace / "config.yaml"
    default_output_dir = workspace / "out_full"

    if default_input_dir.exists():
        args.input_dir = default_input_dir
        args.output_dir = default_output_dir
        args.recursive = True
        args.plot = True
        if args.config is None and default_config.exists():
            args.config = default_config
        return args

    raise SystemExit(
        "未提供 --input_dir 或 --input_file，并且当前工作区下也没有默认目录 ./npy_jobs，无法直接运行。"
    )


def analyze_folder(
    input_dir: Path,
    config: StabilityConfig,
    recursive: bool = True,
    output_dir: Path | None = None,
    plot: bool = False,
    workers: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    files = sorted(input_dir.rglob("*.npy") if recursive else input_dir.glob("*.npy"))
    if output_dir is not None:
        ensure_dir(output_dir)
        clear_batch_outputs(output_dir)
    worker_count = resolve_worker_count(workers)
    LOGGER.info("批量分析启动: 样本数=%s, workers=%s, plot=%s", len(files), worker_count, plot)

    results: list[dict[str, Any]] = []
    if worker_count <= 1:
        for idx, file_path in enumerate(files, start=1):
            try:
                rel = file_path.relative_to(input_dir)
            except ValueError:
                rel = Path(file_path.name)
            LOGGER.info("分析样本 %s/%s: %s", idx, len(files), rel.as_posix())
            result = analyze_file_task(str(file_path), config)
            results.append(result)
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for idx, (file_path, result) in enumerate(zip(files, executor.map(analyze_file_task, (str(file_path) for file_path in files), repeat(config))), start=1):
                    try:
                        rel = file_path.relative_to(input_dir)
                    except ValueError:
                        rel = Path(file_path.name)
                    LOGGER.info("分析样本 %s/%s: %s", idx, len(files), rel.as_posix())
                    results.append(result)
        except (PermissionError, OSError) as exc:
            LOGGER.warning("多进程分析不可用，回退到单进程: %s", exc)
            results = []
            for idx, file_path in enumerate(files, start=1):
                try:
                    rel = file_path.relative_to(input_dir)
                except ValueError:
                    rel = Path(file_path.name)
                LOGGER.info("分析样本 %s/%s: %s", idx, len(files), rel.as_posix())
                result = analyze_file_task(str(file_path), config)
                results.append(result)

    clean_results = [strip_plot_data(result) for result in results]
    summary_rows = [result_to_summary_row(result, config) for result in clean_results]
    results_df = pd.DataFrame(summary_rows).sort_values("file").reset_index(drop=True)
    metrics = evaluate_against_labels(results_df)

    if output_dir is not None:
        mismatch_df = results_df[results_df["is_consistent"] == False].copy()
        mismatch_df.to_csv(output_dir / "mismatch_results.csv", index=False, encoding="utf-8-sig")
        if plot and not mismatch_df.empty:
            plot_jobs: list[tuple[Path, Path, dict[str, Any]]] = []
            result_by_file = {result["file"]: result for result in clean_results}
            for file_str in mismatch_df["file"].tolist():
                file_path = Path(file_str)
                try:
                    rel = file_path.relative_to(input_dir)
                except ValueError:
                    rel = Path(file_path.name)
                plot_jobs.append((file_path, output_dir / "mismatches_plots" / rel.with_suffix(".png"), result_by_file[file_str]))

            LOGGER.info("开始第二阶段批量绘图: 标签不一致样本数=%s", len(plot_jobs))
            if worker_count <= 1:
                for idx, (file_path, save_path, result) in enumerate(plot_jobs, start=1):
                    plot_status = plot_file_task(str(file_path), str(save_path), result, config)
                    if plot_status["ok"]:
                        LOGGER.info("绘图样本 %s/%s: %s", idx, len(plot_jobs), save_path.relative_to(output_dir).as_posix())
                    else:
                        LOGGER.error("样本绘图失败: %s, error=%s", file_path, plot_status["error"])
            else:
                try:
                    with ProcessPoolExecutor(max_workers=worker_count) as executor:
                        plot_iter = executor.map(
                            plot_file_task,
                            (str(file_path) for file_path, _, _ in plot_jobs),
                            (str(save_path) for _, save_path, _ in plot_jobs),
                            (result for _, _, result in plot_jobs),
                            repeat(config),
                        )
                        for idx, ((file_path, save_path, _), plot_status) in enumerate(zip(plot_jobs, plot_iter), start=1):
                            if plot_status["ok"]:
                                LOGGER.info("绘图样本 %s/%s: %s", idx, len(plot_jobs), save_path.relative_to(output_dir).as_posix())
                            else:
                                LOGGER.error("样本绘图失败: %s, error=%s", file_path, plot_status["error"])
                except (PermissionError, OSError) as exc:
                    LOGGER.warning("多进程绘图不可用，回退到单进程: %s", exc)
                    for idx, (file_path, save_path, result) in enumerate(plot_jobs, start=1):
                        plot_status = plot_file_task(str(file_path), str(save_path), result, config)
                        if plot_status["ok"]:
                            LOGGER.info("绘图样本 %s/%s: %s", idx, len(plot_jobs), save_path.relative_to(output_dir).as_posix())
                        else:
                            LOGGER.error("样本绘图失败: %s, error=%s", file_path, plot_status["error"])

    return results_df, clean_results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="暂态稳定性规则检测脚本")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input_dir", type=Path, help="输入样本文件夹")
    group.add_argument("--input_file", type=Path, help="输入单个 .npy 文件")
    parser.add_argument("--output_dir", type=Path, default=Path("out"), help="输出目录")
    parser.add_argument("--config", type=Path, help="YAML 配置文件")
    parser.add_argument("--plot", action="store_true", help="为样本保存诊断图")
    parser.add_argument("--recursive", action="store_true", help="递归分析文件夹")
    parser.add_argument("--log_level", default="INFO", help="日志等级")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数，0 表示自动")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(sys.argv) == 1:
        args = apply_default_run_args(args)
    setup_logging(args.log_level)
    config = load_config(args.config)
    recursive = args.recursive or config.recursive
    workers = args.workers if args.workers > 0 else config.workers
    ensure_dir(args.output_dir)
    if len(sys.argv) == 1:
        LOGGER.info(
            "检测到 VS Code 直接运行模式，使用默认参数: input_dir=%s output_dir=%s recursive=%s plot=%s workers=%s",
            args.input_dir,
            args.output_dir,
            args.recursive,
            args.plot,
            resolve_worker_count(workers),
        )

    if args.input_file:
        result = analyze_one_file(args.input_file, config, plot=args.plot, save_dir=args.output_dir / "single_plot.png" if args.plot else None)
        serializable = strip_plot_data(result)
        save_json(args.output_dir / "single_result.json", serializable)
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
        return

    results_df, _, metrics = analyze_folder(args.input_dir, config, recursive=recursive, output_dir=args.output_dir, plot=args.plot, workers=workers)
    LOGGER.info("分析完成，样本数=%s", len(results_df))
    if metrics:
        LOGGER.info(
            "评估指标: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )


if __name__ == "__main__":
    main()

