from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

MPL_CACHE_DIR = Path(__file__).resolve().parent / '_mpl_cache'
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FONT_CANDIDATES = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'Arial Unicode MS',
    'DejaVu Sans',
]


def setup_matplotlib() -> None:
    plt.rcParams['font.sans-serif'] = FONT_CANDIDATES
    plt.rcParams['axes.unicode_minus'] = False


def _safe_filename(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_') or 'sample'


def build_plot_path(output_dir: str | Path, dataset_name: str, file_rel: str) -> Path:
    dataset_dir = Path(output_dir) / 'plots' / str(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(file_rel.replace('/', '__')) + '.png'
    return dataset_dir / filename


def _series_alpha(num_channels: int, configured_alpha: float) -> float:
    if num_channels <= 0:
        return configured_alpha
    return max(0.12, min(configured_alpha, 10.0 / float(num_channels)))


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return '-'
    if not np.isfinite(value):
        return '-'
    return f'{value:.{digits}f}'


def _resolve_rank_text(row: Dict[str, Any]) -> str:
    for key, label in [
        ('overall_rank', '综合'),
        ('category_rank', '类别'),
        ('dataset_rank', '数据集'),
        ('boundary_rank', '兼容'),
    ]:
        value = row.get(key)
        try:
            value = int(value)
        except Exception:
            continue
        if value > 0:
            return f'{label}排名={value}'
    return '排名=n/a'


def save_sample_plot(sample: Dict[str, Any], row: Dict[str, Any], output_path: str | Path, config: Dict[str, Any]) -> Path:
    setup_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_cfg = config['visualization']
    times = np.asarray(sample['times'], dtype=np.float64)
    voltages = np.asarray(sample['voltages'], dtype=np.float64)
    angles_rel = np.asarray(sample['angles_rel'], dtype=np.float64)
    spread_t = np.asarray(sample.get('spread_t', np.array([])), dtype=np.float64)
    min_voltage_t = np.asarray(sample.get('min_voltage_t', np.array([])), dtype=np.float64)

    figsize = vis_cfg.get('figsize', [14, 8.5])
    dpi = int(vis_cfg.get('dpi', 110))
    width_ratios = [1.0, float(vis_cfg.get('info_panel_width_ratio', 0.48))]
    height_ratios = [1.1, 1.1, float(vis_cfg.get('aux_panel_height_ratio', 0.65))]

    fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
    grid = fig.add_gridspec(3, 2, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.18, hspace=0.18)
    ax_v = fig.add_subplot(grid[0, 0])
    ax_a = fig.add_subplot(grid[1, 0], sharex=ax_v)
    ax_aux = fig.add_subplot(grid[2, 0], sharex=ax_v)
    ax_info = fig.add_subplot(grid[:, 1])

    alpha_v = _series_alpha(voltages.shape[1], float(vis_cfg.get('plot_alpha_voltage', 0.16)))
    alpha_a = _series_alpha(angles_rel.shape[1], float(vis_cfg.get('plot_alpha_angle', 0.18)))

    for idx in range(voltages.shape[1]):
        ax_v.plot(times, voltages[:, idx], color='#0b57a4', linewidth=0.95, alpha=alpha_v)
    ax_v.axhline(0.9, color='#c62828', linestyle='--', linewidth=1.0, alpha=0.85)
    ax_v.axhline(0.85, color='#ef6c00', linestyle=':', linewidth=1.0, alpha=0.85)
    ax_v.set_ylabel('Voltage (p.u.)')
    ax_v.set_ylim(vis_cfg.get('voltage_ylim', [0.5, 1.15]))
    ax_v.grid(alpha=0.25)

    for idx in range(angles_rel.shape[1]):
        ax_a.plot(times, angles_rel[:, idx], color='#1f7a3a', linewidth=0.95, alpha=alpha_a)
    ax_a.axhline(120.0, color='#6a3dad', linestyle='--', linewidth=1.0, alpha=0.75)
    ax_a.axhline(-120.0, color='#6a3dad', linestyle='--', linewidth=1.0, alpha=0.75)
    ax_a.set_ylabel('相对功角 (deg)')
    ax_a.grid(alpha=0.25)

    if spread_t.size:
        ax_aux.plot(times, spread_t, color='#6a3dad', linewidth=1.3, label='spread_t')
    if min_voltage_t.size:
        ax_aux.plot(times, min_voltage_t, color='#d97706', linewidth=1.2, label='min_voltage_t')
    ax_aux.axhline(0.9, color='#c62828', linestyle=':', linewidth=0.9, alpha=0.55)
    ax_aux.set_ylabel('辅助曲线')
    ax_aux.set_xlabel('时间 (s)')
    ax_aux.grid(alpha=0.25)
    if ax_aux.lines:
        ax_aux.legend(loc='upper right', fontsize=8)

    title = (
        f"{Path(str(row['file'])).name} | 数据集={row.get('dataset_name', '-')} | "
        f"原始标签={row.get('original_label', '-')} | seed={int(row.get('is_seed_boundary', 0))} | "
        f"总分={_fmt(row.get('overall_candidate_score'), 4)} | 类别={row.get('category_label', '-')} | {_resolve_rank_text(row)}"
    )
    fig.suptitle(title, fontsize=11.5)

    ax_info.axis('off')
    ax_info.set_facecolor('#fbfcff')
    text_lines = [
        '样本摘要',
        f"文件: {Path(str(row.get('file', ''))).name}",
        f"数据集: {row.get('dataset_name', '-')}",
        f"原始标签: {row.get('original_label', '-')}",
        f"是否 seed 边界: {int(row.get('is_seed_boundary', 0))}",
        f"候选类别: {row.get('category_label', '-')}",
        f"总候选分数: {_fmt(row.get('overall_candidate_score'))}",
        '',
        '关键指标',
        f"tail_voltage_min: {_fmt(row.get('tail_voltage_min'))}",
        f"final_recovered_ratio_0_9: {_fmt(row.get('final_recovered_ratio_0_9'))}",
        f"tail_spread_mean: {_fmt(row.get('tail_spread_mean'), 2)}",
        f"tail_spread_slope: {_fmt(row.get('tail_spread_slope'), 2)}",
        f"angle_speed_median: {_fmt(row.get('angle_speed_median'), 2)}",
        f"tail_amp_top1: {_fmt(row.get('tail_amp_top1'), 2)}",
        f"tail_amp_top2: {_fmt(row.get('tail_amp_top2'), 2)}",
        f"large_amp_channel_count_20: {_fmt(row.get('large_amp_channel_count_20'), 0)}",
        f"seed_similarity_score: {_fmt(row.get('seed_similarity_score'))}",
        '',
        '补充信息',
        f"tail_low_voltage_reentry_count: {_fmt(row.get('tail_low_voltage_reentry_count'), 0)}",
        f"spread_reentry_count: {_fmt(row.get('spread_reentry_count'), 0)}",
        f"oscillation_persistence_score: {_fmt(row.get('oscillation_persistence_score'))}",
    ]
    ax_info.text(
        0.02,
        0.98,
        '\n'.join(text_lines),
        va='top',
        ha='left',
        fontsize=9.1,
        linespacing=1.45,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.96, 'edgecolor': '#d9e2f2'},
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.07, wspace=0.18, hspace=0.22)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path
