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
    return max(0.16, min(configured_alpha, 10.0 / float(num_channels)))


def save_sample_plot(sample: Dict[str, Any], row: Dict[str, Any], output_path: str | Path, config: Dict[str, Any]) -> Path:
    setup_matplotlib()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_cfg = config['visualization']
    times = np.asarray(sample['times'], dtype=np.float64)
    voltages = np.asarray(sample['voltages'], dtype=np.float64)
    angles_rel = np.asarray(sample['angles_rel'], dtype=np.float64)

    figsize = vis_cfg.get('figsize', [12, 8])
    dpi = int(vis_cfg.get('dpi', 110))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(figsize[0], figsize[1]), dpi=dpi)
    ax_v, ax_a = axes

    alpha_v = _series_alpha(voltages.shape[1], float(vis_cfg.get('plot_alpha_voltage', 0.18)))
    alpha_a = _series_alpha(angles_rel.shape[1], float(vis_cfg.get('plot_alpha_angle', 0.22)))

    for idx in range(voltages.shape[1]):
        ax_v.plot(times, voltages[:, idx], color='#0b57a4', linewidth=1.0, alpha=alpha_v)
    ax_v.axhline(0.9, color='#c62828', linestyle='--', linewidth=1.2, alpha=0.9)
    ax_v.axhline(0.85, color='#ef6c00', linestyle=':', linewidth=1.2, alpha=0.9)
    ax_v.set_ylabel('Voltage (p.u.)')
    ax_v.set_ylim(vis_cfg.get('voltage_ylim', [0.5, 1.15]))
    ax_v.grid(alpha=0.25)

    for idx in range(angles_rel.shape[1]):
        ax_a.plot(times, angles_rel[:, idx], color='#1f7a3a', linewidth=1.0, alpha=alpha_a)
    ax_a.axhline(120.0, color='#6a3dad', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_a.axhline(-120.0, color='#6a3dad', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_a.set_ylabel('Relative angle (deg)')
    ax_a.set_xlabel('Time (s)')
    ax_a.grid(alpha=0.25)

    title = (
        f"{Path(row['file']).name} | dataset={row['dataset_name']} | label={row['original_label']} | "
        f"score={row['boundary_score']:.4f} | rank={int(row['boundary_rank'])}"
    )
    fig.suptitle(title, fontsize=12)

    text_lines = [
        f"tail_voltage_min: {row['tail_voltage_min']:.4f}",
        f"final_recovered_ratio_0_9: {row['final_recovered_ratio_0_9']:.4f}",
        f"tail_spread_mean: {row['tail_spread_mean']:.2f}",
        f"tail_spread_slope: {row['tail_spread_slope']:.2f}",
        f"angle_speed_median: {row['angle_speed_median']:.2f}",
        f"boundary_side: {row.get('boundary_side', 'general')}",
        f"stable_side_flag: {int(row.get('stable_side_boundary_flag', 0))}",
        f"high_voltage_fast_flag: {int(row.get('high_voltage_fast_flag', 0))}",
        f"dynamic_unstable_flag: {int(row.get('dynamic_unstable_flag', 0))}",
        f"unstable_side_flag: {int(row.get('unstable_side_boundary_flag', 0))}",
        f"stable_side_signal: {float(row.get('stable_side_signal', float('nan'))):.3f}",
        f"high_voltage_fast_signal: {float(row.get('high_voltage_fast_signal', float('nan'))):.3f}",
        f"unstable_side_signal: {float(row.get('unstable_side_signal', float('nan'))):.3f}",
    ]
    ax_v.text(
        0.01,
        0.01,
        '\n'.join(text_lines),
        transform=ax_v.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#cccccc'},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path
