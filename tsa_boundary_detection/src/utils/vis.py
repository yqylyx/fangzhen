from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault('MPLCONFIGDIR', str((Path.cwd() / '.matplotlib_cache').resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _extract_attention_map(attn: Any) -> np.ndarray:
    if isinstance(attn, dict):
        if attn.get('graph'):
            graph = attn['graph'][-1]
            arr = _to_numpy(graph)
            if arr.ndim == 4:
                return arr[0].mean(axis=0)
        if attn.get('temporal_V'):
            temporal = attn['temporal_V'][-1]
            arr = _to_numpy(temporal)
            if arr.ndim == 5:
                arr = arr[0].mean(axis=1)
                return arr.mean(axis=0)
        if 'channel_importance' in attn:
            arr = _to_numpy(attn['channel_importance'])
            if arr.ndim == 2:
                return arr[:1]
    arr = _to_numpy(attn)
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    return arr


def save_attention_heatmap(
    attn: Any,
    out_png: str | Path,
    dpi: int = 120,
    max_channels: int = 32,
    max_patches: int = 128,
    title: str | None = None,
) -> None:
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = _extract_attention_map(attn)
    if heatmap.ndim != 2:
        heatmap = np.atleast_2d(heatmap)
    heatmap = heatmap[:max_channels, :max_patches]

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect='auto', cmap='viridis')
    plt.colorbar(shrink=0.85)
    plt.title(title or 'Attention / Importance Heatmap')
    plt.xlabel('Patch / Node')
    plt.ylabel('Channel / Query')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_waveforms(
    V: Any,
    delta: Any,
    out_png: str | Path,
    dpi: int = 120,
    max_channels: int = 16,
    downsample: int = 2,
    title: str | None = None,
    time_values: Any | None = None,
) -> None:
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    V_np = _to_numpy(V)
    delta_np = _to_numpy(delta)
    if V_np.ndim != 2 or delta_np.ndim != 2:
        raise ValueError('save_waveforms expects V and delta with shape [T, C]')

    step = max(1, int(downsample))
    V_np = V_np[::step, :max_channels]
    delta_np = delta_np[::step, :max_channels]
    if time_values is None:
        time = np.arange(V_np.shape[0], dtype=np.float32)
    else:
        time = _to_numpy(time_values).reshape(-1)[::step]
        time = time[: V_np.shape[0]]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, V_np, linewidth=1.0)
    axes[0].set_title('Voltage Waveforms')
    axes[0].set_ylabel('V (p.u.)')
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, delta_np, linewidth=1.0)
    axes[1].set_title('Rotor Angle Waveforms')
    axes[1].set_ylabel('delta')
    axes[1].set_xlabel('Time')
    axes[1].grid(alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
