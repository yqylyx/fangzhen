from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import TransientNormalizer


def load_npy_dict(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == object:
        payload = payload.item()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a dict-like npy sample, got {type(payload)!r}")
    return payload


def find_first(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def ensure_2d(array: Any, t_len: int | None = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape={arr.shape}")
    # 有些历史数据把时间维放在 axis=1，这里只在能明显修正时间长度不匹配时才转置。
    if t_len is not None and arr.shape[0] != t_len and arr.shape[1] == t_len:
        arr = arr.T
    return arr


def replace_invalid(arr: np.ndarray, sentinel: float) -> tuple[np.ndarray, np.ndarray]:
    # 缺失值不会被直接丢弃，而是通过 mask 显式记录，
    # 这样模型仍然可以处理可变长度、局部缺失的样本。
    valid = np.isfinite(arr) & (arr != sentinel)
    clean = np.where(valid, arr, np.nan).astype(np.float32)
    return clean, valid


def infer_dataset_name(path: Path) -> str:
    for part in path.parts:
        if part.endswith('data') and part[:-4].isdigit():
            return part[:-4]
    return path.parent.name


def collect_npy_files(root: str | Path, subdir: str) -> list[Path]:
    base = Path(root) / subdir
    if not base.exists():
        raise FileNotFoundError(f"dataset directory does not exist: {base}")
    return sorted(base.rglob('*.npy'))


def _first_valid_per_channel(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    refs = np.zeros((values.shape[1],), dtype=np.float32)
    for idx in range(values.shape[1]):
        col_valid = np.where(valid[:, idx])[0]
        if col_valid.size:
            refs[idx] = values[col_valid[0], idx]
    return refs


def preprocess_angles(values: np.ndarray, valid: np.ndarray, mode: str = 'initial') -> np.ndarray:
    processed = values.copy()
    mode = mode.lower()

    if mode in {'initial', 'initial_then_mean'}:
        # 先减去每个通道自己的初始偏置，让模型更关注相对功角演化，
        # 而不是任意的绝对参考零点。
        refs = _first_valid_per_channel(processed, valid)
        processed = processed - refs[None, :]

    if mode in {'mean', 'coi', 'initial_then_mean'}:
        # 再减去每个时刻的通道均值，近似构造一种惯量中心参考，
        # 让跨通道的功角分散模式更容易被学习到。
        masked = np.where(valid, processed, 0.0)
        counts = valid.sum(axis=1, keepdims=True).astype(np.float32)
        time_mean = np.divide(masked.sum(axis=1, keepdims=True), counts, out=np.zeros((processed.shape[0], 1), dtype=np.float32), where=counts > 0)
        processed = processed - time_mean

    if mode not in {'initial', 'mean', 'coi', 'initial_then_mean'}:
        raise ValueError(f"unsupported angle preprocessing mode: {mode}")
    return processed.astype(np.float32)


class NPYTransientDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        file_list: list[str | Path],
        sentinel: float = -99999.0,
        angle_preprocess: str = 'initial',
        normalizer: TransientNormalizer | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sentinel = float(sentinel)
        self.angle_preprocess = angle_preprocess
        self.normalizer = normalizer

        resolved: list[Path] = []
        for item in file_list:
            path = Path(item)
            if not path.is_absolute():
                candidate = path.resolve(strict=False)
                path = candidate if candidate.exists() else (self.root / path).resolve(strict=False)
            resolved.append(path)
        self.file_list = resolved

    def __len__(self) -> int:
        return len(self.file_list)

    def peek_label(self, idx: int) -> int:
        sample = load_npy_dict(self.file_list[idx])
        return int(find_first(sample, 'y', 'label'))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.file_list[idx]
        sample = load_npy_dict(path)

        # 同时兼容当前字段名和旧字段名，这样训练代码可以直接复用历史数据，
        # 不需要先做额外格式转换。
        y = int(find_first(sample, 'y', 'label'))
        t = find_first(sample, 't', 'times')
        V = find_first(sample, 'V', 'voltages')
        delta = find_first(sample, 'delta', 'angles')
        meta = find_first(sample, 'meta', default={}) or {}

        if V is None or delta is None:
            raise KeyError(f"{path} must contain V/voltages and delta/angles")

        if t is None:
            t_len = np.asarray(V).shape[0]
            t_arr = np.arange(t_len, dtype=np.float32)
        else:
            t_arr = np.asarray(t, dtype=np.float32).reshape(-1)

        V_arr = ensure_2d(V, t_len=t_arr.shape[0])
        delta_arr = ensure_2d(delta, t_len=t_arr.shape[0])
        if V_arr.shape[0] != t_arr.shape[0] or delta_arr.shape[0] != t_arr.shape[0]:
            raise ValueError(
                f"time length mismatch for {path}: t={t_arr.shape[0]}, "
                f"V={V_arr.shape}, delta={delta_arr.shape}"
            )

        V_clean, mask_V = replace_invalid(V_arr, self.sentinel)
        delta_clean, mask_delta = replace_invalid(delta_arr, self.sentinel)
        delta_ref = preprocess_angles(delta_clean, mask_delta, mode=self.angle_preprocess)

        # raw_* 保留物理量纲，供后面的物理特征构造使用；
        # 而送进神经网络的 V/delta 则可以是归一化后的版本。
        V_zero = np.where(mask_V, np.nan_to_num(V_clean, nan=0.0), 0.0).astype(np.float32)
        delta_zero = np.where(mask_delta, np.nan_to_num(delta_ref, nan=0.0), 0.0).astype(np.float32)

        if self.normalizer is None:
            V_model = V_zero.copy()
            delta_model = delta_zero.copy()
        else:
            V_model, delta_model = self.normalizer.transform(
                V_clean=np.where(mask_V, V_clean, np.nan),
                delta_clean=np.where(mask_delta, delta_ref, np.nan),
                mask_V=mask_V,
                mask_delta=mask_delta,
            )

        time_mask = np.logical_or(mask_V.any(axis=1), mask_delta.any(axis=1))
        dataset_name = infer_dataset_name(path)

        return {
            'file': str(path),
            'sample_id': path.stem,
            'dataset_name': dataset_name,
            't': torch.from_numpy(t_arr),
            'V': torch.from_numpy(V_model.astype(np.float32)),
            'delta': torch.from_numpy(delta_model.astype(np.float32)),
            'raw_V': torch.from_numpy(V_zero),
            'raw_delta': torch.from_numpy(delta_zero),
            'mask_V': torch.from_numpy(mask_V.astype(np.bool_)),
            'mask_delta': torch.from_numpy(mask_delta.astype(np.bool_)),
            'ch_mask_V': torch.ones((V_arr.shape[1],), dtype=torch.bool),
            'ch_mask_delta': torch.ones((delta_arr.shape[1],), dtype=torch.bool),
            'time_mask': torch.from_numpy(time_mask.astype(np.bool_)),
            'y': torch.tensor(y, dtype=torch.long),
            'meta': meta,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError('empty batch is not supported')

    batch_size = len(batch)
    max_t = max(item['V'].shape[0] for item in batch)
    max_nv = max(item['V'].shape[1] for item in batch)
    max_ng = max(item['delta'].shape[1] for item in batch)

    def alloc_float(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32)

    def alloc_bool(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bool)

    V = alloc_float((batch_size, max_t, max_nv))
    delta = alloc_float((batch_size, max_t, max_ng))
    raw_V = alloc_float((batch_size, max_t, max_nv))
    raw_delta = alloc_float((batch_size, max_t, max_ng))
    mask_V = alloc_bool((batch_size, max_t, max_nv))
    mask_delta = alloc_bool((batch_size, max_t, max_ng))
    ch_mask_V = alloc_bool((batch_size, max_nv))
    ch_mask_delta = alloc_bool((batch_size, max_ng))
    time_mask = alloc_bool((batch_size, max_t))
    t = alloc_float((batch_size, max_t))
    y = torch.zeros((batch_size,), dtype=torch.long)

    files: list[str] = []
    sample_ids: list[str] = []
    dataset_names: list[str] = []
    metas: list[dict[str, Any]] = []

    for row, item in enumerate(batch):
        cur_t, cur_nv = item['V'].shape
        _, cur_ng = item['delta'].shape

        # 把每个样本补齐到当前 batch 里的最大时间长度和通道数，
        # 同时保留对应 mask，方便后续模块忽略 padding 区域。
        V[row, :cur_t, :cur_nv] = item['V']
        delta[row, :cur_t, :cur_ng] = item['delta']
        raw_V[row, :cur_t, :cur_nv] = item['raw_V']
        raw_delta[row, :cur_t, :cur_ng] = item['raw_delta']
        mask_V[row, :cur_t, :cur_nv] = item['mask_V']
        mask_delta[row, :cur_t, :cur_ng] = item['mask_delta']
        ch_mask_V[row, :cur_nv] = item['ch_mask_V']
        ch_mask_delta[row, :cur_ng] = item['ch_mask_delta']
        time_mask[row, :cur_t] = item['time_mask']
        t[row, : item['t'].shape[0]] = item['t']
        y[row] = item['y']

        files.append(item['file'])
        sample_ids.append(item['sample_id'])
        dataset_names.append(item['dataset_name'])
        metas.append(item['meta'])

    # 只要任意一个模态在这个时刻有观测，就把这个时间步视为有效。
    time_mask = time_mask | mask_V.any(dim=-1) | mask_delta.any(dim=-1)

    return {
        'files': files,
        'sample_ids': sample_ids,
        'dataset_names': dataset_names,
        'meta': metas,
        't': t,
        'V': V,
        'delta': delta,
        'raw_V': raw_V,
        'raw_delta': raw_delta,
        'mask_V': mask_V,
        'mask_delta': mask_delta,
        'ch_mask_V': ch_mask_V,
        'ch_mask_delta': ch_mask_delta,
        'time_mask': time_mask,
        'y': y,
    }

