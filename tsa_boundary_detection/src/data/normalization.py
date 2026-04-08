from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TransientNormalizer:
    def __init__(
        self,
        method: str = 'robust',
        eps: float = 1e-6,
        voltage_center: float = 0.0,
        voltage_scale: float = 1.0,
        delta_center: float = 0.0,
        delta_scale: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.eps = float(eps)
        self.voltage_center = float(voltage_center)
        self.voltage_scale = float(max(voltage_scale, eps))
        self.delta_center = float(delta_center)
        self.delta_scale = float(max(delta_scale, eps))
        self.metadata = metadata or {}

    @staticmethod
    def _downsample(values: np.ndarray, max_values: int, rng: np.random.Generator) -> np.ndarray:
        if values.size <= max_values:
            return values.astype(np.float32, copy=False)
        idx = rng.choice(values.size, size=max_values, replace=False)
        return values[idx].astype(np.float32, copy=False)

    @staticmethod
    def _fit_stats(values: np.ndarray, method: str, eps: float) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0
        values = values.astype(np.float64, copy=False)
        if method == 'standard':
            center = float(np.mean(values))
            scale = float(np.std(values))
        else:
            center = float(np.median(values))
            q1, q3 = np.percentile(values, [25, 75])
            scale = float(q3 - q1)
        return center, max(scale, eps)

    @classmethod
    def fit_from_dataset(
        cls,
        dataset,
        method: str = 'robust',
        eps: float = 1e-6,
        metadata: dict[str, Any] | None = None,
        max_values_per_modality: int = 500000,
        sample_seed: int = 42,
    ) -> 'TransientNormalizer':
        rng = np.random.default_rng(sample_seed)
        v_chunks: list[np.ndarray] = []
        d_chunks: list[np.ndarray] = []
        per_sample_cap = max(1024, max_values_per_modality // max(1, len(dataset)))

        for idx in range(len(dataset)):
            sample = dataset[idx]
            mask_v = sample['mask_V'].numpy()
            mask_d = sample['mask_delta'].numpy()
            raw_v = sample['raw_V'].numpy()
            raw_d = sample['raw_delta'].numpy()
            if mask_v.any():
                v_vals = raw_v[mask_v]
                v_chunks.append(cls._downsample(v_vals, min(per_sample_cap, max_values_per_modality), rng))
            if mask_d.any():
                d_vals = raw_d[mask_d]
                d_chunks.append(cls._downsample(d_vals, min(per_sample_cap, max_values_per_modality), rng))

        all_v = np.concatenate(v_chunks) if v_chunks else np.array([0.0], dtype=np.float32)
        all_d = np.concatenate(d_chunks) if d_chunks else np.array([0.0], dtype=np.float32)
        all_v = cls._downsample(all_v, max_values_per_modality, rng)
        all_d = cls._downsample(all_d, max_values_per_modality, rng)

        v_center, v_scale = cls._fit_stats(all_v, method=method, eps=eps)
        d_center, d_scale = cls._fit_stats(all_d, method=method, eps=eps)
        metadata = dict(metadata or {})
        metadata.update(
            {
                'max_values_per_modality': int(max_values_per_modality),
                'sample_seed': int(sample_seed),
                'num_sampled_voltage_values': int(all_v.size),
                'num_sampled_delta_values': int(all_d.size),
            }
        )
        return cls(
            method=method,
            eps=eps,
            voltage_center=v_center,
            voltage_scale=v_scale,
            delta_center=d_center,
            delta_scale=d_scale,
            metadata=metadata,
        )

    def transform(
        self,
        V_clean: np.ndarray,
        delta_clean: np.ndarray,
        mask_V: np.ndarray,
        mask_delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        V_norm = (V_clean - self.voltage_center) / self.voltage_scale
        delta_norm = (delta_clean - self.delta_center) / self.delta_scale
        V_norm = np.where(mask_V, np.nan_to_num(V_norm, nan=0.0), 0.0).astype(np.float32)
        delta_norm = np.where(mask_delta, np.nan_to_num(delta_norm, nan=0.0), 0.0).astype(np.float32)
        return V_norm, delta_norm

    def to_dict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'eps': self.eps,
            'voltage_center': self.voltage_center,
            'voltage_scale': self.voltage_scale,
            'delta_center': self.delta_center,
            'delta_scale': self.delta_scale,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TransientNormalizer':
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'TransientNormalizer':
        with Path(path).open('r', encoding='utf-8') as fh:
            payload = json.load(fh)
        return cls.from_dict(payload)
