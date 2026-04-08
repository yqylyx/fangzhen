from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch
from torch import Tensor


class StreamInfer:
    def __init__(
        self,
        model: torch.nn.Module,
        patch_len: int,
        step: int,
        boundary_threshold: float = 0.5,
        exit_threshold: float = 0.98,
        device: str = 'cuda',
        max_steps: int = 2048,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.patch_len = int(patch_len)
        self.step = max(1, int(step))
        self.boundary_threshold = float(boundary_threshold)
        self.exit_threshold = float(exit_threshold)
        self.device = torch.device(device)
        self.max_steps = max(self.patch_len, int(max_steps))

        self.V_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.delta_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.mask_V_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.mask_delta_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.counter = 0

    def _prepare_step(self, value: Any) -> Tensor:
        if torch.is_tensor(value):
            return value.detach().float().cpu().flatten()
        return torch.as_tensor(np.asarray(value), dtype=torch.float32).flatten()

    def _normalize_masks(self, V_t: Tensor, delta_t: Tensor, mask_t: Any | None) -> tuple[Tensor, Tensor]:
        if mask_t is None:
            return torch.isfinite(V_t), torch.isfinite(delta_t)
        if isinstance(mask_t, dict):
            return (
                torch.as_tensor(mask_t.get('V'), dtype=torch.bool).flatten(),
                torch.as_tensor(mask_t.get('delta'), dtype=torch.bool).flatten(),
            )
        if isinstance(mask_t, (tuple, list)) and len(mask_t) == 2:
            return (
                torch.as_tensor(mask_t[0], dtype=torch.bool).flatten(),
                torch.as_tensor(mask_t[1], dtype=torch.bool).flatten(),
            )

        flat = torch.as_tensor(mask_t, dtype=torch.bool).flatten()
        if flat.numel() != V_t.numel() + delta_t.numel():
            raise ValueError('flat mask_t must match Nv + Ng')
        return flat[: V_t.numel()], flat[V_t.numel() :]

    def _build_batch(self) -> dict[str, Tensor]:
        V = torch.stack(list(self.V_buffer), dim=0).unsqueeze(0)
        delta = torch.stack(list(self.delta_buffer), dim=0).unsqueeze(0)
        mask_V = torch.stack(list(self.mask_V_buffer), dim=0).unsqueeze(0)
        mask_delta = torch.stack(list(self.mask_delta_buffer), dim=0).unsqueeze(0)
        ch_mask_V = mask_V.any(dim=1)
        ch_mask_delta = mask_delta.any(dim=1)
        time_mask = mask_V.any(dim=-1) | mask_delta.any(dim=-1)
        return {
            'V': V.to(self.device),
            'delta': delta.to(self.device),
            'mask_V': mask_V.to(self.device),
            'mask_delta': mask_delta.to(self.device),
            'ch_mask_V': ch_mask_V.to(self.device),
            'ch_mask_delta': ch_mask_delta.to(self.device),
            'time_mask': time_mask.to(self.device),
        }

    def update(self, V_t: Any, delta_t: Any, mask_t: Any | None = None) -> tuple[int, float, int, dict[str, Any]] | None:
        V_step = self._prepare_step(V_t)
        delta_step = self._prepare_step(delta_t)
        mask_v, mask_d = self._normalize_masks(V_step, delta_step, mask_t)

        V_step = torch.where(mask_v, torch.nan_to_num(V_step, nan=0.0), torch.zeros_like(V_step))
        delta_step = torch.where(mask_d, torch.nan_to_num(delta_step, nan=0.0), torch.zeros_like(delta_step))

        self.V_buffer.append(V_step)
        self.delta_buffer.append(delta_step)
        self.mask_V_buffer.append(mask_v)
        self.mask_delta_buffer.append(mask_d)
        self.counter += 1

        if self.counter % self.step != 0:
            return None
        return self.predict()

    def predict(self) -> tuple[int, float, int, dict[str, Any]]:
        batch = self._build_batch()
        with torch.no_grad():
            outputs = self.model(**batch, return_attn=True)

        logits_list = outputs['logits_list']
        chosen = len(logits_list) - 1
        chosen_prob = None
        exit_confidences: list[float] = []
        exit_boundary_prob: list[float] = []

        for idx, logits in enumerate(logits_list):
            prob = float(torch.sigmoid(logits)[0].item())
            conf = max(prob, 1.0 - prob)
            exit_confidences.append(conf)
            exit_boundary_prob.append(prob)
            if conf >= self.exit_threshold:
                chosen = idx
                chosen_prob = prob
                break

        if chosen_prob is None:
            chosen_prob = exit_boundary_prob[chosen]

        pred = int(chosen_prob >= self.boundary_threshold)
        attn = outputs.get('attn', {})
        explanation_stub = {
            'exit_confidences': exit_confidences,
            'exit_boundary_prob': exit_boundary_prob,
            'channel_importance': attn.get('channel_importance'),
            'fusion_gate': attn.get('fusion_gate'),
        }
        return pred, chosen_prob, chosen, explanation_stub
