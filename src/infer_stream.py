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
        exit_threshold: float = 0.98,
        device: str = "cuda",
        max_steps: int = 2048,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.patch_len = int(patch_len)
        self.step = max(1, int(step))
        self.exit_threshold = float(exit_threshold)
        self.device = torch.device(device)
        self.max_steps = max(self.patch_len, int(max_steps))

        # 用滑动缓冲区持续积累最新时刻的 V / delta 观测，随时组装成在线推理窗口。
        self.V_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.delta_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.mask_V_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.mask_delta_buffer: deque[Tensor] = deque(maxlen=self.max_steps)
        self.counter = 0

    def _prepare_step(self, value: Any) -> Tensor:
        # 把单个时刻的输入统一整理成一维 float 张量，兼容 numpy / torch / list。
        if torch.is_tensor(value):
            return value.detach().float().cpu().flatten()
        return torch.as_tensor(np.asarray(value), dtype=torch.float32).flatten()

    def _normalize_masks(
        self,
        V_t: Tensor,
        delta_t: Tensor,
        mask_t: Any | None,
    ) -> tuple[Tensor, Tensor]:
        # 在线输入允许多种 mask 形式，这里统一拆成电压和功角两套布尔掩码。
        if mask_t is None:
            return torch.isfinite(V_t), torch.isfinite(delta_t)
        if isinstance(mask_t, dict):
            return (
                torch.as_tensor(mask_t.get("V"), dtype=torch.bool).flatten(),
                torch.as_tensor(mask_t.get("delta"), dtype=torch.bool).flatten(),
            )
        if isinstance(mask_t, (tuple, list)) and len(mask_t) == 2:
            return (
                torch.as_tensor(mask_t[0], dtype=torch.bool).flatten(),
                torch.as_tensor(mask_t[1], dtype=torch.bool).flatten(),
            )

        flat = torch.as_tensor(mask_t, dtype=torch.bool).flatten()
        if flat.numel() != V_t.numel() + delta_t.numel():
            raise ValueError("flat mask_t must match Nv + Ng")
        return flat[: V_t.numel()], flat[V_t.numel() :]

    def _build_batch(self) -> dict[str, Tensor]:
        # 把缓冲区内容拼成和离线训练一致的 batch 结构，便于直接复用模型前向。
        V = torch.stack(list(self.V_buffer), dim=0).unsqueeze(0)
        delta = torch.stack(list(self.delta_buffer), dim=0).unsqueeze(0)
        mask_V = torch.stack(list(self.mask_V_buffer), dim=0).unsqueeze(0)
        mask_delta = torch.stack(list(self.mask_delta_buffer), dim=0).unsqueeze(0)
        ch_mask_V = mask_V.any(dim=1)
        ch_mask_delta = mask_delta.any(dim=1)
        time_mask = mask_V.any(dim=-1) | mask_delta.any(dim=-1)
        return {
            "V": V.to(self.device),
            "delta": delta.to(self.device),
            "mask_V": mask_V.to(self.device),
            "mask_delta": mask_delta.to(self.device),
            "ch_mask_V": ch_mask_V.to(self.device),
            "ch_mask_delta": ch_mask_delta.to(self.device),
            "time_mask": time_mask.to(self.device),
        }

    def update(self, V_t: Any, delta_t: Any, mask_t: Any | None = None) -> tuple[int, float, float, int, dict[str, Any]] | None:
        # 每接入一个新时刻就更新缓冲区；只有走到指定步长时才真正触发一次预测。
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

    def predict(self) -> tuple[int, float, float, int, dict[str, Any]]:
        # 逐出口检查置信度，满足阈值就提前退出，否则退化为最后一个出口的结果。
        batch = self._build_batch()
        with torch.no_grad():
            outputs = self.model(**batch, return_attn=True)

        logits_list = outputs["logits_list"]
        risk_list = outputs["risk_list"]
        chosen = len(logits_list) - 1
        chosen_cls = None
        chosen_conf = None
        exit_confidences: list[float] = []
        exit_unstable_prob: list[float] = []

        for idx, logits in enumerate(logits_list):
            probs = torch.softmax(logits, dim=-1)[0]
            conf = float(probs.max().item())
            pred = int(probs.argmax().item())
            exit_confidences.append(conf)
            exit_unstable_prob.append(float(probs[1].item()))
            if conf >= self.exit_threshold:
                chosen = idx
                chosen_cls = pred
                chosen_conf = conf
                break

        if chosen_cls is None or chosen_conf is None:
            final_probs = torch.softmax(logits_list[chosen], dim=-1)[0]
            chosen_cls = int(final_probs.argmax().item())
            chosen_conf = float(final_probs.max().item())

        risk = float(risk_list[chosen][0, 0].item())
        attn = outputs.get("attn", {})
        explanation_stub = {
            "exit_confidences": exit_confidences,
            "exit_unstable_prob": exit_unstable_prob,
            "channel_importance": attn.get("channel_importance"),
            "fusion_gate": attn.get("fusion_gate"),
        }
        return chosen_cls, chosen_conf, risk, chosen, explanation_stub
