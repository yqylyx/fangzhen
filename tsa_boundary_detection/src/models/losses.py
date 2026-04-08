from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split('.'):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


class BoundaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, prob, 1.0 - prob)
        focal = ((1.0 - pt).clamp_min(1e-6) ** self.gamma) * bce
        return focal.mean()


def build_boundary_loss(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    # ??????? BCEWithLogitsLoss ? Focal Loss ?????
    # ????????????????
    pos_weight_value = float(_cfg_get(cfg, 'loss.resolved_pos_weight', _cfg_get(cfg, 'loss.pos_weight', 1.0)))
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    loss_type = str(_cfg_get(cfg, 'loss.loss_type', 'bce_logits')).strip().lower()
    if loss_type == 'focal':
        return BoundaryFocalLoss(
            gamma=float(_cfg_get(cfg, 'loss.focal_gamma', 2.0)),
            pos_weight=pos_weight,
        )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def total_loss(
    outputs: dict[str, Any],
    y: Tensor,
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    # ?????????????????? exit ? boundary logit ???????
    # ???????????????????
    y = y.float().view(-1)
    logits_list = outputs['logits_list']
    if not logits_list:
        raise ValueError('model did not return any boundary logits')
    criterion = build_boundary_loss(cfg, device=y.device)
    losses = [criterion(logits.view(-1), y) for logits in logits_list]
    total = torch.stack(losses).mean()
    loss_dict = {
        'loss': float(total.detach().cpu()),
        'loss_boundary': float(total.detach().cpu()),
    }
    return total, loss_dict
