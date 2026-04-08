from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def supervised_contrastive_loss(
    emb: Tensor,
    y: Tensor,
    risk: Tensor | None = None,
    temperature: float = 0.07,
    hard_neg: bool = True,
) -> Tensor:
    if emb.ndim != 2 or emb.shape[0] <= 1:
        return emb.new_tensor(0.0)

    # 把同类样本在嵌入空间里拉近，把异类样本推远，
    # 同时对靠近决策边界的难样本给予更高权重。
    emb = F.normalize(emb, dim=-1)
    sim = torch.matmul(emb, emb.T) / temperature
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()

    labels = y.view(-1, 1)
    same = torch.eq(labels, labels.T)
    eye = torch.eye(emb.shape[0], device=emb.device, dtype=torch.bool)
    pos_mask = same & ~eye
    if pos_mask.sum() == 0:
        return emb.new_tensor(0.0)
    neg_mask = ~same & ~eye

    pair_weight = torch.ones_like(sim)
    if risk is not None:
        risk_vec = risk.view(-1, 1)
        risk_dist = torch.abs(risk_vec - risk_vec.T)
        pos_weight = torch.exp(-risk_dist)
        neg_weight = 1.0 + torch.exp(-risk_dist) if hard_neg else torch.ones_like(risk_dist)
        pair_weight = torch.where(pos_mask, pos_weight, pair_weight)
        pair_weight = torch.where(neg_mask, neg_weight, pair_weight)

    exp_sim = torch.exp(sim) * (~eye).float() * pair_weight
    log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    mean_log_prob_pos = (pair_weight * pos_mask.float() * log_prob).sum(dim=-1) / pos_mask.float().sum(dim=-1).clamp_min(1.0)
    return (-mean_log_prob_pos.mean()).clamp_min(0.0)


def _masked_mean(x: Tensor, mask: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    mask_f = mask.float()
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    num = (x * mask_f).sum(dim=dim, keepdim=keepdim)
    den = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return num / den


def _masked_spread(delta: Tensor, mask: Tensor) -> Tensor:
    neg_inf = torch.full_like(delta, float("-inf"))
    pos_inf = torch.full_like(delta, float("inf"))
    d_max = torch.where(mask, delta, neg_inf).amax(dim=-1)
    d_min = torch.where(mask, delta, pos_inf).amin(dim=-1)
    valid = mask.any(dim=-1)
    spread = torch.where(valid, d_max - d_min, delta.new_zeros(()).expand_as(d_max))
    return spread


def _angle_to_deg(delta: Tensor, mask: Tensor) -> Tensor:
    if mask.sum() == 0:
        return delta
    q95 = torch.quantile(torch.abs(delta[mask]), 0.95) if mask.any() else delta.new_tensor(0.0)
    scale = 57.2957795 if q95 < 8.0 else 1.0
    return delta * scale


def physics_risk_features(
    V: Tensor,
    delta: Tensor,
    mask_V: Tensor,
    mask_delta: Tensor,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    # 这里构造一组基于物理经验的风险特征，作为硬标签之外的软风险补充。
    thresholds = thresholds or {"v_warn": 0.9, "v_danger": 0.85, "angle_warn_deg": 120.0, "angle_danger_deg": 180.0}
    v_warn = thresholds.get("v_warn", 0.9)
    v_danger = thresholds.get("v_danger", 0.85)
    angle_warn = thresholds.get("angle_warn_deg", 120.0)
    angle_danger = thresholds.get("angle_danger_deg", 180.0)

    mask_V = mask_V.bool()
    mask_delta = mask_delta.bool()
    delta_deg = _angle_to_deg(delta, mask_delta)

    valid_v = mask_V.float().sum(dim=(1, 2)).clamp_min(1.0)
    low_v_ratio = ((V < v_warn) & mask_V).float().sum(dim=(1, 2)) / valid_v
    danger_v_ratio = ((V < v_danger) & mask_V).float().sum(dim=(1, 2)) / valid_v

    tail_steps = max(1, V.shape[1] // 10)
    tail_low_ratio = ((V[:, -tail_steps:, :] < v_warn) & mask_V[:, -tail_steps:, :]).float().sum(dim=(1, 2))
    tail_den = mask_V[:, -tail_steps:, :].float().sum(dim=(1, 2)).clamp_min(1.0)
    tail_low_ratio = tail_low_ratio / tail_den

    spread_t = _masked_spread(delta_deg, mask_delta)
    angle_peak = spread_t.max(dim=1).values
    angle_tail = spread_t[:, -tail_steps:].mean(dim=1)

    valid_delta_channels = mask_delta.any(dim=1).float().sum(dim=1).clamp_min(1.0)
    angle_speed = torch.diff(delta_deg, dim=1, prepend=delta_deg[:, :1, :])
    angle_speed = torch.abs(angle_speed)
    angle_speed = _masked_mean(angle_speed, mask_delta, dim=(1, 2))

    total_possible = (
        V.shape[1] * mask_V.any(dim=1).float().sum(dim=1)
        + delta.shape[1] * mask_delta.any(dim=1).float().sum(dim=1)
    ).clamp_min(1.0)
    missing_ratio = 1.0 - (mask_V.float().sum(dim=(1, 2)) + mask_delta.float().sum(dim=(1, 2))) / total_possible
    missing_ratio = missing_ratio.clamp(0.0, 1.0)

    angle_peak_norm = ((angle_peak - angle_warn) / max(angle_danger - angle_warn, 1e-3)).clamp(0.0, 1.5)
    angle_tail_norm = ((angle_tail - angle_warn) / max(angle_danger - angle_warn, 1e-3)).clamp(0.0, 1.5)
    speed_norm = (angle_speed / 60.0).clamp(0.0, 1.5)

    # 把低电压、功角扩散、尾部行为和缺失率组合成一个有界辅助风险目标，
    # 供风险回归和物理一致性损失共同使用。
    risk_target = (
        0.32 * danger_v_ratio
        + 0.18 * low_v_ratio
        + 0.16 * tail_low_ratio
        + 0.16 * angle_peak_norm
        + 0.10 * angle_tail_norm
        + 0.04 * speed_norm
        + 0.04 * missing_ratio
    ).clamp(0.0, 1.0)
    phys_label = ((risk_target >= 0.5) | (danger_v_ratio >= 0.12) | (angle_peak >= angle_danger)).long()

    return {
        "risk_target": risk_target.unsqueeze(-1),
        "phys_label": phys_label,
        "low_v_ratio": low_v_ratio.unsqueeze(-1),
        "danger_v_ratio": danger_v_ratio.unsqueeze(-1),
        "tail_low_ratio": tail_low_ratio.unsqueeze(-1),
        "angle_peak": angle_peak.unsqueeze(-1),
        "angle_tail": angle_tail.unsqueeze(-1),
        "angle_speed": angle_speed.unsqueeze(-1),
        "missing_ratio": missing_ratio.unsqueeze(-1),
        "valid_delta_channels": valid_delta_channels.unsqueeze(-1),
    }


def physics_consistency_loss(logits: Tensor, risk_hat: Tensor, phys_feat: dict[str, Tensor]) -> Tensor:
    # 约束分类概率、模型预测风险和物理推导出的风险目标保持一致，
    # 避免三者彼此偏离。
    prob_unstable = logits.softmax(dim=-1)[..., 1]
    risk_target = phys_feat["risk_target"].squeeze(-1)
    phys_label = phys_feat["phys_label"].float()

    prob_loss = F.mse_loss(prob_unstable, risk_target)
    monotonic_loss = F.smooth_l1_loss(risk_hat.squeeze(-1), risk_target, beta=0.1)
    label_loss = F.binary_cross_entropy(prob_unstable.clamp(1e-5, 1 - 1e-5), phys_label)
    return prob_loss + monotonic_loss + 0.5 * label_loss


def total_loss(
    outputs: dict[str, Any],
    y: Tensor,
    phys_feat: dict[str, Tensor],
    cfg: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    # 最终目标把同一个样本的四种学习信号放在一起：标签拟合、
    # 平滑风险回归、物理一致性约束和嵌入空间分离。
    gamma = float(_cfg_get(cfg, "loss.focal_gamma", 2.0))
    unstable_weight = float(_cfg_get(cfg, "loss.class_weight_unstable", 3.0))
    class_weight = torch.tensor([1.0, unstable_weight], device=y.device)
    focal = FocalLoss(gamma=gamma, weight=class_weight)

    # 所有出口都参与监督，这样中间层形成的决策状态也能保持可用。
    cls_losses = [focal(logits, y) for logits in outputs["logits_list"]]
    cls_loss = torch.stack(cls_losses).mean()

    risk_target = phys_feat["risk_target"]
    beta = float(_cfg_get(cfg, "loss.risk_smooth_l1_beta", 0.1))
    risk_losses = [F.smooth_l1_loss(risk_hat, risk_target, beta=beta) for risk_hat in outputs["risk_list"]]
    risk_loss = torch.stack(risk_losses).mean()

    phys_losses = [
        physics_consistency_loss(logits, risk_hat, phys_feat)
        for logits, risk_hat in zip(outputs["logits_list"], outputs["risk_list"])
    ]
    phys_loss = torch.stack(phys_losses).mean()

    con_loss = supervised_contrastive_loss(
        outputs["emb"],
        y,
        risk=risk_target.detach(),
        temperature=float(_cfg_get(cfg, "loss.contrastive_temperature", 0.07)),
        hard_neg=bool(_cfg_get(cfg, "loss.hard_neg", True)),
    )

    total = (
        float(_cfg_get(cfg, "loss.w_cls", 1.0)) * cls_loss
        + float(_cfg_get(cfg, "loss.w_risk", 0.5)) * risk_loss
        + float(_cfg_get(cfg, "loss.lambda_phys", 0.2)) * phys_loss
        + float(_cfg_get(cfg, "loss.lambda_con", 0.1)) * con_loss
    )

    loss_dict = {
        "loss": float(total.detach().cpu()),
        "loss_cls": float(cls_loss.detach().cpu()),
        "loss_risk": float(risk_loss.detach().cpu()),
        "loss_phys": float(phys_loss.detach().cpu()),
        "loss_contrastive": float(con_loss.detach().cpu()),
    }
    return total, loss_dict

