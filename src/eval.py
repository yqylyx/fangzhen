from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 兼容直接以脚本方式运行时的相对导入。
ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

# 把 matplotlib 缓存写到项目内，避免无权限目录导致绘图失败。
os.environ.setdefault('MPLCONFIGDIR', str((Path.cwd() / '.matplotlib_cache').resolve()))
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ''}:
    from src.models.losses import physics_risk_features, total_loss
else:
    from .models.losses import physics_risk_features, total_loss

try:
    from torchmetrics.classification import BinaryAUROC
    HAS_TORCHMETRICS = True
except Exception:
    HAS_TORCHMETRICS = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # 统一把 batch 里的张量搬到目标设备，其它元信息保持不变。
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def binary_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # 纯 numpy 版本的 AUC 兜底实现，避免没有 torchmetrics 时无法评估。
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, device: torch.device | None = None) -> dict[str, Any]:
    # 统一从不稳定概率和阈值导出分类指标，方便训练后和独立评估脚本复用。
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    # 同时计算稳定类的 F1，再求 macro F1，避免只盯住不稳定类。
    precision_stable = tn / max(1, tn + fn)
    recall_stable = tn / max(1, tn + fp)
    f1_stable = 2 * precision_stable * recall_stable / max(1e-8, precision_stable + recall_stable)

    macro_f1 = 0.5 * (f1 + f1_stable)
    micro_f1 = accuracy
    auc = binary_auc_np(y_true, y_prob)
    if HAS_TORCHMETRICS and device is not None:
        try:
            metric = BinaryAUROC().to(device)
            auc = float(metric(torch.as_tensor(y_prob, dtype=torch.float32, device=device), torch.as_tensor(y_true, dtype=torch.long, device=device)).detach().cpu())
        except Exception:
            pass

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'auc': float(auc),
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def save_confusion_matrix_png(cm: np.ndarray, save_path: str | Path, title: str) -> None:
    # 把混淆矩阵单独存成图片，便于快速查看不同数据域上的误判结构。
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Pred Stable', 'Pred Unstable'])
    ax.set_yticks([0, 1], labels=['True Stable', 'True Unstable'])
    ax.set_title(title)
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(int(value)), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, cfg: dict[str, Any], split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # 在单个数据划分上完整跑一遍前向、损失统计和逐样本预测导出。
    model.eval()
    device = next(model.parameters()).device
    amp_enabled = bool(cfg['train'].get('amp', True)) and device.type == 'cuda'
    threshold = float(cfg['eval'].get('threshold', 0.5))
    max_latency_batches = int(cfg['eval'].get('max_latency_batches', 20))

    losses = {'loss': 0.0, 'loss_cls': 0.0, 'loss_risk': 0.0, 'loss_phys': 0.0, 'loss_contrastive': 0.0}
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    latency_ms: list[float] = []
    predictions: list[dict[str, Any]] = []
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        # 评估阶段仍然重建物理风险特征，这样总损失口径与训练保持一致。
        phys_feat = physics_risk_features(
            batch['raw_V'],
            batch['raw_delta'],
            batch['mask_V'],
            batch['mask_delta'],
            thresholds=cfg.get('physics', None),
        )
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        start = time.perf_counter()
        with autocast_ctx:
            outputs = model(
                batch['V'],
                batch['delta'],
                batch['mask_V'],
                batch['mask_delta'],
                batch['ch_mask_V'],
                batch['ch_mask_delta'],
                batch['time_mask'],
            )
            loss, loss_dict = total_loss(outputs, batch['y'], phys_feat, cfg)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if batch_idx < max_latency_batches:
            latency_ms.append(elapsed / max(1, batch['y'].shape[0]))

        for key in losses:
            losses[key] += float(loss_dict[key])
        num_batches += 1

        # 默认使用最后一个出口的分类结果做离线评估。
        logits = outputs['logits_list'][-1]
        probs = torch.softmax(logits, dim=-1)
        unstable_prob = probs[:, 1]
        pred = (unstable_prob >= threshold).long()
        risk = outputs['risk_list'][-1].squeeze(-1)

        y_true_all.append(batch['y'].detach().cpu().numpy())
        y_prob_all.append(unstable_prob.detach().cpu().numpy())

        for idx in range(batch['y'].shape[0]):
            predictions.append({
                'split_name': split_name,
                'dataset_name': batch['dataset_names'][idx],
                'file': batch['files'][idx],
                'sample_id': batch['sample_ids'][idx],
                'label': int(batch['y'][idx].detach().cpu()),
                'pred': int(pred[idx].detach().cpu()),
                'prob_stable': float(probs[idx, 0].detach().cpu()),
                'prob_unstable': float(probs[idx, 1].detach().cpu()),
                'risk': float(risk[idx].detach().cpu()),
            })

    for key in losses:
        losses[key] /= max(1, num_batches)

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_prob_np = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float32)
    metrics = compute_metrics(y_true_np, y_prob_np, threshold, device=device)
    metrics.update(losses)
    metrics['avg_latency_ms'] = float(np.mean(latency_ms)) if latency_ms else float('nan')
    metrics['split_name'] = split_name
    metrics['num_samples'] = int(y_true_np.shape[0])
    return metrics, predictions


def save_predictions_csv(predictions: list[dict[str, Any]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not predictions:
        with save_path.open('w', encoding='utf-8-sig', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['split_name', 'dataset_name', 'file', 'sample_id', 'label', 'pred', 'prob_stable', 'prob_unstable', 'risk'])
        return
    fieldnames = list(predictions[0].keys())
    with save_path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def save_eval_artifacts(metrics: dict[str, Any], predictions: list[dict[str, Any]], output_dir: str | Path, stem: str, title: str) -> None:
    # 把指标、逐样本预测和混淆矩阵集中保存，便于后续和训练脚本产物对齐。
    output_dir = Path(output_dir)
    metrics_dir = ensure_dir(output_dir / 'metrics')
    pred_dir = ensure_dir(output_dir / 'predictions')
    plot_dir = ensure_dir(output_dir / 'plots')

    save_json(metrics_dir / f'metrics_{stem}.json', metrics)
    save_predictions_csv(predictions, pred_dir / f'predictions_{stem}.csv')
    save_confusion_matrix_png(np.asarray(metrics['confusion_matrix']), plot_dir / f'confusion_{stem}.png', title)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Phys-HPGT under configurable cross-system protocol')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--splits', nargs='*', default=None)
    args = parser.parse_args()

    if __package__ in {None, ''}:
        from src.train import build_dataloaders, build_model, get_device, load_config, seed_everything
    else:
        from .train import build_dataloaders, build_model, get_device, load_config, seed_everything

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(cfg)
    out_dir = Path(args.output_dir or cfg.get('output_dir', 'runs/phys_hpgt_cross_system'))
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'checkpoints')
    ensure_dir(out_dir / 'metrics')
    ensure_dir(out_dir / 'predictions')
    ensure_dir(out_dir / 'plots')
    ensure_dir(out_dir / 'normalization')
    ensure_dir(out_dir / 'splits')

    # 评估脚本直接复用训练侧的数据划分和归一化配置，保证协议一致。
    loaders, _, proto = build_dataloaders(cfg, out_dir)
    model = build_model(cfg, device)

    checkpoint = args.checkpoint or str(out_dir / 'checkpoints' / f"best_by_{proto['val_split_name']}_f1.pt")
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    split_registry = {
        proto['val_split_name']: {
            'loader_key': 'val',
            'stem': proto['val_split_name'],
            'banner': proto['val_banner'],
            'title': proto['val_title'],
        }
    }
    for ood_meta in proto['ood_splits']:
        split_registry[ood_meta['split_name']] = {
            'loader_key': ood_meta['split_key'],
            'stem': ood_meta['split_name'],
            'banner': ood_meta['banner'],
            'title': ood_meta['title'],
        }

    selected_splits = args.splits or list(split_registry.keys())
    for split_name in selected_splits:
        if split_name not in split_registry:
            raise ValueError(f'unknown split: {split_name}')
        meta = split_registry[split_name]
        print(meta['banner'])
        metrics, predictions = evaluate_loader(model, loaders[meta['loader_key']], cfg, meta['stem'])
        save_eval_artifacts(metrics, predictions, out_dir, meta['stem'], meta['title'])
        print(
            f"{meta['stem']}: acc={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}"
        )


if __name__ == '__main__':
    main()
