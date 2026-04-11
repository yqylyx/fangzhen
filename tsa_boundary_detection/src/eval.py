from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str((ROOT / '.matplotlib_cache').resolve()))
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ in {None, ''}:
    from src.metrics import build_threshold_grid, compute_binary_metrics, select_best_threshold
    from src.models.losses import total_loss
    from src.runtime import build_dataloaders, build_model, ensure_dir, get_device, load_config, save_json, seed_everything, to_device
    from src.utils.vis import save_waveforms
else:
    from .metrics import build_threshold_grid, compute_binary_metrics, select_best_threshold
    from .models.losses import total_loss
    from .runtime import build_dataloaders, build_model, ensure_dir, get_device, load_config, save_json, seed_everything, to_device
    from .utils.vis import save_waveforms


def _safe_stem(text: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]+', '_', text)


def _prediction_row(batch: dict[str, Any], idx: int, p_boundary: float) -> dict[str, Any]:
    # ??????????????????????????????????
    return {
        'file': batch['files'][idx],
        'dataset_name': batch['dataset_names'][idx],
        'sample_id': batch['sample_ids'][idx],
        'true_boundary_label': int(batch['boundary_label'][idx].detach().cpu().item()),
        'p_boundary': float(p_boundary),
    }


def apply_threshold_to_prediction_rows(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    # ??????? boundary / non-boundary ??????????????????????
    finalized: list[dict[str, Any]] = []
    for row in rows:
        pred = int(float(row['p_boundary']) >= float(threshold))
        finalized.append(
            {
                **row,
                'pred_boundary_label': pred,
                'threshold_used': float(threshold),
                'is_correct': int(pred == int(row['true_boundary_label'])),
            }
        )
    return finalized


def save_confusion_matrix_png(cm: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Pred Non-Boundary', 'Pred Boundary'])
    ax.set_yticks([0, 1], labels=['True Non-Boundary', 'True Boundary'])
    ax.set_title(title)
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(int(value)), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_predictions_csv(predictions: list[dict[str, Any]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'file',
        'dataset_name',
        'sample_id',
        'true_boundary_label',
        'pred_boundary_label',
        'p_boundary',
        'threshold_used',
        'is_correct',
    ]
    with save_path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def save_count_summary_csv(metrics: dict[str, Any], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'split_name',
        'threshold',
        'real_boundary_count',
        'pred_boundary_count',
        'tp',
        'fp',
        'fn',
        'tn',
        'count_error',
    ]
    row = {key: metrics.get(key) for key in fieldnames}
    with save_path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _maybe_save_visuals(
    batch: dict[str, Any],
    # ?????????? tp/fp/fn ???????
    # ??????????????????
    idx: int,
    split_name: str,
    threshold: float,
    p_boundary: float,
    output_root: Path,
    counters: dict[str, int],
    cfg: dict[str, Any],
) -> None:
    vis_cfg = cfg.get('visualization', {})
    if not bool(vis_cfg.get('enabled', False)):
        return
    allowed_splits = set(vis_cfg.get('splits', []))
    if allowed_splits and split_name not in allowed_splits:
        return

    true_label = int(batch['boundary_label'][idx].detach().cpu().item())
    pred_label = int(p_boundary >= threshold)
    title = (
        f"{batch['dataset_names'][idx]} | {Path(batch['files'][idx]).name} | "
        f"true={true_label} pred={pred_label} p={p_boundary:.4f}"
    )

    bucket_to_limit = {
        'predicted_boundary': int(vis_cfg.get('max_predicted_boundary', 80)),
        'tp': int(vis_cfg.get('max_tp', 80)),
        'fp': int(vis_cfg.get('max_fp', 80)),
        'fn': int(vis_cfg.get('max_fn', 80)),
    }
    active_buckets: list[str] = []
    if pred_label == 1:
        active_buckets.append('predicted_boundary')
    if pred_label == 1 and true_label == 1:
        active_buckets.append('tp')
    if pred_label == 1 and true_label == 0:
        active_buckets.append('fp')
    if pred_label == 0 and true_label == 1:
        active_buckets.append('fn')

    t_len = int(batch['time_mask'][idx].detach().cpu().sum().item())
    nv = int(batch['ch_mask_V'][idx].detach().cpu().sum().item())
    ng = int(batch['ch_mask_delta'][idx].detach().cpu().sum().item())
    V = batch['raw_V'][idx, :t_len, :nv].detach().cpu()
    delta = batch['raw_delta'][idx, :t_len, :ng].detach().cpu()
    times = batch['t'][idx, :t_len].detach().cpu()
    file_stem = _safe_stem(f"{batch['dataset_names'][idx]}_{batch['sample_ids'][idx]}_{p_boundary:.4f}")

    for bucket in active_buckets:
        if counters.get(bucket, 0) >= bucket_to_limit[bucket]:
            continue
        counters[bucket] = counters.get(bucket, 0) + 1
        save_waveforms(
            V,
            delta,
            output_root / split_name / bucket / f'{file_stem}.png',
            title=title,
            time_values=times,
            max_channels=int(vis_cfg.get('max_channels', 16)),
            downsample=int(vis_cfg.get('downsample', 2)),
        )


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    # ????????????????????????????
    loader,
    cfg: dict[str, Any],
    split_name: str,
    threshold: float,
    save_visuals: bool = False,
    visual_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    device = next(model.parameters()).device
    amp_enabled = bool(cfg.get('train', {}).get('amp', True)) and device.type == 'cuda'

    losses = {'loss': 0.0, 'loss_boundary': 0.0}
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []
    latency_ms: list[float] = []
    base_predictions: list[dict[str, Any]] = []
    num_batches = 0
    counters: dict[str, int] = {}
    visual_path = Path(visual_root) if visual_root is not None else None

    for batch_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
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
            loss, loss_dict = total_loss(outputs, batch['boundary_label'], cfg)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        latency_ms.append(elapsed / max(1, batch['boundary_label'].shape[0]))

        for key in losses:
            losses[key] += float(loss_dict[key])
        num_batches += 1

        # ???????????? exit ? logits ?????????
        logits = outputs['logits_list'][-1]
        probs = torch.sigmoid(logits)
        y_true_all.append(batch['boundary_label'].detach().cpu().numpy().astype(np.int64))
        y_prob_all.append(probs.detach().cpu().numpy().astype(np.float64))

        for idx in range(batch['boundary_label'].shape[0]):
            p_boundary = float(probs[idx].detach().cpu().item())
            base_predictions.append(_prediction_row(batch, idx, p_boundary))
            if save_visuals and visual_path is not None:
                _maybe_save_visuals(batch, idx, split_name, threshold, p_boundary, visual_path, counters, cfg)

    for key in losses:
        losses[key] /= max(1, num_batches)

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_prob_np = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float64)
    predictions = apply_threshold_to_prediction_rows(base_predictions, threshold)
    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold)
    metrics.update(losses)
    metrics['avg_latency_ms'] = float(np.mean(latency_ms)) if latency_ms else float('nan')
    metrics['split_name'] = split_name
    metrics['num_samples'] = int(y_true_np.shape[0])
    return metrics, predictions, {'y_true': y_true_np, 'y_prob': y_prob_np}


def save_eval_artifacts(metrics: dict[str, Any], predictions: list[dict[str, Any]], output_dir: str | Path, stem: str, title: str) -> None:
    # ?? split ????????????????????????????
    output_dir = Path(output_dir)
    metrics_dir = ensure_dir(output_dir / 'metrics')
    pred_dir = ensure_dir(output_dir / 'predictions')
    plot_dir = ensure_dir(output_dir / 'plots')
    count_dir = ensure_dir(output_dir / 'counts')

    save_json(metrics_dir / f'metrics_{stem}.json', metrics)
    save_predictions_csv(predictions, pred_dir / f'predictions_{stem}.csv')
    pred_boundary_only = [row for row in predictions if int(row['pred_boundary_label']) == 1]
    save_predictions_csv(pred_boundary_only, pred_dir / f'predicted_boundary_only_{stem}.csv')
    save_count_summary_csv(metrics, count_dir / f'count_summary_{stem}.csv')
    save_confusion_matrix_png(np.asarray(metrics['confusion_matrix']), plot_dir / f'confusion_{stem}.png', title)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate the boundary detection Phys-HPGT variant')
    parser.add_argument('--config', type=str, default=str(ROOT / 'configs' / 'config.yaml'))
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--splits', nargs='*', default=None)
    parser.add_argument('--threshold-objective', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(cfg)
    out_dir = Path(args.output_dir or cfg.get('output_dir', 'runs/boundary_detection'))
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'metrics')
    ensure_dir(out_dir / 'predictions')
    ensure_dir(out_dir / 'plots')
    ensure_dir(out_dir / 'counts')
    ensure_dir(out_dir / 'normalization')
    ensure_dir(out_dir / 'splits')

    loaders, _, proto, _, _ = build_dataloaders(cfg, out_dir)
    model = build_model(cfg, device)

    checkpoint = args.checkpoint or str(out_dir / 'checkpoints' / f"best_by_{proto['val_split_name']}_f1.pt")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    threshold_cfg = cfg.get('threshold', {})
    objective = str(args.threshold_objective or threshold_cfg.get('objective', 'f1'))
    # ????? 36-val ????????????? OOD ??????????
    val_metrics_05, val_predictions_05, val_raw = evaluate_loader(model, loaders['val'], cfg, proto['val_split_name'], threshold=0.5, save_visuals=False)
    selection = select_best_threshold(
        val_raw['y_true'],
        val_raw['y_prob'],
        objective=objective,
        threshold_grid=build_threshold_grid(
            float(threshold_cfg.get('min', 0.05)),
            float(threshold_cfg.get('max', 0.95)),
            int(threshold_cfg.get('num_steps', 181)),
        ),
    )
    threshold = float(selection.threshold)
    val_metrics = {**selection.metrics, 'loss': val_metrics_05['loss'], 'loss_boundary': val_metrics_05['loss_boundary'], 'avg_latency_ms': val_metrics_05['avg_latency_ms'], 'split_name': proto['val_split_name'], 'num_samples': val_metrics_05['num_samples']}
    val_predictions = apply_threshold_to_prediction_rows(val_predictions_05, threshold)
    save_eval_artifacts(val_metrics, val_predictions, out_dir, proto['val_split_name'], f'Boundary validation on {proto["train_tag"]}data')

    split_registry = {
        proto['val_split_name']: {'loader_key': 'val', 'stem': proto['val_split_name'], 'title': f'Boundary validation on {proto["train_tag"]}data'},
    }
    for ood_meta in proto['ood_splits']:
        split_registry[ood_meta['split_name']] = {'loader_key': ood_meta['split_key'], 'stem': ood_meta['split_name'], 'title': ood_meta['title']}

    selected_splits = args.splits or list(split_registry.keys())
    for split_name in selected_splits:
        meta = split_registry[split_name]
        if split_name == proto['val_split_name']:
            print(json.dumps({'split': split_name, **val_metrics}, ensure_ascii=False, indent=2))
            continue
        metrics, predictions, _ = evaluate_loader(
            model,
            loaders[meta['loader_key']],
            cfg,
            meta['stem'],
            threshold=threshold,
            save_visuals=bool(cfg.get('visualization', {}).get('enabled', False)),
            visual_root=out_dir / 'plots' / 'samples',
        )
        save_eval_artifacts(metrics, predictions, out_dir, meta['stem'], meta['title'])
        print(json.dumps({'split': split_name, **metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
