from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str((ROOT / '.matplotlib_cache').resolve()))
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ''}:
    from src.eval import apply_threshold_to_prediction_rows, evaluate_loader, save_eval_artifacts
    from src.metrics import build_threshold_grid, select_best_threshold
    from src.models.losses import total_loss
    from src.runtime import (
        ROOT as PROJECT_ROOT,
        build_dataloaders,
        build_model,
        ensure_dir,
        get_device,
        load_config,
        maybe_load_checkpoint,
        resolve_path,
        save_checkpoint,
        save_json,
        seed_everything,
        to_device,
    )
else:
    from .eval import apply_threshold_to_prediction_rows, evaluate_loader, save_eval_artifacts
    from .metrics import build_threshold_grid, select_best_threshold
    from .models.losses import total_loss
    from .runtime import (
        ROOT as PROJECT_ROOT,
        build_dataloaders,
        build_model,
        ensure_dir,
        get_device,
        load_config,
        maybe_load_checkpoint,
        resolve_path,
        save_checkpoint,
        save_json,
        seed_everything,
        to_device,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: dict[str, Any],
    train_desc: str = 'train',
) -> dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    amp_enabled = bool(cfg.get('train', {}).get('amp', True)) and device.type == 'cuda'
    grad_clip = float(cfg.get('train', {}).get('grad_clip', 1.0))

    meters = {'loss': 0.0, 'loss_boundary': 0.0}
    num_batches = 0
    progress = tqdm(loader, desc=train_desc, leave=False, dynamic_ncols=False, ncols=72, mininterval=1.0, file=sys.stdout, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for batch in progress:
        # ??????????? boundary ??????????????? risk ?????
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
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

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for key in meters:
            meters[key] += float(loss_dict[key])
        num_batches += 1

    for key in meters:
        meters[key] /= max(1, num_batches)
    meters['lr'] = float(optimizer.param_groups[0]['lr'])
    return meters


def main() -> None:
    parser = argparse.ArgumentParser(description='Boundary detection training on top of the Phys-HPGT framework')
    parser.add_argument('--config', type=str, default=str(ROOT / 'configs' / 'config.yaml'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(cfg)
    out_dir = ensure_dir(resolve_path(cfg.get('output_dir', 'runs/boundary_detection'), PROJECT_ROOT))
    ensure_dir(out_dir / 'checkpoints')
    ensure_dir(out_dir / 'metrics')
    ensure_dir(out_dir / 'predictions')
    ensure_dir(out_dir / 'plots')
    ensure_dir(out_dir / 'plots' / 'samples')
    ensure_dir(out_dir / 'counts')
    ensure_dir(out_dir / 'normalization')
    ensure_dir(out_dir / 'splits')

    # ??????????????????? split ? DataLoader?
    loaders, _, proto, _, label_stats = build_dataloaders(cfg, out_dir)
    print(json.dumps({'csv_boundary_count_total': label_stats['csv_boundary_count_total'], 'dataset_boundary_counts': label_stats['dataset_boundary_counts']}, ensure_ascii=False, indent=2))
    print(json.dumps({'train_label_stats': label_stats['train'], 'val_label_stats': label_stats['val'], 'loss_pos_weight': label_stats['loss_pos_weight']}, ensure_ascii=False, indent=2))

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get('train', {}).get('lr', 3e-4)),
        weight_decay=float(cfg.get('train', {}).get('weight_decay', 1e-2)),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('train', {}).get('amp', True)) and device.type == 'cuda')
    start_epoch = maybe_load_checkpoint(model, optimizer, scaler, args.resume or cfg.get('train', {}).get('resume'), device)

    best_ckpt = out_dir / 'checkpoints' / f"best_by_{proto['val_split_name']}_f1.pt"
    last_ckpt = out_dir / 'checkpoints' / 'last.pt'
    history: list[dict[str, Any]] = []
    best_score = -float('inf')
    patience = 0

    # ???????????? 0.5????? 36-val ????????
    threshold_cfg = cfg.get('threshold', {})
    objective = str(threshold_cfg.get('objective', 'f1'))
    threshold_grid = build_threshold_grid(
        float(threshold_cfg.get('min', 0.05)),
        float(threshold_cfg.get('max', 0.95)),
        int(threshold_cfg.get('num_steps', 181)),
    )

    if args.eval_only:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        epochs = int(cfg.get('train', {}).get('epochs', 25))
        early_patience = int(cfg.get('train', {}).get('early_stopping_patience', 5))
        for epoch in range(start_epoch, epochs):
            # ?? epoch ??????? 36-val ????????????????
            train_metrics = train_one_epoch(model, loaders['train'], optimizer, scaler, cfg, train_desc=f"train_boundary_{proto['train_tag']}")
            val_metrics_05, val_predictions_05, val_raw = evaluate_loader(model, loaders['val'], cfg, proto['val_split_name'], threshold=0.5, save_visuals=False)
            selection = select_best_threshold(val_raw['y_true'], val_raw['y_prob'], objective=objective, threshold_grid=threshold_grid)
            best_threshold = float(selection.threshold)
            val_metrics = {
                **selection.metrics,
                'loss': val_metrics_05['loss'],
                'loss_boundary': val_metrics_05['loss_boundary'],
                'avg_latency_ms': val_metrics_05['avg_latency_ms'],
                'split_name': proto['val_split_name'],
                'num_samples': val_metrics_05['num_samples'],
            }
            val_predictions = apply_threshold_to_prediction_rows(val_predictions_05, best_threshold)
            score = float(val_metrics['f1'])
            history.append({'epoch': epoch, 'train': train_metrics, 'threshold': best_threshold, proto['val_split_name']: val_metrics})
            save_json(out_dir / 'metrics' / 'history.json', {'history': history})
            save_eval_artifacts(val_metrics, val_predictions, out_dir, f"{proto['val_split_name']}_latest", f"Boundary validation on {proto['train_tag']}data epoch {epoch}")
            save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, cfg, val_metrics, best_threshold)
            save_json(
                out_dir / 'metrics' / 'threshold_selection_latest.json',
                {
                    'epoch': epoch,
                    'objective': objective,
                    'best_threshold': best_threshold,
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                },
            )

            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"best_threshold={best_threshold:.4f} "
                f"val_precision={val_metrics['precision']:.4f} "
                f"val_recall={val_metrics['recall']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if score > best_score:
                # best checkpoint ???? 36-val ???? F1?
                # ??????????????
                best_score = score
                patience = 0
                save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, cfg, val_metrics, best_threshold)
                save_json(out_dir / 'metrics' / f"metrics_{proto['val_split_name']}_best.json", val_metrics)
                save_json(
                    out_dir / 'metrics' / 'threshold_selection_best.json',
                    {
                        'epoch': epoch,
                        'objective': objective,
                        'best_threshold': best_threshold,
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1'],
                    },
                )
            else:
                patience += 1
                if patience >= early_patience:
                    print(f'Early stopping triggered at epoch {epoch}.')
                    break

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])

    val_metrics_05, val_predictions_05, val_raw = evaluate_loader(model, loaders['val'], cfg, proto['val_split_name'], threshold=0.5, save_visuals=False)
    final_selection = select_best_threshold(val_raw['y_true'], val_raw['y_prob'], objective=objective, threshold_grid=threshold_grid)
    final_threshold = float(final_selection.threshold)
    val_metrics = {
        **final_selection.metrics,
        'loss': val_metrics_05['loss'],
        'loss_boundary': val_metrics_05['loss_boundary'],
        'avg_latency_ms': val_metrics_05['avg_latency_ms'],
        'split_name': proto['val_split_name'],
        'num_samples': val_metrics_05['num_samples'],
    }
    val_predictions = apply_threshold_to_prediction_rows(val_predictions_05, final_threshold)
    save_json(
        out_dir / 'metrics' / 'threshold_selection_final.json',
        {
            'objective': objective,
            'best_threshold': final_threshold,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
        },
    )
    save_eval_artifacts(val_metrics, val_predictions, out_dir, proto['val_split_name'], f"Boundary validation on {proto['train_tag']}data")
    print(json.dumps({'split': proto['val_split_name'], **val_metrics}, ensure_ascii=False, indent=2))

    for ood_meta in proto['ood_splits']:
        metrics, predictions, _ = evaluate_loader(
            model,
            loaders[ood_meta['split_key']],
            cfg,
            ood_meta['split_name'],
            threshold=final_threshold,
            save_visuals=bool(cfg.get('visualization', {}).get('enabled', False)),
            visual_root=out_dir / 'plots' / 'samples',
        )
        save_eval_artifacts(metrics, predictions, out_dir, ood_meta['split_name'], ood_meta['title'])
        print(json.dumps({'split': ood_meta['split_name'], **metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
