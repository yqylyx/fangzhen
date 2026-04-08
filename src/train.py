from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str((Path.cwd() / '.matplotlib_cache').resolve()))
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ''}:
    from src.data import NPYTransientDataset, TransientNormalizer, collect_npy_files, collate_fn
    from src.eval import evaluate_loader, save_eval_artifacts
    from src.models.losses import physics_risk_features, total_loss
    from src.models.model import PhysHPGT
else:
    from .data import NPYTransientDataset, TransientNormalizer, collect_npy_files, collate_fn
    from .eval import evaluate_loader, save_eval_artifacts
    from .models.losses import physics_risk_features, total_loss
    from .models.model import PhysHPGT


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + '\n')


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg['train'].get('device', 'cuda'))
    if requested.startswith('cuda') and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device('cpu')


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def split_train_val(files: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    # 验证集只从训练域内部切分，外部系统的数据保持完整，
    # 避免 OOD 评估被模型选择过程污染。
    if not files:
        raise ValueError('training dataset contains no npy files')
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    n_val = min(n_val, max(1, len(files) - 1))
    val_files = sorted(files[:n_val])
    train_files = sorted(files[n_val:])
    return train_files, val_files


def dataset_tag(dataset_dir: str | Path) -> str:
    stem = Path(dataset_dir).name
    if stem.endswith('data') and stem[:-4].isdigit():
        return stem[:-4]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return digits or stem


def build_cross_system_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg['data']
    root = Path(data_cfg['root'])
    # 当前默认协议是 36 训练、37 和 74 做 OOD 测试。
    # 以前的 37->36+74 和 74->36+37 设置仍保留在配置注释里，方便对照。
    train_dir = str(data_cfg.get('train_dir', '36data'))
    test_dirs = list(data_cfg.get('test_dirs', ['37data', '74data']))
    val_ratio = float(data_cfg.get('val_ratio', 0.1))
    seed = int(data_cfg.get('split_seed', cfg.get('seed', 42)))

    train_tag = dataset_tag(train_dir)
    # 这里根据配置动态构造协议：一个训练域内验证集，加上多个完整的 OOD 数据集，
    # 不再把具体数据集名字写死在代码里。
    train_files_all = collect_npy_files(root, train_dir)
    train_files, val_files = split_train_val(train_files_all, val_ratio=val_ratio, seed=seed)

    # 旧版这里写死成 train_36 / val_36 / test_37 / test_74。
    # 现在改成完全按配置生成协议，便于切换成 train-on-74 / OOD-on-36+37。
    ood_splits: list[dict[str, Any]] = []
    split_files: dict[str, list[Path]] = {
        'train': train_files,
        'val': val_files,
    }
    for test_dir in test_dirs:
        tag = dataset_tag(test_dir)
        split_key = f'ood_{tag}'
        split_files[split_key] = collect_npy_files(root, test_dir)
        ood_splits.append(
            {
                'split_key': split_key,
                'dataset_tag': tag,
                'dataset_dir': test_dir,
                'split_name': f'{tag}test',
                'banner': f'OOD test on {tag}data',
                'title': f'OOD test on {tag}data',
            }
        )

    return {
        'train_tag': train_tag,
        'train_dir': train_dir,
        'val_ratio': val_ratio,
        'splits': split_files,
        'val_split_name': f'{train_tag}val',
        'val_banner': f'In-domain validation on {train_tag}data',
        'val_title': f'In-domain validation on {train_tag}data',
        'ood_splits': ood_splits,
    }


def save_protocol(protocol: dict[str, Any], out_dir: Path) -> None:
    serializable = {
        'train_tag': protocol['train_tag'],
        'train_dir': protocol['train_dir'],
        'val_ratio': protocol['val_ratio'],
        'val_split_name': protocol['val_split_name'],
        'ood_splits': protocol['ood_splits'],
        'splits': {key: [str(path) for path in paths] for key, paths in protocol['splits'].items()},
    }
    save_json(out_dir / 'splits' / 'cross_system_protocol.json', serializable)


def reset_run_outputs(out_dir: Path, keep_metric_files: set[str] | None = None) -> None:
    keep_metric_files = keep_metric_files or set()
    for subdir_name in ['checkpoints', 'metrics', 'predictions', 'plots', 'normalization', 'splits']:
        subdir = out_dir / subdir_name
        if not subdir.exists():
            continue
        for item in subdir.iterdir():
            if subdir_name == 'metrics' and item.name in keep_metric_files:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)


def build_best_run_record(proto: dict[str, Any], cfg: dict[str, Any], epoch: int | None, score: float | None, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'train_tag': proto['train_tag'],
        'val_split_name': proto['val_split_name'],
        'epoch': epoch,
        'score': score,
        'metrics': metrics,
        'output_dir': str(cfg.get('output_dir', '')),
        'train_cfg': cfg.get('train', {}),
        'model_cfg': cfg.get('model', {}),
        'data_cfg': {
            'train_dir': cfg['data'].get('train_dir'),
            'test_dirs': cfg['data'].get('test_dirs'),
            'angle_preprocess': cfg['data'].get('angle_preprocess'),
        },
    }


def archive_existing_best_run(out_dir: Path, proto: dict[str, Any], cfg: dict[str, Any]) -> None:
    best_metrics_path = out_dir / 'metrics' / f"metrics_{proto['val_split_name']}_best.json"
    history_path = out_dir / 'metrics' / 'history.json'
    history_log_path = out_dir / 'metrics' / 'best_run_history.jsonl'
    if not best_metrics_path.exists():
        return

    metrics = load_json(best_metrics_path)
    best_epoch = None
    best_score = None
    if history_path.exists():
        history = load_json(history_path).get('history', [])
        if history:
            best_entry = max(history, key=lambda item: float(item.get('score', -float('inf'))))
            best_epoch = int(best_entry.get('epoch', -1))
            best_score = float(best_entry.get('score', 0.0))

    record = build_best_run_record(proto, cfg, best_epoch, best_score, metrics, source='pre_reset_snapshot')
    append_jsonl(history_log_path, record)


def append_best_run_history(out_dir: Path, proto: dict[str, Any], cfg: dict[str, Any], history: list[dict[str, Any]]) -> None:
    if not history:
        return
    best_entry = max(history, key=lambda item: float(item.get('score', -float('inf'))))
    metrics = best_entry.get(proto['val_split_name'], {})
    record = build_best_run_record(
        proto,
        cfg,
        int(best_entry.get('epoch', -1)),
        float(best_entry.get('score', 0.0)),
        metrics,
        source='completed_run',
    )
    append_jsonl(out_dir / 'metrics' / 'best_run_history.jsonl', record)


def build_dataloaders(cfg: dict[str, Any], out_dir: Path) -> tuple[dict[str, DataLoader], TransientNormalizer, dict[str, Any]]:
    data_cfg = cfg['data']
    proto = build_cross_system_protocol(cfg)
    save_protocol(proto, out_dir)

    root = Path(data_cfg['root'])
    sentinel = float(data_cfg.get('sentinel', -99999.0))
    angle_preprocess = str(data_cfg.get('angle_preprocess', 'initial'))
    batch_size = int(cfg['train'].get('batch_size', 32))
    eval_batch_size = int(cfg['eval'].get('batch_size', batch_size))
    num_workers = int(data_cfg.get('num_workers', 0))

    # 归一化统计量只在源域训练子集上拟合。
    raw_train_dataset = NPYTransientDataset(
        root=root,
        split='train_raw',
        file_list=proto['splits']['train'],
        sentinel=sentinel,
        angle_preprocess=angle_preprocess,
        normalizer=None,
    )
    norm_cfg = cfg.get('normalization', {})
    normalizer_path = out_dir / 'normalization' / f"scaler_{proto['train_tag']}train.json"
    if bool(norm_cfg.get('reuse_saved', True)) and normalizer_path.exists():
        normalizer = TransientNormalizer.load(normalizer_path)
    else:
        normalizer = TransientNormalizer.fit_from_dataset(
            raw_train_dataset,
            method=str(norm_cfg.get('method', 'robust')),
            eps=float(norm_cfg.get('eps', 1e-6)),
            metadata={'fit_on': f"{proto['train_tag']}data-train", 'angle_preprocess': angle_preprocess},
            max_values_per_modality=int(norm_cfg.get('max_values_per_modality', 500000)),
            sample_seed=int(norm_cfg.get('sample_seed', cfg.get('seed', 42))),
        )
        normalizer.save(normalizer_path)

    # 所有 split 都复用同一个训练集拟合出的归一化器，
    # OOD 域只参与 transform，绝不反向影响预处理统计量。
    datasets: dict[str, NPYTransientDataset] = {
        'train': NPYTransientDataset(root, 'train', proto['splits']['train'], sentinel=sentinel, angle_preprocess=angle_preprocess, normalizer=normalizer),
        'val': NPYTransientDataset(root, 'val', proto['splits']['val'], sentinel=sentinel, angle_preprocess=angle_preprocess, normalizer=normalizer),
    }
    for ood_meta in proto['ood_splits']:
        split_key = ood_meta['split_key']
        datasets[split_key] = NPYTransientDataset(
            root,
            split_key,
            proto['splits'][split_key],
            sentinel=sentinel,
            angle_preprocess=angle_preprocess,
            normalizer=normalizer,
        )

    loaders: dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        shuffle = split_name == 'train'
        cur_batch_size = batch_size if shuffle else eval_batch_size
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cur_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(data_cfg.get('pin_memory', True)),
            persistent_workers=bool(data_cfg.get('persistent_workers', False)) and num_workers > 0,
            collate_fn=collate_fn,
            drop_last=False,
        )
    return loaders, normalizer, proto


def estimate_class_weight(dataset: NPYTransientDataset, default_weight: float = 3.0) -> float:
    # 当失稳样本稀少时，提高失稳类权重；
    # 但不会低于配置里设定的默认强调程度。
    labels = [dataset.peek_label(idx) for idx in range(len(dataset))]
    pos = max(1, sum(int(v == 1) for v in labels))
    neg = max(1, sum(int(v == 0) for v in labels))
    return float(max(default_weight, neg / pos))


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model = PhysHPGT(**cfg['model']).to(device)
    if bool(cfg['train'].get('compile', False)) and hasattr(torch, 'compile'):
        model = torch.compile(model)
    return model


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, cfg: dict[str, Any], metrics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'cfg': cfg,
            'metrics': metrics,
        },
        path,
    )


def maybe_load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, path: str | None, device: torch.device) -> int:
    if not path:
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('epoch', 0)) + 1


def monitor_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    # Early stopping 不是只看单一指标，而是综合 F1、Recall 和 AUC，
    # 这样在类别不平衡时更稳妥。
    weights = cfg['eval'].get('monitor_weights', {})
    auc = float(metrics.get('auc', 0.0))
    auc = 0.0 if np.isnan(auc) else auc
    return (
        float(weights.get('f1', 1.0)) * float(metrics.get('f1', 0.0))
        + float(weights.get('recall', 0.5)) * float(metrics.get('recall', 0.0))
        + float(weights.get('auc', 0.5)) * auc
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, cfg: dict[str, Any], train_desc: str = 'train') -> dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    amp_enabled = bool(cfg['train'].get('amp', True)) and device.type == 'cuda'
    grad_clip = float(cfg['train'].get('grad_clip', 1.0))

    meters = {'loss': 0.0, 'loss_cls': 0.0, 'loss_risk': 0.0, 'loss_phys': 0.0, 'loss_contrastive': 0.0}
    num_batches = 0

    progress = tqdm(loader, desc=train_desc, leave=False, dynamic_ncols=False, ncols=72, mininterval=1.0, file=sys.stdout, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for batch in progress:
        batch = to_device(batch, device)
        # 每个 batch 都会从原始信号重新计算一遍物理软目标，
        # 并和模型输出一起参与组合损失。
        phys_feat = physics_risk_features(
            batch['raw_V'],
            batch['raw_delta'],
            batch['mask_V'],
            batch['mask_delta'],
            thresholds=cfg.get('physics', None),
        )
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
            loss, loss_dict = total_loss(outputs, batch['y'], phys_feat, cfg)

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


def final_evaluate_all(model: nn.Module, loaders: dict[str, DataLoader], cfg: dict[str, Any], out_dir: Path, proto: dict[str, Any]) -> None:
    # 训练结束后，先报告源域验证集结果，
    # 再用同一个最优 checkpoint 统一评估配置里的 OOD 数据集。
    print(proto['val_banner'])
    metrics_val, preds_val = evaluate_loader(model, loaders['val'], cfg, proto['val_split_name'])
    save_eval_artifacts(metrics_val, preds_val, out_dir, proto['val_split_name'], proto['val_title'])

    print(
        f"{proto['val_split_name']}: acc={metrics_val['accuracy']:.4f}, recall={metrics_val['recall']:.4f}, "
        f"f1={metrics_val['f1']:.4f}, auc={metrics_val['auc']:.4f}"
    )

    for ood_meta in proto['ood_splits']:
        print(ood_meta['banner'])
        metrics_ood, preds_ood = evaluate_loader(model, loaders[ood_meta['split_key']], cfg, ood_meta['split_name'])
        save_eval_artifacts(metrics_ood, preds_ood, out_dir, ood_meta['split_name'], ood_meta['title'])
        print(
            f"{ood_meta['split_name']}: acc={metrics_ood['accuracy']:.4f}, recall={metrics_ood['recall']:.4f}, "
            f"f1={metrics_ood['f1']:.4f}, auc={metrics_ood['auc']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Cross-system Phys-HPGT training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device(cfg)
    out_dir = ensure_dir(cfg.get('output_dir', 'runs/phys_hpgt_cross_system'))
    proto_preview = build_cross_system_protocol(cfg)

    resume_path = args.resume or cfg['train'].get('resume')
    # 新开训练会清理 output_dir 里的旧产物；
    # 而 resume 会保留现有结果，并从保存的状态继续训练。
    if not args.eval_only and not resume_path:
        archive_existing_best_run(out_dir, proto_preview, cfg)
        # 每次新开训练会覆盖大部分旧结果，只保留历次最佳摘要文件用于对照参数。
        reset_run_outputs(out_dir, keep_metric_files={'best_run_history.jsonl'})

    ensure_dir(out_dir / 'checkpoints')
    ensure_dir(out_dir / 'metrics')
    ensure_dir(out_dir / 'predictions')
    ensure_dir(out_dir / 'plots')
    ensure_dir(out_dir / 'normalization')
    ensure_dir(out_dir / 'splits')

    loaders, normalizer, proto = build_dataloaders(cfg, out_dir)
    save_json(out_dir / 'normalization' / 'scaler_summary.json', normalizer.to_dict())

    train_dataset = loaders['train'].dataset
    if bool(cfg['loss'].get('auto_class_weight', True)):
        # 根据这次实际切出来的训练子集重新估计类别权重，
        # 而不是假设所有源域数据都共享同一个固定类别比例。
        cfg['loss']['class_weight_unstable'] = estimate_class_weight(
            train_dataset,
            default_weight=float(cfg['loss'].get('class_weight_unstable', 3.0)),
        )

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['train'].get('lr', 3e-4)),
        weight_decay=float(cfg['train'].get('weight_decay', 1e-2)),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['train'].get('amp', True)) and device.type == 'cuda')
    start_epoch = maybe_load_checkpoint(model, optimizer, scaler, resume_path, device)

    best_ckpt = out_dir / 'checkpoints' / f"best_by_{proto['val_split_name']}_f1.pt"
    last_ckpt = out_dir / 'checkpoints' / 'last.pt'
    history: list[dict[str, Any]] = []
    best_score = -float('inf')
    patience = 0

    if args.eval_only:
        final_evaluate_all(model, loaders, cfg, out_dir, proto)
        return

    epochs = int(cfg['train'].get('epochs', 25))
    early_patience = int(cfg['train'].get('early_stopping_patience', 5))

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, loaders['train'], optimizer, scaler, cfg, train_desc=f"train_{proto['train_tag']}")
        # 模型选择始终只依据训练系统内部切出的验证集，
        # 不会使用任何外部 OOD 目标来挑模型。
        print(proto['val_banner'])
        val_metrics, val_predictions = evaluate_loader(model, loaders['val'], cfg, proto['val_split_name'])
        score = monitor_score(val_metrics, cfg)
        history.append({'epoch': epoch, 'train': train_metrics, proto['val_split_name']: val_metrics, 'score': score})
        save_json(out_dir / 'metrics' / 'history.json', {'history': history})
        save_eval_artifacts(val_metrics, val_predictions, out_dir, f"{proto['val_split_name']}_latest", f"{proto['train_tag']}data val epoch {epoch}")
        save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, cfg, val_metrics)

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"{proto['val_split_name']}_f1={val_metrics['f1']:.4f} "
            f"{proto['val_split_name']}_recall={val_metrics['recall']:.4f} "
            f"{proto['val_split_name']}_auc={val_metrics['auc']:.4f}"
        )

        if score > best_score:
            # 保存那个在源域验证集上最平衡 F1、Recall 和 AUC 的 checkpoint，
            # 最终 OOD 评估使用的就是这一份模型。
            best_score = score
            patience = 0
            save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, cfg, val_metrics)
            save_json(out_dir / 'metrics' / f"metrics_{proto['val_split_name']}_best.json", val_metrics)
        else:
            patience += 1
            if patience >= early_patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    append_best_run_history(out_dir, proto, cfg, history)

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])

    final_evaluate_all(model, loaders, cfg, out_dir, proto)


if __name__ == '__main__':
    main()
