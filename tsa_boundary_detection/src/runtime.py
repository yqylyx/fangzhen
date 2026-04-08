from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ''}:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('MPLCONFIGDIR', str((ROOT / '.matplotlib_cache').resolve()))
if os.name == 'nt':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in {None, ''}:
    from src.data import BoundaryLabelIndex, NPYTransientDataset, TransientNormalizer, collect_npy_files, collate_fn
    from src.models.model import PhysHPGT
else:
    from .data import BoundaryLabelIndex, NPYTransientDataset, TransientNormalizer, collect_npy_files, collate_fn
    from .models.model import PhysHPGT

LOGGER = logging.getLogger(__name__)


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    # ???????????????????????????????????
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    anchor = Path(base_dir or ROOT)
    anchored = (anchor / candidate).resolve(strict=False)
    if anchored.exists():
        return anchored
    if candidate.exists():
        return candidate.resolve(strict=False)
    return anchored


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path, ROOT)
    with config_path.open('r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg['_config_path'] = str(config_path)
    return cfg


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


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg.get('train', {}).get('device', 'cuda'))
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


def dataset_tag(dataset_dir: str | Path) -> str:
    stem = Path(dataset_dir).name
    if stem.endswith('data') and stem[:-4].isdigit():
        return stem[:-4]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return digits or stem


def summarize_labels(files: list[Path], label_index: BoundaryLabelIndex) -> dict[str, int]:
    positives = sum(label_index.label_for_path(path) for path in files)
    total = len(files)
    return {'total': total, 'boundary': positives, 'non_boundary': total - positives}


def split_train_val_stratified(
    files: list[Path],
    labels: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    # ?????????????????? train/val?
    # ??????????? boundary ? non-boundary?
    if len(files) != len(labels):
        raise ValueError('files and labels must have the same length')
    if not files:
        raise ValueError('36data contains no npy files')

    rng = random.Random(seed)
    groups: dict[int, list[Path]] = {0: [], 1: []}
    for path, label in zip(files, labels):
        groups[int(label)].append(path)

    train_files: list[Path] = []
    val_files: list[Path] = []
    for group in groups.values():
        rng.shuffle(group)
        if not group:
            continue
        if len(group) == 1:
            train_files.extend(group)
            continue
        n_val = int(round(len(group) * float(val_ratio)))
        n_val = max(1, n_val)
        n_val = min(n_val, len(group) - 1)
        val_files.extend(group[:n_val])
        train_files.extend(group[n_val:])

    if not val_files:
        shuffled = list(files)
        rng.shuffle(shuffled)
        n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(val_ratio)))))
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

    return sorted(train_files), sorted(val_files)


def _build_sampler(train_labels: list[int]) -> WeightedRandomSampler | None:
    # ?????????????????? epoch ????????
    total = len(train_labels)
    if total == 0:
        return None
    pos = max(1, sum(int(v == 1) for v in train_labels))
    neg = max(1, sum(int(v == 0) for v in train_labels))
    weights = [float(total / (2 * pos)) if label == 1 else float(total / (2 * neg)) for label in train_labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _estimate_pos_weight(train_labels: list[int], default_value: float | None = None) -> float:
    pos = sum(int(v == 1) for v in train_labels)
    neg = sum(int(v == 0) for v in train_labels)
    if pos == 0:
        return float(default_value if default_value is not None else 1.0)
    return float(max(default_value or 1.0, neg / max(pos, 1)))


def build_boundary_protocol(cfg: dict[str, Any]) -> tuple[dict[str, Any], BoundaryLabelIndex]:
    # ??????????????36 ??????37/74 ? OOD ???
    data_cfg = cfg['data']
    root = resolve_path(data_cfg['root'], ROOT)
    csv_path = resolve_path(data_cfg['boundary_csv'], ROOT)
    train_dir = str(data_cfg.get('train_dir', '36data'))
    test_dirs = list(data_cfg.get('test_dirs', ['37data', '74data']))
    val_ratio = float(data_cfg.get('val_ratio', 0.1))
    seed = int(data_cfg.get('split_seed', cfg.get('seed', 42)))

    # ????????????????? npy ??? boundary ???
    label_index = BoundaryLabelIndex.from_csv(csv_path, root)
    train_files_all = collect_npy_files(root, train_dir)
    train_labels_all = [label_index.label_for_path(path) for path in train_files_all]
    train_files, val_files = split_train_val_stratified(train_files_all, train_labels_all, val_ratio=val_ratio, seed=seed)

    splits: dict[str, list[Path]] = {'train': train_files, 'val': val_files}
    ood_splits: list[dict[str, Any]] = []
    for test_dir in test_dirs:
        tag = dataset_tag(test_dir)
        split_key = f'ood_{tag}'
        split_name = f'{tag}test'
        test_files = collect_npy_files(root, test_dir)
        splits[split_key] = test_files
        ood_splits.append(
            {
                'split_key': split_key,
                'dataset_tag': tag,
                'dataset_dir': test_dir,
                'split_name': split_name,
                'title': f'OOD boundary test on {tag}data',
            }
        )

    dataset_summaries = {
        dataset_tag(train_dir): label_index.summarize_paths(train_files_all),
    }
    for test_dir in test_dirs:
        dataset_summaries[dataset_tag(test_dir)] = label_index.summarize_paths(collect_npy_files(root, test_dir))
    label_index.log_summary(dataset_summaries)

    return {
        'data_root': str(root),
        'boundary_csv': str(csv_path),
        'train_tag': dataset_tag(train_dir),
        'train_dir': train_dir,
        'test_dirs': test_dirs,
        'val_ratio': val_ratio,
        'splits': splits,
        'val_split_name': f"{dataset_tag(train_dir)}val",
        'ood_splits': ood_splits,
        'dataset_boundary_counts': {
            key: {'total': value.total, 'boundary': value.positives, 'non_boundary': value.negatives}
            for key, value in dataset_summaries.items()
        },
    }, label_index


def save_protocol(out_dir: Path, protocol: dict[str, Any]) -> None:
    payload = {
        'data_root': protocol['data_root'],
        'boundary_csv': protocol['boundary_csv'],
        'train_dir': protocol['train_dir'],
        'test_dirs': protocol['test_dirs'],
        'val_ratio': protocol['val_ratio'],
        'val_split_name': protocol['val_split_name'],
        'ood_splits': protocol['ood_splits'],
        'split_counts': {key: len(value) for key, value in protocol['splits'].items()},
        'dataset_boundary_counts': protocol['dataset_boundary_counts'],
    }
    save_json(out_dir / 'splits' / 'boundary_protocol.json', payload)


def build_dataloaders(
    cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, DataLoader], TransientNormalizer, dict[str, Any], BoundaryLabelIndex, dict[str, Any]]:
    # ???????????????????????? DataLoader ???
    protocol, label_index = build_boundary_protocol(cfg)
    save_protocol(out_dir, protocol)

    data_cfg = cfg['data']
    root = Path(protocol['data_root'])
    sentinel = float(data_cfg.get('sentinel', -99999.0))
    angle_preprocess = str(data_cfg.get('angle_preprocess', 'initial'))
    batch_size = int(cfg.get('train', {}).get('batch_size', 32))
    eval_batch_size = int(cfg.get('eval', {}).get('batch_size', batch_size))
    num_workers = int(data_cfg.get('num_workers', 0))

    # ??????????????????
    raw_train_dataset = NPYTransientDataset(
        root=root,
        split='train_raw',
        file_list=protocol['splits']['train'],
        label_index=label_index,
        sentinel=sentinel,
        angle_preprocess=angle_preprocess,
        normalizer=None,
    )
    norm_cfg = cfg.get('normalization', {})
    normalizer_path = out_dir / 'normalization' / f"scaler_{protocol['train_tag']}train.json"
    if bool(norm_cfg.get('reuse_saved', True)) and normalizer_path.exists():
        normalizer = TransientNormalizer.load(normalizer_path)
    else:
        normalizer = TransientNormalizer.fit_from_dataset(
            raw_train_dataset,
            method=str(norm_cfg.get('method', 'robust')),
            eps=float(norm_cfg.get('eps', 1e-6)),
            metadata={'fit_on': f"{protocol['train_tag']}data-train", 'angle_preprocess': angle_preprocess},
            max_values_per_modality=int(norm_cfg.get('max_values_per_modality', 500000)),
            sample_seed=int(norm_cfg.get('sample_seed', cfg.get('seed', 42))),
        )
        normalizer.save(normalizer_path)

    # ?? split ??????????????????
    # OOD ???? transform??????????????
    datasets: dict[str, NPYTransientDataset] = {
        'train': NPYTransientDataset(root, 'train', protocol['splits']['train'], label_index, sentinel=sentinel, angle_preprocess=angle_preprocess, normalizer=normalizer),
        'val': NPYTransientDataset(root, 'val', protocol['splits']['val'], label_index, sentinel=sentinel, angle_preprocess=angle_preprocess, normalizer=normalizer),
    }
    for ood_meta in protocol['ood_splits']:
        split_key = ood_meta['split_key']
        datasets[split_key] = NPYTransientDataset(
            root,
            split_key,
            protocol['splits'][split_key],
            label_index,
            sentinel=sentinel,
            angle_preprocess=angle_preprocess,
            normalizer=normalizer,
        )

    train_labels = [datasets['train'].peek_label(idx) for idx in range(len(datasets['train']))]
    sampler = None
    if bool(cfg.get('train', {}).get('use_weighted_sampler', False)):
        # ????????????? boundary ???????????
        sampler = _build_sampler(train_labels)

    loaders: dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        is_train = split_name == 'train'
        cur_batch_size = batch_size if is_train else eval_batch_size
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=cur_batch_size,
            shuffle=is_train and sampler is None,
            sampler=sampler if is_train else None,
            num_workers=num_workers,
            pin_memory=bool(data_cfg.get('pin_memory', True)),
            persistent_workers=bool(data_cfg.get('persistent_workers', False)) and num_workers > 0,
            collate_fn=collate_fn,
            drop_last=False,
        )

    # ??????? boundary ????????????????????????
    default_pos_weight = cfg.get('loss', {}).get('pos_weight')
    pos_weight = _estimate_pos_weight(train_labels, float(default_pos_weight) if default_pos_weight is not None else None)
    cfg.setdefault('loss', {})['resolved_pos_weight'] = pos_weight

    label_stats = {
        'csv_boundary_count_total': label_index.csv_boundary_count,
        'train': summarize_labels(protocol['splits']['train'], label_index),
        'val': summarize_labels(protocol['splits']['val'], label_index),
    }
    for ood_meta in protocol['ood_splits']:
        label_stats[ood_meta['split_name']] = summarize_labels(protocol['splits'][ood_meta['split_key']], label_index)
    label_stats['dataset_boundary_counts'] = protocol['dataset_boundary_counts']
    label_stats['loss_pos_weight'] = pos_weight
    save_json(out_dir / 'metrics' / 'label_stats.json', label_stats)
    save_json(out_dir / 'normalization' / 'scaler_summary.json', normalizer.to_dict())
    return loaders, normalizer, protocol, label_index, label_stats


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model = PhysHPGT(**cfg['model']).to(device)
    if bool(cfg.get('train', {}).get('compile', False)) and hasattr(torch, 'compile'):
        model = torch.compile(model)
    return model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    threshold: float,
) -> None:
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
            'threshold': float(threshold),
        },
        path,
    )


def maybe_load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    path: str | None,
    device: torch.device,
) -> int:
    if not path:
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return int(ckpt.get('epoch', 0)) + 1
