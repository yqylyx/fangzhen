from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _normalize_dataset_tag(value: Any) -> str | None:
    # ??? 36data / 36 / DATA36 ?????????????????
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.endswith('data'):
        lowered = lowered[:-4]
    digits = ''.join(ch for ch in lowered if ch.isdigit())
    if digits:
        return digits
    return lowered or None


def _normalize_file_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).name.lower()


def _normalize_rel_path(value: Any, data_root: Path) -> str | None:
    # ??? CSV ????????? data_root ??????
    # ??????????????????
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve(strict=False).relative_to(data_root.resolve(strict=False))
        except ValueError:
            return None
    parts = [part for part in candidate.parts if part not in {'.', ''}]
    if not parts:
        return None
    return Path(*parts).as_posix().lower()


@dataclass(frozen=True)
class LabelSummary:
    total: int
    positives: int
    negatives: int


class BoundaryLabelIndex:
    def __init__(
        self,
        *,
        csv_path: Path,
        data_root: Path,
        exact_positive_keys: set[str],
        joint_positive_keys: set[tuple[str, str]],
        file_positive_keys: set[str],
        csv_total_rows: int,
        duplicate_rows: int,
        unmatched_rows: list[str],
        strategy_counts: dict[str, int],
        file_column: str | None,
        dataset_column: str | None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.exact_positive_keys = set(exact_positive_keys)
        self.joint_positive_keys = set(joint_positive_keys)
        self.file_positive_keys = set(file_positive_keys)
        self.csv_total_rows = int(csv_total_rows)
        self.duplicate_rows = int(duplicate_rows)
        self.unmatched_rows = list(unmatched_rows)
        self.strategy_counts = dict(strategy_counts)
        self.file_column = file_column
        self.dataset_column = dataset_column

    @property
    def csv_boundary_count(self) -> int:
        return len(self.exact_positive_keys) + len(self.joint_positive_keys) + len(self.file_positive_keys)

    @classmethod
    def from_csv(cls, csv_path: str | Path, data_root: str | Path) -> 'BoundaryLabelIndex':
        # ?????????? CSV ?????
        # ????????????????????
        csv_path = Path(csv_path).resolve(strict=False)
        data_root = Path(data_root).resolve(strict=False)
        if not csv_path.exists():
            raise FileNotFoundError(f'boundary csv not found: {csv_path}')

        payload = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        if payload.size == 0:
            raise ValueError(f'boundary csv has no rows: {csv_path}')
        rows = payload if payload.ndim > 0 else np.asarray([payload], dtype=payload.dtype)
        fieldnames = list(rows.dtype.names or [])
        if not fieldnames:
            raise ValueError(f'boundary csv does not contain a header row: {csv_path}')

        actual_files = sorted(data_root.rglob('*.npy'))
        exact_counts: dict[str, int] = {}
        joint_counts: dict[tuple[str, str], int] = {}
        file_counts: dict[str, int] = {}
        for path in actual_files:
            rel_key = path.resolve(strict=False).relative_to(data_root).as_posix().lower()
            dataset_dir = next((part for part in path.parts if part.lower().endswith('data')), path.parent.name)
            dataset_tag = _normalize_dataset_tag(dataset_dir)
            file_name = path.name.lower()
            exact_counts[rel_key] = exact_counts.get(rel_key, 0) + 1
            if dataset_tag is not None:
                joint_key = (dataset_tag, file_name)
                joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
            file_counts[file_name] = file_counts.get(file_name, 0) + 1

        file_col_candidates = ('sample_name', 'file_name', 'file', 'source_abs_path', 'path')
        dataset_col_candidates = ('dataset_name', 'dataset_dir_name', 'dataset')
        file_column = next((name for name in file_col_candidates if name in fieldnames), None)
        dataset_column = next((name for name in dataset_col_candidates if name in fieldnames), None)
        if file_column is None:
            raise ValueError(
                f'boundary csv must contain one of {file_col_candidates}, found columns={fieldnames}'
            )

        # ??????????????????dataset+file_name?? file_name?
        exact_positive_keys: set[str] = set()
        joint_positive_keys: set[tuple[str, str]] = set()
        file_positive_keys: set[str] = set()
        duplicate_rows = 0
        unmatched_rows: list[str] = []
        strategy_counts = {'exact': 0, 'joint': 0, 'file_only': 0}

        for row in rows:
            row_dict = {name: row[name] for name in fieldnames}
            rel_key = None
            for path_col in ('file', 'source_abs_path', 'path'):
                if path_col in fieldnames:
                    rel_key = _normalize_rel_path(row_dict.get(path_col), data_root)
                    if rel_key is not None:
                        break

            file_name = _normalize_file_name(row_dict.get(file_column))
            dataset_tag = _normalize_dataset_tag(row_dict.get(dataset_column)) if dataset_column else None
            if dataset_tag is None and rel_key is not None:
                dataset_tag = _normalize_dataset_tag(Path(rel_key).parts[0])

            if rel_key is not None:
                if rel_key not in exact_counts:
                    unmatched_rows.append(rel_key)
                    continue
                if rel_key in exact_positive_keys:
                    duplicate_rows += 1
                exact_positive_keys.add(rel_key)
                strategy_counts['exact'] += 1
                continue

            if dataset_tag is not None and file_name is not None:
                joint_key = (dataset_tag, file_name)
                if joint_counts.get(joint_key, 0) > 1:
                    raise ValueError(
                        'boundary csv only provides dataset_name + file_name for an ambiguous sample. '
                        f'Please include a relative path column like `file`. Ambiguous key={joint_key!r}'
                    )
                if joint_counts.get(joint_key, 0) == 0:
                    unmatched_rows.append(f'{dataset_tag}/{file_name}')
                    continue
                if joint_key in joint_positive_keys:
                    duplicate_rows += 1
                joint_positive_keys.add(joint_key)
                strategy_counts['joint'] += 1
                continue

            if file_name is not None:
                if file_counts.get(file_name, 0) > 1:
                    raise ValueError(
                        'boundary csv does not contain a dataset column, but the file name is not globally unique. '
                        f'Please provide dataset_name + file_name or a full relative path. file_name={file_name!r}'
                    )
                if file_counts.get(file_name, 0) == 0:
                    unmatched_rows.append(file_name)
                    continue
                if file_name in file_positive_keys:
                    duplicate_rows += 1
                file_positive_keys.add(file_name)
                strategy_counts['file_only'] += 1
                continue

            raise ValueError(f'could not resolve a usable boundary key from csv row: {row_dict}')

        return cls(
            csv_path=csv_path,
            data_root=data_root,
            exact_positive_keys=exact_positive_keys,
            joint_positive_keys=joint_positive_keys,
            file_positive_keys=file_positive_keys,
            csv_total_rows=int(rows.shape[0]),
            duplicate_rows=duplicate_rows,
            unmatched_rows=unmatched_rows,
            strategy_counts=strategy_counts,
            file_column=file_column,
            dataset_column=dataset_column,
        )

    def label_for_path(self, path: str | Path) -> int:
        # ???? npy ???????????????????????????
        path = Path(path).resolve(strict=False)
        rel_key = path.relative_to(self.data_root.resolve(strict=False)).as_posix().lower()
        if rel_key in self.exact_positive_keys:
            return 1

        file_name = path.name.lower()
        dataset_dir = next((part for part in path.parts if part.lower().endswith('data')), path.parent.name)
        dataset_tag = _normalize_dataset_tag(dataset_dir)
        if dataset_tag is not None and (dataset_tag, file_name) in self.joint_positive_keys:
            return 1
        if file_name in self.file_positive_keys:
            return 1
        return 0

    def summarize_paths(self, files: list[Path]) -> LabelSummary:
        positives = sum(self.label_for_path(path) for path in files)
        total = len(files)
        return LabelSummary(total=total, positives=positives, negatives=total - positives)

    def log_summary(self, dataset_summaries: dict[str, LabelSummary]) -> None:
        LOGGER.info(
            'Loaded boundary CSV %s: rows=%s, unique_boundary_keys=%s, duplicate_rows=%s, exact=%s, joint=%s, file_only=%s',
            self.csv_path,
            self.csv_total_rows,
            self.csv_boundary_count,
            self.duplicate_rows,
            self.strategy_counts.get('exact', 0),
            self.strategy_counts.get('joint', 0),
            self.strategy_counts.get('file_only', 0),
        )
        for dataset_name, summary in dataset_summaries.items():
            LOGGER.info(
                'Boundary labels in %s: positives=%s, negatives=%s, total=%s',
                dataset_name,
                summary.positives,
                summary.negatives,
                summary.total,
            )
        if self.unmatched_rows:
            preview = ', '.join(self.unmatched_rows[:5])
            LOGGER.warning(
                'Boundary CSV contains %s rows that did not match any npy sample under %s. Examples: %s',
                len(self.unmatched_rows),
                self.data_root,
                preview,
            )
