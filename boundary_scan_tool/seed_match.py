from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_dataset_name(value: Any) -> str:
    text = str(value or '').strip().lower()
    digits = ''.join(ch for ch in text if ch.isdigit())
    return digits or text


def _normalize_filename(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    return Path(text).name.lower()


def _normalize_rel_path(value: Any, input_root: Path) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve(strict=False).relative_to(input_root.resolve(strict=False))
        except ValueError:
            return ''
    return candidate.as_posix().lower()


@dataclass(frozen=True)
class SeedBoundaryIndex:
    csv_path: Path
    input_root: Path
    total_rows: int
    exact_file_keys: frozenset[str]
    abs_path_keys: frozenset[str]
    sample_dataset_keys: frozenset[tuple[str, str]]

    @classmethod
    def from_csv(cls, csv_path: str | Path, input_root: str | Path) -> 'SeedBoundaryIndex':
        csv_path = Path(csv_path).resolve(strict=False)
        input_root = Path(input_root).resolve(strict=False)
        if not csv_path.exists():
            raise FileNotFoundError(f'seed csv not found: {csv_path}')

        with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        exact_file_keys: set[str] = set()
        abs_path_keys: set[str] = set()
        sample_dataset_keys: set[tuple[str, str]] = set()

        for row in rows:
            rel_path = ''
            if 'file' in fieldnames:
                rel_path = _normalize_rel_path(row.get('file'), input_root)
            if rel_path:
                exact_file_keys.add(rel_path)

            if 'source_abs_path' in fieldnames:
                abs_path = str(row.get('source_abs_path') or '').strip()
                if abs_path:
                    abs_path_keys.add(str(Path(abs_path).resolve(strict=False)).lower())

            dataset_name = _normalize_dataset_name(row.get('dataset_name') or row.get('dataset_dir_name'))
            sample_name = _normalize_filename(row.get('sample_name') or row.get('file'))
            if dataset_name and sample_name:
                sample_dataset_keys.add((dataset_name, sample_name))

        return cls(
            csv_path=csv_path,
            input_root=input_root,
            total_rows=len(rows),
            exact_file_keys=frozenset(exact_file_keys),
            abs_path_keys=frozenset(abs_path_keys),
            sample_dataset_keys=frozenset(sample_dataset_keys),
        )

    def match_row(self, row: dict[str, Any]) -> tuple[int, str]:
        rel_path = _normalize_rel_path(row.get('file'), self.input_root)
        abs_path = str(Path(str(row.get('source_abs_path', ''))).resolve(strict=False)).lower() if row.get('source_abs_path') else ''
        dataset_name = _normalize_dataset_name(row.get('dataset_name') or row.get('dataset_dir_name'))
        sample_name = _normalize_filename(row.get('sample_name') or row.get('file'))

        if rel_path and rel_path in self.exact_file_keys:
            return 1, f'file::{rel_path}'
        if abs_path and abs_path in self.abs_path_keys:
            return 1, f'abs::{abs_path}'
        if dataset_name and sample_name and (dataset_name, sample_name) in self.sample_dataset_keys:
            return 1, f'name::{dataset_name}/{sample_name}'
        fallback_key = rel_path or abs_path or f'{dataset_name}/{sample_name}'
        return 0, fallback_key

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        flags: list[int] = []
        keys: list[str] = []
        for row in result.to_dict(orient='records'):
            flag, key = self.match_row(row)
            flags.append(flag)
            keys.append(key)
        result['is_seed_boundary'] = flags
        result['seed_match_key'] = keys
        return result
