from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FILENAME_LABEL_RE = re.compile(r"^(?P<prefix>.+)_(?P<label>[01])\.npy$", re.IGNORECASE)


def normalize_path(path: Path) -> str:
    return str(path.resolve(strict=False)).lower()


def parse_label_from_filename(path: Path) -> int | None:
    match = FILENAME_LABEL_RE.match(path.name)
    if not match:
        return None
    return int(match.group("label"))


def flip_filename_label(path: Path) -> Path | None:
    match = FILENAME_LABEL_RE.match(path.name)
    if not match:
        return None
    flipped = "1" if match.group("label") == "0" else "0"
    return path.with_name(f"{match.group('prefix')}_{flipped}.npy")


def parse_internal_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "stable"}:
            return 0
        if text in {"1", "unstable"}:
            return 1
    return None


def load_npy_dict(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == object:
        obj = payload.item()
    else:
        obj = payload
    if not isinstance(obj, dict):
        raise TypeError(f"npy payload is not dict: {type(obj)}")
    return obj


def atomic_save_npy_dict(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy", dir=str(path.parent)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.save(tmp_path, data, allow_pickle=True)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def collect_protected_npy_paths(saved_visuals_dir: Path, npy_root: Path) -> set[str]:
    protected: set[str] = set()
    if not saved_visuals_dir.exists():
        return protected
    for png_path in saved_visuals_dir.rglob("*.png"):
        rel = png_path.relative_to(saved_visuals_dir)
        npy_path = (npy_root / rel).with_suffix(".npy")
        protected.add(normalize_path(npy_path))
    return protected


def load_targets(mismatch_csv: Path) -> list[Path]:
    with mismatch_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "file" not in reader.fieldnames:
            raise ValueError("mismatch csv must include `file` column")
        values = [Path(row["file"]) for row in reader if row.get("file")]

    seen: set[str] = set()
    ordered: list[Path] = []
    for path in values:
        key = normalize_path(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_current_path(path_from_csv: Path) -> tuple[Path | None, str]:
    if path_from_csv.exists():
        return path_from_csv, "exact"
    flipped = flip_filename_label(path_from_csv)
    if flipped is not None and flipped.exists():
        return flipped, "flipped"
    return None, "missing"


def process_one(path: Path, dry_run: bool) -> tuple[str, str, int | None, int | None, str]:
    expected = parse_label_from_filename(path)
    if expected is None:
        return "unsupported_filename", "", None, None, "filename does not end with _0.npy/_1.npy"

    try:
        data = load_npy_dict(path)
    except Exception as exc:  # pragma: no cover
        return "load_error", "", None, expected, repr(exc)

    if "label" not in data:
        return "missing_label_key", "", None, expected, "key `label` not found in npy dict"

    old_value = data.get("label")
    old_label = parse_internal_label(old_value)

    if old_label == expected:
        return "unchanged", "", old_label, expected, ""

    data["label"] = int(expected)
    if not dry_run:
        try:
            atomic_save_npy_dict(path, data)
        except Exception as exc:  # pragma: no cover
            return "save_error", "", old_label, expected, repr(exc)
    return "updated", "", old_label, expected, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync npy internal `label` with filename suffix (_0/_1) for out_full mismatches."
    )
    parser.add_argument("--mismatch_csv", type=Path, default=Path("out_full/mismatch_results.csv"))
    parser.add_argument("--saved_visuals_dir", type=Path, default=Path("saved_visuals"))
    parser.add_argument("--npy_root", type=Path, default=Path("npy_jobs"))
    parser.add_argument("--manifest", type=Path, default=Path("label_fix_manifest.csv"))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    targets = load_targets(args.mismatch_csv)
    protected = collect_protected_npy_paths(args.saved_visuals_dir, args.npy_root)

    rows: list[dict[str, Any]] = []
    summary = {
        "total_targets": len(targets),
        "protected": 0,
        "updated": 0,
        "unchanged": 0,
        "missing": 0,
        "errors": 0,
    }

    for original in targets:
        current, resolve_mode = resolve_current_path(original)
        if current is None:
            summary["missing"] += 1
            rows.append(
                {
                    "source_path": str(original),
                    "final_path": "",
                    "old_internal_label": "",
                    "new_internal_label": "",
                    "old_name_label": "",
                    "new_name_label": "",
                    "status": "missing_file",
                    "note": f"resolve_mode={resolve_mode}",
                }
            )
            continue

        current_norm = normalize_path(current)
        filename_label = parse_label_from_filename(current)
        if current_norm in protected:
            summary["protected"] += 1
            rows.append(
                {
                    "source_path": str(original),
                    "final_path": str(current),
                    "old_internal_label": "",
                    "new_internal_label": "",
                    "old_name_label": filename_label,
                    "new_name_label": filename_label,
                    "status": "excluded",
                    "note": "protected sample from saved_visuals",
                }
            )
            continue

        status, note, old_internal, new_internal, extra = process_one(current, dry_run=args.dry_run)
        if status == "updated":
            summary["updated"] += 1
        elif status == "unchanged":
            summary["unchanged"] += 1
        elif status in {"missing_file"}:
            summary["missing"] += 1
        else:
            summary["errors"] += 1

        note_text = note if note else extra
        rows.append(
            {
                "source_path": str(original),
                "final_path": str(current),
                "old_internal_label": old_internal if old_internal is not None else "",
                "new_internal_label": new_internal if new_internal is not None else "",
                "old_name_label": filename_label if filename_label is not None else "",
                "new_name_label": filename_label if filename_label is not None else "",
                "status": status,
                "note": note_text,
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_path",
                "final_path",
                "old_internal_label",
                "new_internal_label",
                "old_name_label",
                "new_name_label",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
