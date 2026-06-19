"""CSV loading, image path parsing, and output writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import REPO_ROOT
from schema.columns import INPUT_COLUMNS, OUTPUT_COLUMNS, SAMPLE_COLUMNS


@dataclass(frozen=True)
class ImageRef:
    """One submitted image referenced by a claim row."""

    path: str
    image_id: str
    absolute_path: Path

    @property
    def exists(self) -> bool:
        return self.absolute_path.is_file()


@dataclass
class ClaimRecord:
    """One damage claim row with parsed image references."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    images: list[ImageRef] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def is_labeled(self) -> bool:
        return bool(self.labels)


def parse_image_paths(raw: str, repo_root: Path | None = None) -> list[ImageRef]:
    """Split semicolon-separated paths and derive image IDs from filename stems."""
    root = repo_root or REPO_ROOT
    if not raw or not str(raw).strip():
        return []

    refs: list[ImageRef] = []
    for segment in str(raw).split(";"):
        path = segment.strip()
        if not path:
            continue
        image_id = Path(path).stem
        refs.append(
            ImageRef(
                path=path,
                image_id=image_id,
                absolute_path=(root / path).resolve(),
            )
        )
    return refs


def _row_to_claim(row: pd.Series, repo_root: Path) -> ClaimRecord:
    record = ClaimRecord(
        user_id=str(row["user_id"]),
        image_paths=str(row["image_paths"]),
        user_claim=str(row["user_claim"]),
        claim_object=str(row["claim_object"]),
        images=parse_image_paths(str(row["image_paths"]), repo_root=repo_root),
    )
    for col in OUTPUT_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            record.labels[col] = str(row[col])
    return record


def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required input columns: {missing}")
    return df


def load_claims(csv_path: Path | None = None, repo_root: Path | None = None) -> list[ClaimRecord]:
    """Load input-only claims (e.g. dataset/claims.csv)."""
    path = csv_path or (REPO_ROOT / "dataset" / "claims.csv")
    root = repo_root or REPO_ROOT
    df = _read_csv(path)
    return [_row_to_claim(row, root) for _, row in df.iterrows()]


def load_sample_claims(
    csv_path: Path | None = None, repo_root: Path | None = None
) -> list[ClaimRecord]:
    """Load labeled sample claims including expected output columns."""
    path = csv_path or (REPO_ROOT / "dataset" / "sample_claims.csv")
    root = repo_root or REPO_ROOT
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [col for col in SAMPLE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return [_row_to_claim(row, root) for _, row in df.iterrows()]


def claim_to_output_row(record: ClaimRecord, predictions: dict[str, str]) -> dict[str, str]:
    """Build one output.csv row preserving exact column order."""
    row: dict[str, str] = {
        "user_id": record.user_id,
        "image_paths": record.image_paths,
        "user_claim": record.user_claim,
        "claim_object": record.claim_object,
    }
    for col in OUTPUT_COLUMNS:
        if col not in predictions:
            raise KeyError(f"Missing prediction field: {col}")
        row[col] = str(predictions[col])
    return row


def write_output_csv(
    records: list[ClaimRecord],
    predictions: list[dict[str, str]],
    output_path: Path | None = None,
) -> Path:
    """Write predictions to output.csv with exact schema column order."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")

    path = output_path or (REPO_ROOT / "dataset" / "output.csv")
    rows = [claim_to_output_row(rec, pred) for rec, pred in zip(records, predictions)]
    df = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    df.to_csv(path, index=False, quoting=1)
    return path
