"""Base adapter interface for all data sources."""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from engines.nil_props.config.settings import Settings

logger = logging.getLogger(__name__)


class AdapterResult:
    """Container for adapter fetch results."""

    def __init__(
        self,
        source: str,
        raw_payload: Any,
        records: list[dict],
        snapshot_path: Path | None = None,
        fetch_timestamp: datetime | None = None,
    ):
        self.source = source
        self.raw_payload = raw_payload
        self.records = records
        self.snapshot_path = snapshot_path
        self.fetch_timestamp = fetch_timestamp or datetime.utcnow()

    @property
    def record_count(self) -> int:
        return len(self.records)

    def payload_hash(self) -> str:
        raw = json.dumps(self.raw_payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class BaseAdapter(ABC):
    """Shared interface for all source adapters."""

    SOURCE_NAME: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(f"adapter.{self.SOURCE_NAME}")

    @abstractmethod
    def authenticate(self) -> bool:
        """Validate credentials / connectivity."""

    @abstractmethod
    def fetch_historical(self, season: str, **kwargs) -> AdapterResult:
        """Fetch full historical data for a season."""

    @abstractmethod
    def fetch_incremental(self, since: datetime, **kwargs) -> AdapterResult:
        """Fetch data since a given timestamp."""

    @abstractmethod
    def normalize_payload(self, raw: Any) -> list[dict]:
        """Convert raw API response to canonical dicts."""

    def persist_raw_snapshot(self, result: AdapterResult, base_dir: Path) -> Path:
        """Save raw payload to disk for audit trail."""
        ts = result.fetch_timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_dir = base_dir / "raw" / self.SOURCE_NAME
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"{self.SOURCE_NAME}_{ts}.json"
        with open(path, "w") as f:
            json.dump(result.raw_payload, f, indent=2, default=str)
        result.snapshot_path = path
        self.logger.info(f"Saved raw snapshot: {path} ({result.record_count} records)")
        return path

    def validate_records(self, records: list[dict]) -> dict:
        """Basic validation metrics."""
        if not records:
            return {"total": 0, "valid": 0, "missing_fields": {}}
        required = self._required_fields()
        missing = {}
        valid = 0
        for rec in records:
            row_ok = True
            for field in required:
                if field not in rec or rec[field] is None:
                    missing[field] = missing.get(field, 0) + 1
                    row_ok = False
            if row_ok:
                valid += 1
        return {"total": len(records), "valid": valid, "missing_fields": missing}

    def _required_fields(self) -> list[str]:
        """Override in subclasses to specify required fields."""
        return []
