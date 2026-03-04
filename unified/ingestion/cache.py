"""Unified SQLite cache with endpoint health tracking.

All ingestion clients use this to avoid redundant API calls.
If data exists in cache and is fresh, the API call is skipped.

Based on: NBA_Props_AI/core_best_v3/nba_props/cache.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CacheHit:
    key: str
    asof_utc: str
    data_json: str


@dataclass(frozen=True)
class EndpointHealth:
    endpoint: str
    calls: int
    success_count: int
    failure_count: int
    success_rate: float
    last_success_utc: Optional[str]
    last_failure_utc: Optional[str]
    last_error: Optional[str]


class SQLiteCache:
    """SQLite-backed cache for API responses with endpoint health tracking."""

    def __init__(self, path: str = "data/cache.sqlite"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
              key TEXT PRIMARY KEY,
              endpoint TEXT NOT NULL,
              params_json TEXT NOT NULL,
              asof_utc TEXT NOT NULL,
              data_json TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS endpoint_health (
              endpoint TEXT PRIMARY KEY,
              calls INTEGER NOT NULL DEFAULT 0,
              success_count INTEGER NOT NULL DEFAULT 0,
              failure_count INTEGER NOT NULL DEFAULT 0,
              last_success_utc TEXT,
              last_failure_utc TEXT,
              last_error TEXT,
              updated_utc TEXT NOT NULL
            )
        """)
        con.commit()
        con.close()

    def _hash_key(self, endpoint: str, params: dict) -> str:
        blob = json.dumps(
            {"endpoint": endpoint, "params": params},
            sort_keys=True, default=str,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, endpoint: str, params: dict) -> Optional[CacheHit]:
        """Look up cached response by endpoint + params."""
        key = self._hash_key(endpoint, params)
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            "SELECT asof_utc, data_json FROM snapshots WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return CacheHit(key=key, asof_utc=row[0], data_json=row[1])

    def set(
        self,
        endpoint: str,
        params: dict,
        data: Any,
        asof_utc: Optional[str] = None,
    ):
        """Store API response in cache."""
        key = self._hash_key(endpoint, params)
        if asof_utc is None:
            asof_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            "REPLACE INTO snapshots (key, endpoint, params_json, asof_utc, data_json) "
            "VALUES (?,?,?,?,?)",
            (
                key, endpoint,
                json.dumps(params, sort_keys=True, default=str),
                asof_utc,
                json.dumps(data, default=str),
            ),
        )
        con.commit()
        con.close()

    def record_endpoint_call(
        self,
        endpoint: str,
        *,
        ok: bool,
        error: Optional[str] = None,
        asof_utc: Optional[str] = None,
    ) -> None:
        """Track endpoint health for monitoring."""
        if asof_utc is None:
            asof_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        err_text = str(error or "").strip()
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO endpoint_health (
              endpoint, calls, success_count, failure_count,
              last_success_utc, last_failure_utc, last_error, updated_utc
            )
            VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
              calls = calls + 1,
              success_count = success_count + ?,
              failure_count = failure_count + ?,
              last_success_utc = CASE WHEN ? = 1 THEN excluded.last_success_utc ELSE last_success_utc END,
              last_failure_utc = CASE WHEN ? = 1 THEN excluded.last_failure_utc ELSE last_failure_utc END,
              last_error = CASE WHEN ? = 1 THEN excluded.last_error ELSE last_error END,
              updated_utc = excluded.updated_utc
            """,
            (
                str(endpoint),
                1 if ok else 0,
                0 if ok else 1,
                asof_utc if ok else None,
                None if ok else asof_utc,
                None if ok else err_text[:500],
                asof_utc,
                1 if ok else 0,
                0 if ok else 1,
                1 if ok else 0,
                0 if ok else 1,
                0 if ok else 1,
            ),
        )
        con.commit()
        con.close()

    def get_endpoint_health(self, endpoint: str) -> Optional[EndpointHealth]:
        """Query health stats for a single endpoint."""
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            "SELECT endpoint, calls, success_count, failure_count, "
            "last_success_utc, last_failure_utc, last_error "
            "FROM endpoint_health WHERE endpoint = ?",
            (str(endpoint),),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        calls = int(row[1] or 0)
        success = int(row[2] or 0)
        failure = int(row[3] or 0)
        return EndpointHealth(
            endpoint=str(row[0]),
            calls=calls,
            success_count=success,
            failure_count=failure,
            success_rate=float(success / calls) if calls > 0 else 0.0,
            last_success_utc=row[4],
            last_failure_utc=row[5],
            last_error=row[6],
        )

    def get_endpoint_health_map(self, endpoints: List[str]) -> Dict[str, EndpointHealth]:
        """Query health for multiple endpoints."""
        out: Dict[str, EndpointHealth] = {}
        for name in endpoints:
            h = self.get_endpoint_health(name)
            if h is not None:
                out[str(name)] = h
        return out
