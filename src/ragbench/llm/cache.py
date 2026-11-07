"""SQLite-backed response cache so reruns are free and reproducible."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ragbench.types import LLMResponse


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._conn.commit()

    @staticmethod
    def key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return LLMResponse.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, resp: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, json.dumps(resp.to_dict()))
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
