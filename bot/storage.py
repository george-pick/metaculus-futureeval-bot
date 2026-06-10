"""JSONL audit log: one record per question per run, with full reasoning."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from . import config


def log_forecast(record: dict[str, Any]) -> str:
    """Append a forecast record to today's JSONL log. Returns the path."""
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    path = config.LOG_DIR / f"forecasts_{day}.jsonl"
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        **record,
    }
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return str(path)
