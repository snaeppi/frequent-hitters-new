from __future__ import annotations

import time
from pathlib import Path


class RateLimiter:
    """Simple time-based rate limiter."""

    def __init__(self, max_per_second: float) -> None:
        self._interval = 1.0 / max_per_second if max_per_second > 0 else 0.0
        self._next_time = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        if now < self._next_time:
            time.sleep(self._next_time - now)
        self._next_time = time.monotonic() + self._interval


def assay_table_path(base_dir: Path, aid: int) -> Path:
    """Return the per-assay table path for a given AID.

    Prefers the current naming (aid_<AID>.parquet) but falls back to the
    legacy suffix (aid_<AID>_cid_agg.parquet) if present.
    """
    preferred = base_dir / f"aid_{aid}.parquet"
    legacy = base_dir / f"aid_{aid}_cid_agg.parquet"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred
