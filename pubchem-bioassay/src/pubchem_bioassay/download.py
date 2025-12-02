"""Fetch assay data tables via compressed CSV download and cache as Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import httpx
import polars as pl

from .utils import RateLimiter
    

METADATA_TAGS = {
    "RESULT_TYPE",
    "RESULT_DESCR",
    "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION",
    "RESULT_ATTR_CONC_MICROMOL",
}
NUMERIC_TYPE_HINTS = {
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "NUMBER",
    "PERCENT",
    "INTEGER",
    "INT",
}


@dataclass(slots=True)
class AssayTablePaths:
    aid: int
    parquet_path: Path


class AssayCSVFetcher:
    """Fetch assay data via the compressed web download and cache as Parquet."""

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        rate_limit_hz: float = 4.0,
        cache_dir: Path | None = None,
        force_download: bool = False,
        parquet_compression: str = "zstd",
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": "pubchem-bioassay/0.1"}
        )
        self._limiter = RateLimiter(max_per_second=rate_limit_hz)
        self._cache_dir = cache_dir or Path("data/assay_tables")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._force_download = force_download
        self._parquet_compression = parquet_compression

    def close(self) -> None:
        self._client.close()

    def fetch_dataframe(self, aid: int) -> AssayTablePaths:
        parquet_path = self._cache_dir / f"aid_{aid}.parquet"
        if parquet_path.exists() and not self._force_download:
            # Ensure we can read it, but otherwise just return the path.
            pl.read_parquet(parquet_path)
            return AssayTablePaths(aid=aid, parquet_path=parquet_path)

        csv_path = self._cache_dir / f"aid_{aid}.csv.gz"
        self._download_gzip(aid, csv_path)
        df = pl.read_csv(csv_path)
        df = self._normalize_columns(df)
        df, type_hints = self._separate_metadata(df)
        df = self._apply_type_hints(df, type_hints)
        df.write_parquet(parquet_path, compression=self._parquet_compression)
        try:
            csv_path.unlink()
        except FileNotFoundError:
            pass
        return AssayTablePaths(aid=aid, parquet_path=parquet_path)

    def _download_gzip(self, aid: int, target: Path) -> None:
        tmp_path = target.with_suffix(".tmp")
        self._limiter.wait()
        params = {
            "query": "download",
            "record_type": "datatable",
            "actvty": "all",
            "response_type": "save",
            "aid": str(aid),
            "compress": "1",
        }
        with self._client.stream(
            "GET",
            "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi",
            params=params,
        ) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        tmp_path.replace(target)

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {col: col.strip() for col in df.columns if col.strip() != col}
        if rename_map:
            df = df.rename(rename_map)
        return df

    def _separate_metadata(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, dict[str, str]]:
        if "PUBCHEM_RESULT_TAG" not in df.columns:
            return df, {}
        metadata_mask = pl.col("PUBCHEM_RESULT_TAG").is_in(METADATA_TAGS)
        metadata = df.filter(metadata_mask)
        payload = df.filter(~metadata_mask)
        type_hints: dict[str, str] = {}
        if metadata.height > 0:
            type_row = metadata.filter(pl.col("PUBCHEM_RESULT_TAG") == "RESULT_TYPE")
            if type_row.height > 0:
                type_dict = type_row.to_dicts()[0]
                type_hints = {
                    col: (value or "")
                    for col, value in type_dict.items()
                    if col != "PUBCHEM_RESULT_TAG" and value
                }
        return payload, type_hints

    def _apply_type_hints(
        self, df: pl.DataFrame, type_hints: dict[str, str]
    ) -> pl.DataFrame:
        casts = []
        for col, hint in type_hints.items():
            dtype = self._dtype_for_hint(hint)
            if dtype is None or col not in df.columns:
                continue
            casts.append(pl.col(col).cast(dtype, strict=False).alias(col))
        if casts:
            df = df.with_columns(casts)
        return df

    def _dtype_for_hint(self, hint: str | None) -> pl.DataType | None:
        if not hint:
            return None
        normalized = hint.strip().upper()
        if normalized in NUMERIC_TYPE_HINTS:
            return pl.Float64
        return None


def download_assays(
    *,
    aids: Sequence[int],
    cache_dir: Path,
    force_download: bool = False,
    timeout: float = 120.0,
    rate_limit_hz: float = 4.0,
) -> list[AssayTablePaths]:
    """Download CSV.gz tables for the given AIDs and materialize them as Parquet."""
    fetcher = AssayCSVFetcher(
        timeout=timeout,
        rate_limit_hz=rate_limit_hz,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    try:
        results: list[AssayTablePaths] = []
        for aid in aids:
            results.append(fetcher.fetch_dataframe(aid))
        return results
    finally:
        fetcher.close()


