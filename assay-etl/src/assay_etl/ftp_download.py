"""Parallel FTP assay downloader with robust type inference, compound-level aggregation, Rich progress bar, and force_download support."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast
import zipfile
import shutil
import httpx
import polars as pl
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

METADATA_TAGS = {
    "RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION", "RESULT_ATTR_CONC_MICROMOL",
}
GROUP_KEY = "PUBCHEM_CID"
EXCLUDE_GROUP_COLUMNS = {"PUBCHEM_SID"}

@dataclass(slots=True)
class AssayTablePaths:
    aid: int
    parquet_path: Path

Compression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]


class FTPAssayFetcher:
    FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data"

    def __init__(self, cache_dir: Path, parquet_compression: Compression = "zstd") -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_compression = parquet_compression
        self._client = httpx.Client(timeout=300.0)

    def close(self) -> None:
        self._client.close()

    def fetch_assays(self, aids: Sequence[int], force_download: bool, max_workers: int = 4) -> list[AssayTablePaths]:
        # First, detect which assays already have a materialized parquet so we
        # can avoid re-downloading fully processed ranges after an interrupted
        # run. When force_download is True we always recompute everything.
        existing_results: list[AssayTablePaths] = []
        missing_aids: list[int] = []

        for aid in aids:
            parquet_path = self._cache_dir / f"aid_{aid}.parquet"
            if parquet_path.exists() and not force_download:
                existing_results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
            else:
                missing_aids.append(aid)

        ranges = self._group_into_ranges(missing_aids)
        results: list[AssayTablePaths] = list(existing_results)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
        ) as progress:
            assay_task_id: TaskID = progress.add_task("Downloading & Processing assays", total=len(aids))

            # Immediately mark any assays that already have parquet files as
            # completed so the main progress bar "jumps" to the next missing
            # assay on restart.
            if existing_results:
                progress.advance(assay_task_id, advance=len(existing_results))

            # Download and process each ZIP range in a pipelined fashion so that
            # processing can start as soon as the first archive finishes. Only
            # active downloads get their own progress bar to avoid clutter.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: list[Future[list[AssayTablePaths]]] = []

                for start, end, subset in ranges:
                    futures.append(
                        executor.submit(
                            self._download_and_process_range,
                            start,
                            end,
                            subset,
                            assay_task_id,
                            force_download,
                            progress,
                        )
                    )

                for future in as_completed(futures):
                    results.extend(cast(list[AssayTablePaths], future.result()))

            progress.update(assay_task_id, description="[green]All assays processed")

        return results

    def _group_into_ranges(self, aids: Sequence[int]) -> list[tuple[int,int,set[int]]]:
        ranges: dict[tuple[int,int], set[int]] = {}
        for aid in aids:
            start = ((aid - 1) // 1000) * 1000 + 1
            end = start + 999
            ranges.setdefault((start, end), set()).add(aid)
        return [(s, e, subset) for (s, e), subset in ranges.items()]

    def _download_zip(
        self,
        start: int,
        end: int,
        progress: Progress | None = None,
    ) -> Path:
        url = f"{self.FTP_BASE}/{start:07d}_{end:07d}.zip"
        zip_path = self._cache_dir / f"{start:07d}_{end:07d}.zip"
        if zip_path.exists():
            return zip_path

        task_id: TaskID | None = None
        if progress is not None:
            task_id = progress.add_task(
                f"Downloading {start:07d}_{end:07d}.zip",
                total=0,
            )
        else:
            print(f"Downloading {url}...")

        total_bytes = 0
        with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_length_header = resp.headers.get("Content-Length")
            try:
                total_bytes = int(content_length_header) if content_length_header else 0
            except ValueError:
                total_bytes = 0

            if progress is not None and task_id is not None:
                if total_bytes > 0:
                    progress.update(task_id, total=total_bytes)
                else:
                    progress.update(task_id, total=1)

            with zip_path.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
                    if progress is not None and task_id is not None and total_bytes > 0:
                        progress.update(task_id, advance=len(chunk))

        if progress is not None and task_id is not None:
            if total_bytes == 0:
                progress.update(task_id, advance=1)
            # Mark the task as completed and then remove it so that only
            # currently active downloads are shown.
            progress.update(task_id, description=f"[green]Downloaded {zip_path.name}")
            progress.remove_task(task_id)

        return zip_path

    def _download_and_process_range(
        self,
        start: int,
        end: int,
        subset: set[int],
        assay_task_id: TaskID,
        force_download: bool,
        progress: Progress,
    ) -> list[AssayTablePaths]:
        zip_path = self._cache_dir / f"{start:07d}_{end:07d}.zip"
        if not zip_path.exists():
            zip_path = self._download_zip(start, end, progress)

        return self._process_zip(
            zip_path=zip_path,
            subset=subset,
            progress=progress,
            task_id=assay_task_id,
            force_download=force_download,
        )

    def _process_zip(
        self,
        zip_path: Path,
        subset: set[int],
        progress: Progress,
        task_id: TaskID,
        force_download: bool,
    ) -> list[AssayTablePaths]:
        target_dir = self._cache_dir / f"tmp_{zip_path.stem}"
        target_dir.mkdir(exist_ok=True)
        results: list[AssayTablePaths] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            zip_stem = zip_path.stem
            for aid in subset:
                progress.update(task_id, description=f"Processing AID {aid}")
                parquet_path = self._cache_dir / f"aid_{aid}.parquet"
                if parquet_path.exists() and not force_download:
                    results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
                    progress.advance(task_id)
                    continue

                fname = f"{aid}.csv.gz"
                full_name = f"{zip_stem}/{fname}"
                if full_name in zf.namelist():
                    zf.extract(full_name, path=target_dir)
                    csv_path = target_dir / full_name
                    df = pl.read_csv(csv_path)
                    df = self._normalize_columns(df)
                    df, _ = self._separate_metadata(df)
                    df = self._apply_full_type_inference(df)
                    df = self._aggregate_by_compound(df)
                    df.write_parquet(parquet_path, compression=self._parquet_compression)
                    results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
                progress.advance(task_id)
        shutil.rmtree(target_dir)
        zip_path.unlink(missing_ok=True)
        return results

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {col: col.strip() for col in df.columns if col.strip() != col}
        return df.rename(rename_map) if rename_map else df

    def _separate_metadata(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str,str]]:
        if "PUBCHEM_RESULT_TAG" not in df.columns:
            return df, {}
        metadata_mask = pl.col("PUBCHEM_RESULT_TAG").is_in(METADATA_TAGS)
        metadata = df.filter(metadata_mask)
        payload = df.filter(~metadata_mask)
        return payload, {}

    def _apply_full_type_inference(self, df: pl.DataFrame) -> pl.DataFrame:
        casts = []
        for name, dtype in zip(df.columns, df.dtypes):
            if name == GROUP_KEY or name in EXCLUDE_GROUP_COLUMNS:
                continue
            if dtype.is_numeric():
                continue
            try:
                df[name].cast(pl.Float64, strict=True)
                casts.append(pl.col(name).cast(pl.Float64, strict=False).alias(name))
            except Exception:
                pass
        return df.with_columns(casts) if casts else df

    def _aggregate_by_compound(self, df: pl.DataFrame) -> pl.DataFrame:
        if GROUP_KEY not in df.columns:
            return df
        numeric_cols, categorical_cols = [], []
        for name, dtype in zip(df.columns, df.dtypes):
            if name == GROUP_KEY or name in EXCLUDE_GROUP_COLUMNS:
                continue
            if dtype.is_numeric():
                numeric_cols.append(name)
            else:
                categorical_cols.append(name)
        aggregations: list[pl.Expr] = []
        aggregations.extend(pl.col(col).median().alias(col) for col in numeric_cols)
        mode_aliases = {col: f"__mode_{col}" for col in categorical_cols}
        aggregations.extend(pl.col(col).mode().alias(alias) for col, alias in mode_aliases.items())
        aggregated = df.group_by(GROUP_KEY).agg(aggregations)
        if mode_aliases:
            aggregated = (
                aggregated.with_columns(
                    [pl.col(alias).list.first().alias(col) for col, alias in mode_aliases.items()]
                ).drop(list(mode_aliases.values()))
            )
        return aggregated

def download_assays(
    *,
    aids: Sequence[int],
    cache_dir: Path,
    force_download: bool = False,
    max_workers: int = 4
) -> list[AssayTablePaths]:
    fetcher = FTPAssayFetcher(cache_dir=cache_dir)
    try:
        return fetcher.fetch_assays(aids, force_download, max_workers=max_workers)
    finally:
        fetcher.close()
