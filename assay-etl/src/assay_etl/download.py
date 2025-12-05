"""Parallel FTP assay downloader with Atomic Writes, Robust Type Inference, and Smart Aggregation."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast, Set, Tuple, List, Dict
import zipfile
import shutil
import os
import signal
import concurrent.futures
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

import httpx
import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

METADATA_TAGS = {
    "RESULT_TYPE",
    "RESULT_DESCR",
    "RESULT_UNIT",
    "RESULT_IS_ACTIVE_CONCENTRATION",
    "RESULT_ATTR_CONC_MICROMOL",
}
GROUP_KEY = "PUBCHEM_CID"
EXCLUDE_GROUP_COLUMNS = {"PUBCHEM_SID"}

# Columns that must remain String
PROTECTED_TEXT_COLS = {"PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME"}

# Mapping for Activity Outcome Aggregation
# Logic: Active > Inconclusive/Unspecified > Inactive
OUTCOME_TO_INT = {"Active": 3, "Inconclusive": 2, "Unspecified": 2, "Inactive": 1}
INT_TO_OUTCOME = {3: "Active", 2: "Inconclusive", 1: "Inactive"}


@dataclass(slots=True)
class AssayTablePaths:
    aid: int
    parquet_path: Path


Compression = Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"]

# --- WORKER FUNCTIONS ---


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {col: col.strip() for col in df.columns if col.strip() != col}
    return df.rename(rename_map) if rename_map else df


def _separate_metadata(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    if "PUBCHEM_RESULT_TAG" not in df.columns:
        return df, {}
    metadata_mask = pl.col("PUBCHEM_RESULT_TAG").is_in(METADATA_TAGS)
    payload = df.filter(~metadata_mask)
    return payload, {}


def _apply_full_type_inference(df: pl.DataFrame) -> pl.DataFrame:
    """
    1. Keeps PROTECTED_TEXT_COLS as String.
    2. Keeps GROUP_KEY (CID) as is.
    3. Tries to cast EVERYTHING else to Float64.
    4. Drops any column that becomes purely null (garbage text cols).
    """

    # Identify columns to process (exclude Key and Protected)
    # We will process everything else.
    cols_to_process = [
        col
        for col in df.columns
        if col != GROUP_KEY and col not in EXCLUDE_GROUP_COLUMNS and col not in PROTECTED_TEXT_COLS
    ]

    if not cols_to_process:
        return df

    # We build a list of expressions.
    # If a cast succeeds, we keep it. If it fails (all nulls), we exclude it later.
    # To do this efficiently in Polars without iterating Python side too much:

    # 1. Cast all candidates to Float64 (strict=False turns errors to null)
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in cols_to_process]
    )

    # 2. Identify columns that are now ALL null (meaning they were text/garbage)
    #    and should be dropped to save space.
    #    We check null counts.
    null_counts = df.select([pl.col(c).null_count().alias(c) for c in cols_to_process]).row(0)

    total_rows = df.height
    cols_to_drop = [
        c for c, null_count in zip(cols_to_process, null_counts) if null_count == total_rows
    ]

    if cols_to_drop:
        df = df.drop(cols_to_drop)

    return df


def _aggregate_by_compound(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates rows by PUBCHEM_CID.
    - Numeric: Median
    - SMILES: Mode (ignoring nulls)
    - Activity: Custom Max Logic (Active > Inconclusive > Inactive)
    """
    if GROUP_KEY not in df.columns:
        return df

    # Identify columns by current type (after inference)
    numeric_cols = []
    str_cols = []

    for name, dtype in df.schema.items():
        if name == GROUP_KEY or name in EXCLUDE_GROUP_COLUMNS:
            continue
        if dtype.is_numeric():
            numeric_cols.append(name)
        else:
            str_cols.append(name)

    aggregations: list[pl.Expr] = []

    # 1. Numeric -> Median
    if numeric_cols:
        aggregations.extend(pl.col(col).median().alias(col) for col in numeric_cols)

    # 2. String Columns (Specific Handling)
    for col in str_cols:
        if col == "PUBCHEM_EXT_DATASOURCE_SMILES":
            # Mode returns a list of most common values. We take the first.
            # Polars mode ignores nulls automatically.
            aggregations.append(pl.col(col).mode().first().alias(col))

        elif col == "PUBCHEM_ACTIVITY_OUTCOME":
            # Map to Int -> Max -> Map back
            # We map strings to integers based on priority
            aggregations.append(
                pl.col(col)
                .replace(
                    OUTCOME_TO_INT, default=1, return_dtype=pl.Int8
                )  # Default to Inactive (1) if unknown
                .max()
                .replace(INT_TO_OUTCOME, default="Inactive")
                .alias(col)
            )
        else:
            # Fallback for other strings (e.g. URLs?): Mode
            aggregations.append(pl.col(col).mode().first().alias(col))

    aggregated = df.group_by(GROUP_KEY).agg(aggregations)
    return aggregated


def process_zip_worker(
    zip_path: Path,
    subset: Set[int],
    cache_dir: Path,
    compression: Compression,
    force_download: bool,
) -> list[AssayTablePaths]:
    """
    Worker function executed in a separate Process.
    """
    target_dir = cache_dir / f"tmp_{zip_path.stem}_{os.getpid()}"
    target_dir.mkdir(exist_ok=True, parents=True)
    results: list[AssayTablePaths] = []

    try:
        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(f"File {zip_path} is not a valid zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zip_stem = zip_path.stem
            all_files = set(zf.namelist())

            for aid in subset:
                parquet_path = cache_dir / f"aid_{aid}.parquet"
                if parquet_path.exists() and not force_download:
                    results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
                    continue

                fname = f"{aid}.csv.gz"
                full_name = f"{zip_stem}/{fname}"
                if full_name in all_files:
                    zf.extract(full_name, path=target_dir)
                    csv_path = target_dir / full_name
                    try:
                        df = pl.read_csv(
                            csv_path, infer_schema_length=1000, null_values=["", "NA", "NaN"]
                        )
                        df = _normalize_columns(df)
                        df, _ = _separate_metadata(df)

                        # New Logic applied here
                        df = _apply_full_type_inference(df)
                        df = _aggregate_by_compound(df)

                        df.write_parquet(parquet_path, compression=compression)
                        results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
                    finally:
                        csv_path.unlink(missing_ok=True)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(target_dir, ignore_errors=True)

    return results


# --- ORCHESTRATOR CLASS ---


class FTPAssayFetcher:
    FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data"

    def __init__(self, cache_dir: Path, parquet_compression: Compression = "zstd") -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_compression = parquet_compression
        self._client = httpx.Client(timeout=60.0, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def _download_zip(self, start: int, end: int) -> Path:
        url = f"{self.FTP_BASE}/{start:07d}_{end:07d}.zip"
        zip_path = self._cache_dir / f"{start:07d}_{end:07d}.zip"
        part_path = self._cache_dir / f"{start:07d}_{end:07d}.zip.part"

        if zip_path.exists():
            if zipfile.is_zipfile(zip_path):
                return zip_path
            else:
                zip_path.unlink()

        try:
            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                with part_path.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
            part_path.rename(zip_path)
            return zip_path
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

    def fetch_assays(
        self, aids: Sequence[int], force_download: bool, io_workers: int = 4, cpu_workers: int = 4
    ) -> list[AssayTablePaths]:
        existing_results: list[AssayTablePaths] = []
        missing_aids: list[int] = []

        for aid in aids:
            parquet_path = self._cache_dir / f"aid_{aid}.parquet"
            if parquet_path.exists() and not force_download:
                existing_results.append(AssayTablePaths(aid=aid, parquet_path=parquet_path))
            else:
                missing_aids.append(aid)

        ranges = self._group_into_ranges(missing_aids)
        final_results = list(existing_results)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                "[bold green]Total Progress", total=len(missing_aids) + len(existing_results)
            )
            progress.advance(main_task, len(existing_results))
            download_task = progress.add_task("[blue]Downloads", total=len(ranges))
            process_task = progress.add_task("[magenta]Processing", total=len(ranges))

            dl_pool = ThreadPoolExecutor(max_workers=io_workers)
            proc_pool = ProcessPoolExecutor(max_workers=cpu_workers)

            dl_futures: Dict[Future, Tuple[int, int, Set[int]]] = {}
            proc_futures: Dict[Future, Tuple[Path, int, int, int, Set[int]]] = {}

            try:
                queue_iter = iter(ranges)
                initial_batch_size = min(len(ranges), io_workers + 2)

                for _ in range(initial_batch_size):
                    try:
                        s, e, sub = next(queue_iter)
                        fut = dl_pool.submit(self._download_zip, s, e)
                        dl_futures[fut] = (s, e, sub)
                    except StopIteration:
                        break

                while dl_futures or proc_futures:
                    done_dl, _ = concurrent.futures.wait(
                        dl_futures.keys(),
                        timeout=0.05,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for fut in done_dl:
                        start, end, subset = dl_futures.pop(fut)
                        try:
                            zip_path = fut.result()
                            progress.advance(download_task)
                            proc_fut = proc_pool.submit(
                                process_zip_worker,
                                zip_path,
                                subset,
                                self._cache_dir,
                                self._parquet_compression,
                                force_download,
                            )
                            proc_futures[proc_fut] = (zip_path, len(subset), start, end, subset)
                        except Exception as err:
                            progress.console.print(f"[red]Download failed for {start}-{end}: {err}")

                        try:
                            ns, ne, nsub = next(queue_iter)
                            nfut = dl_pool.submit(self._download_zip, ns, ne)
                            dl_futures[nfut] = (ns, ne, nsub)
                        except StopIteration:
                            pass

                    done_proc, _ = concurrent.futures.wait(
                        proc_futures.keys(),
                        timeout=0.05,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for fut in done_proc:
                        zip_path, count, start_range, end_range, sub = proc_futures.pop(fut)
                        try:
                            batch_results = cast(list[AssayTablePaths], fut.result())
                            final_results.extend(batch_results)
                            progress.advance(process_task)
                            progress.advance(main_task, count)
                            zip_path.unlink(missing_ok=True)
                        except zipfile.BadZipFile:
                            progress.console.print(
                                f"[yellow]Corrupt Zip detected for {start_range}-{end_range}. Re-queueing download."
                            )
                            progress.update(download_task, advance=-1)
                            nfut = dl_pool.submit(self._download_zip, start_range, end_range)
                            dl_futures[nfut] = (start_range, end_range, sub)
                        except Exception as err:
                            progress.console.print(f"[red]Processing failed for {zip_path}: {err}")

            except KeyboardInterrupt:
                progress.console.print("[bold red]\nStopping... (Ctrl+C detected)")
                for f in dl_futures:
                    f.cancel()
                for f in proc_futures:
                    f.cancel()
                dl_pool.shutdown(wait=False, cancel_futures=True)
                proc_pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                dl_pool.shutdown(wait=True)
                proc_pool.shutdown(wait=True)

        return final_results

    def _group_into_ranges(self, aids: Sequence[int]) -> list[tuple[int, int, set[int]]]:
        ranges: dict[tuple[int, int], set[int]] = {}
        for aid in aids:
            start = ((aid - 1) // 1000) * 1000 + 1
            end = start + 999
            ranges.setdefault((start, end), set()).add(aid)
        return [(s, e, subset) for (s, e), subset in ranges.items()]


def download_assays(
    *,
    aids: Sequence[int],
    cache_dir: Path,
    force_download: bool = False,
    io_workers: int = 4,
    cpu_workers: int = 4,
) -> list[AssayTablePaths]:
    fetcher = FTPAssayFetcher(cache_dir=cache_dir)
    try:
        return fetcher.fetch_assays(
            aids, force_download, io_workers=io_workers, cpu_workers=cpu_workers
        )
    finally:
        fetcher.close()
