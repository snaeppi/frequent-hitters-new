#!/usr/bin/env python3
"""Split a large file into smaller chunks.

This is useful for working around per-file size limits (e.g. Git LFS),
by storing multiple smaller `.partNN` files instead of one huge file.

Example (for assay_rscores.parquet):
    python split_large_file.py \
        --input assay-etl/outputs/assay_rscores.parquet \
        --chunk-size-gb 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a large file into fixed-size chunks.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input file to split.",
    )
    parser.add_argument(
        "--chunk-size-gb",
        type=float,
        default=1.5,
        help="Target chunk size in GiB (default: 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write chunks into (defaults to the input's directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing part files if they already exist.",
    )
    return parser.parse_args()


def split_file(path: Path, *, chunk_size_bytes: int, output_dir: Path, force: bool) -> int:
    if not path.is_file():
        raise SystemExit(f"Input file does not exist or is not a file: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_size = path.stat().st_size
    if total_size <= chunk_size_bytes:
        raise SystemExit(
            f"Input file is only {total_size} bytes; "
            f"use a smaller chunk size or skip splitting."
        )

    part_index = 1
    with path.open("rb") as src:
        while True:
            chunk = src.read(chunk_size_bytes)
            if not chunk:
                break

            part_name = f"{path.name}.part{part_index:02d}"
            part_path = output_dir / part_name

            if part_path.exists() and not force:
                raise SystemExit(
                    f"Refusing to overwrite existing file: {part_path}. "
                    f"Pass --force to overwrite."
                )

            with part_path.open("wb") as dst:
                dst.write(chunk)

            part_index += 1

    return part_index - 1


def main() -> None:
    args = _parse_args()
    chunk_size_bytes = int(args.chunk_size_gb * (1024**3))

    output_dir = args.output_dir if args.output_dir is not None else args.input.parent

    parts_written = split_file(
        args.input,
        chunk_size_bytes=chunk_size_bytes,
        output_dir=output_dir,
        force=bool(args.force),
    )

    print(f"Wrote {parts_written} part file(s) to {output_dir}")


if __name__ == "__main__":
    main()
