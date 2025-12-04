#!/usr/bin/env python3
"""Reassemble a file previously split into `.partNN` chunks.

By default this looks for `<prefix>.part*` files and concatenates them
in lexicographic order into a single output file.

Example (for assay_rscores.parquet):
    python join_large_file.py \
        --prefix assay-etl/outputs/assay_rscores.parquet
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join .partNN chunk files back into a single file."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        type=Path,
        help=(
            "Base path used when splitting (e.g. "
            "'assay-etl/outputs/assay_rscores.parquet'). "
            "Chunks are expected to be named '<prefix>.partNN'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination path for the reassembled file. "
            "Defaults to the prefix path."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser.parse_args()


def join_file(prefix: Path, *, output: Path, force: bool) -> int:
    parts = sorted(prefix.parent.glob(prefix.name + ".part*"))
    if not parts:
        raise SystemExit(f"No part files found matching {prefix.name}.part* in {prefix.parent}")

    if output.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing file: {output}. "
            f"Pass --force to overwrite."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst)

    return len(parts)


def main() -> None:
    args = _parse_args()
    output = args.output or args.prefix

    parts_used = join_file(args.prefix, output=output, force=bool(args.force))
    print(f"Reassembled {parts_used} part file(s) into {output}")


if __name__ == "__main__":
    main()
