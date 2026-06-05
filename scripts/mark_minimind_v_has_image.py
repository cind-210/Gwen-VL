"""Mark MiniMind-V parquet rows with a has_image column."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_image_value(value) -> bytes:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected single-image sample, got {len(value)} images")
        return parse_image_value(value[0])
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise ValueError(f"image value must be bytes or single-element bytes list, got {type(value).__name__}")


def is_black_image(image_bytes: bytes, mean_threshold: float, std_threshold: float) -> bool:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    return float(arr.mean()) <= mean_threshold and float(arr.std()) <= std_threshold


def mark_file(input_path: Path, output_path: Path, image_column: str, mean_threshold: float, std_threshold: float) -> None:
    table = pd.read_parquet(input_path)
    if image_column not in table.columns:
        raise KeyError(f"{input_path} must contain {image_column!r}")
    has_image = []
    cache = {}
    cache_hits = 0
    for value in table[image_column]:
        image_bytes = parse_image_value(value)
        image_hash = hashlib.md5(image_bytes).digest()
        cached = cache.get(image_hash)
        if cached is None:
            cached = not is_black_image(image_bytes, mean_threshold, std_threshold)
            cache[image_hash] = cached
        else:
            cache_hits += 1
        has_image.append(cached)
    table["has_image"] = has_image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    true_count = sum(has_image)
    print(f"{input_path} -> {output_path}")
    print(f"rows={len(has_image)} has_image={true_count} no_image={len(has_image) - true_count}")
    print(f"unique_images={len(cache)} cache_hits={cache_hits}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add has_image column to MiniMind-V parquet")
    parser.add_argument("--input", required=True, help="Input parquet file or directory")
    parser.add_argument("--output", required=True, help="Output parquet file or directory")
    parser.add_argument("--image_column", default="image_bytes")
    parser.add_argument("--mean_threshold", type=float, default=1.0)
    parser.add_argument("--std_threshold", type=float, default=1.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for parquet_path in sorted(input_path.glob("*.parquet")):
            mark_file(
                parquet_path,
                output_path / parquet_path.name,
                args.image_column,
                args.mean_threshold,
                args.std_threshold,
            )
    else:
        mark_file(input_path, output_path, args.image_column, args.mean_threshold, args.std_threshold)


if __name__ == "__main__":
    main()
