"""Convert MiniMind-V parquet data into GWen single-image jsonl."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image


def read_table(path: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to convert parquet data") from exc
    return pd.read_parquet(path)


def parse_conversations(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def image_from_value(value) -> Image.Image:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected single-image sample, got {len(value)} images")
        return image_from_value(value[0])
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise ValueError(f"MiniMind-V image_bytes must be bytes or single-element bytes list, got {type(value).__name__}")


def convert_one_file(parquet_path: str, output_handle, image_dir: Path, image_column: str, conversations_column: str, prefix: str) -> int:
    table = read_table(parquet_path)
    count = 0
    for idx, row in table.iterrows():
        conversations = parse_conversations(row[conversations_column])
        image = image_from_value(row[image_column])
        image_name = f"{prefix}_{idx:08d}.jpg"
        image_path = image_dir / image_name
        image.save(image_path, format="JPEG", quality=95)
        item = {
            "image": image_name,
            "messages": conversations,
        }
        output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert MiniMind-V parquet to GWen VLM jsonl")
    parser.add_argument("--input", required=True, help="Parquet file or directory")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--image_column", default="image_bytes")
    parser.add_argument("--conversations_column", default="conversations")
    args = parser.parse_args()

    input_path = Path(args.input)
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(input_path.glob("*.parquet")) if input_path.is_dir() else [input_path]
    total = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as output_handle:
        for file_idx, parquet_path in enumerate(parquet_files):
            total += convert_one_file(
                str(parquet_path),
                output_handle,
                image_dir,
                args.image_column,
                args.conversations_column,
                f"{parquet_path.stem}_{file_idx}",
            )
    print(f"Converted {total} samples to {args.output_jsonl}; images saved to {args.image_dir}")


if __name__ == "__main__":
    main()
