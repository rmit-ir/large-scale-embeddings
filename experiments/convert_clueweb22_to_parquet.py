#!/usr/bin/env python3
"""
Convert ClueWeb22-B JSON.gz files to optimized Parquet format.
Optimized for small size and fast random read access for API serving.
"""

import json
import gzip
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


def find_json_gz_files(root_dir: Path) -> list[Path]:
    """
    Recursively find all .json.gz files in the given directory.

    Args:
        root_dir: Root directory to search (e.g., /path/to/ClueWeb22-B or /path/to/ClueWeb22-B/txt/en)

    Returns:
        List of Path objects to .json.gz files
    """
    if not root_dir.exists():
        raise ValueError(f"Directory not found: {root_dir}")

    json_gz_files = sorted(root_dir.rglob("*.json.gz"))

    return json_gz_files


def read_json_gz_lines(file_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Read and parse JSON lines from a gzipped file.

    Args:
        file_path: Path to .json.gz file

    Yields:
        Parsed JSON objects as dictionaries
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Failed to parse JSON at {file_path}:{line_num} - {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


def create_optimized_schema() -> pa.Schema:
    """
    Create an optimized PyArrow schema for ClueWeb22 data.
    Uses dictionary encoding for string fields to reduce size and improve read speed.
    """
    return pa.schema([
        ('URL', pa.dictionary(pa.int32(), pa.string())),
        ('URL_hash', pa.string()),  # Hash values are unique, no dictionary benefit
        ('Language', pa.dictionary(pa.int8(), pa.string())),  # Few unique values
        ('ClueWeb22_ID', pa.string()),  # IDs are unique
        ('Clean_Text', pa.string()),  # Text content
    ])


def process_records_to_parquet(
    json_gz_files: list[Path],
    output_path: Path,
    batch_size: int = 100000,
    compression: str = 'zstd',
    compression_level: int = 3,
):
    """
    Process JSON.gz files and write to optimized Parquet format.

    Args:
        json_gz_files: List of JSON.gz files to process
        output_path: Output Parquet file path
        batch_size: Number of records to batch before writing (affects row group size)
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'lz4')
        compression_level: Compression level (higher = smaller but slower)
    """
    schema = create_optimized_schema()
    writer = None

    batch_data = {
        'URL': [],
        'URL_hash': [],
        'Language': [],
        'ClueWeb22_ID': [],
        'Clean_Text': []
    }

    # Track ID to row position for index
    id_to_row = []
    total_records = 0

    try:
        with tqdm(total=len(json_gz_files), desc="Processing files", unit="file") as pbar:
            for file_path in json_gz_files:
                for record in read_json_gz_lines(file_path):
                    clueweb_id = record.get('ClueWeb22-ID', '')
                    clean_text = record.get('Clean-Text', '').strip()
                    if not clean_text:
                        continue  # Skip records with empty Clean-Text

                    # Extract fields with fallback for missing keys
                    batch_data['URL'].append(record.get('URL', '').strip())
                    batch_data['URL_hash'].append(record.get('URL-hash', ''))
                    batch_data['Language'].append(record.get('Language', ''))
                    batch_data['ClueWeb22_ID'].append(clueweb_id)
                    batch_data['Clean_Text'].append(clean_text)

                    # Track ID to row mapping
                    id_to_row.append((clueweb_id, total_records))

                    total_records += 1

                    # Write batch when it reaches batch_size
                    if len(batch_data['URL']) >= batch_size:
                        table = pa.Table.from_pydict(batch_data, schema=schema)

                        if writer is None:
                            # Initialize writer on first batch
                            writer = pq.ParquetWriter(
                                output_path,
                                schema,
                                compression=compression,
                                compression_level=compression_level,
                                use_dictionary=True,  # Enable dictionary encoding
                                write_statistics=True,  # Enable statistics for faster filtering
                                data_page_size=1024*1024,  # 1MB pages for good random access
                            )

                        writer.write_table(table)

                        # Clear batch
                        for key in batch_data:
                            batch_data[key] = []

                        tqdm.write(f"Processed {total_records:,} records...")

                pbar.update(1)

        # Write remaining records
        if batch_data['URL']:
            table = pa.Table.from_pydict(batch_data, schema=schema)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression=compression,
                    compression_level=compression_level,
                    use_dictionary=True,
                    write_statistics=True,
                    data_page_size=1024*1024,
                )

            writer.write_table(table)
            print(f"Wrote final batch. Total records: {total_records:,}")

    finally:
        if writer is not None:
            writer.close()

    # Create index file
    create_index_file(output_path, id_to_row)

    # Print summary
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"Total records: {total_records:,}")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Compression: {compression} (level {compression_level})")
        print(f"{'='*60}")

    return total_records


def create_index_file(parquet_path: Path, id_to_row: list):
    """
    Create a separate index file for fast ClueWeb22_ID lookups.
    The index maps ID -> row position in the main Parquet file.

    When reading it, just read it to memory and do binary search on the sorted IDs.

    Args:
        parquet_path: Path to the main Parquet file
        id_to_row: List of (ClueWeb22_ID, row_position) tuples
    """
    index_path = parquet_path.with_suffix('.index.parquet')

    print(f"\nCreating index file: {index_path}")
    print(f"Sorting {len(id_to_row):,} ID mappings...")

    # Sort by ID for binary search capability
    id_to_row.sort(key=lambda x: x[0])

    # Write index in one go (index should be much smaller than data)
    index_data = {
        'ClueWeb22_ID': [item[0] for item in id_to_row],
        'row_position': [item[1] for item in id_to_row]
    }

    index_table = pa.Table.from_pydict(index_data, schema=pa.schema([
        ('ClueWeb22_ID', pa.string()),
        ('row_position', pa.int64()),
    ]))

    pq.write_table(
        index_table,
        index_path,
        compression='zstd',
        compression_level=9,  # Higher compression for index
        use_dictionary=False,  # IDs are unique, no benefit
    )

    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"Index file created: {index_size_mb:.2f} MB")
    print(f"Index is sorted by ClueWeb22_ID for fast lookups")


def verify_parquet_file(parquet_path: Path, sample_size: int = 5):
    """
    Verify the Parquet file and show sample records.

    Args:
        parquet_path: Path to Parquet file
        sample_size: Number of sample records to display
    """
    print(f"\nVerifying Parquet file: {parquet_path}")

    # Read metadata
    parquet_file = pq.ParquetFile(parquet_path)
    print(f"\nMetadata:")
    print(f"  Number of row groups: {parquet_file.num_row_groups}")
    print(f"  Number of rows: {parquet_file.metadata.num_rows:,}")
    print(f"  Schema: {parquet_file.schema_arrow}")

    # Read sample
    print(f"\nSample records (first {sample_size}):")
    table = parquet_file.read(columns=None)
    df = table.to_pandas()
    print(df.head(sample_size).to_string())

    # Test random access
    print(f"\nTesting random access (record at index 100):")
    if len(df) > 100:
        print(df.iloc[100].to_dict())


def main():
    parser = argparse.ArgumentParser(
        description='Convert ClueWeb22-B JSON.gz files to optimized Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all languages
  python convert_clueweb22_to_parquet.py \\
      --input /home/eh6/E128356/scratch/collections/ClueWeb22-B/txt \\
      --output clueweb22_all.parquet

  # Process only English documents
  python convert_clueweb22_to_parquet.py \\
      --input /home/eh6/E128356/scratch/collections/ClueWeb22-B/txt/en \\
      --output clueweb22_en.parquet

  # Use SNAPPY compression for faster reads
  python convert_clueweb22_to_parquet.py \\
      --input /home/eh6/E128356/scratch/collections/ClueWeb22-B/txt \\
      --output clueweb22_all.parquet \\
      --compression snappy

Compression options:
  - zstd (default): Best balance of size and speed (recommended)
  - snappy: Fastest decompression, larger files
  - gzip: Good compression, slower than zstd
  - lz4: Fast, but larger than zstd
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory to recursively search for .json.gz files (e.g., /path/to/ClueWeb22-B/txt or /path/to/ClueWeb22-B/txt/en)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output Parquet file path'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100000,
        help='Batch size for writing (affects row group size). Default: 100000'
    )

    parser.add_argument(
        '--compression',
        type=str,
        default='zstd',
        choices=['snappy', 'gzip', 'zstd', 'lz4'],
        help='Compression codec. Default: zstd (best for database-like access)'
    )

    parser.add_argument(
        '--compression-level',
        type=int,
        default=3,
        help='Compression level (1-22 for zstd, 1-9 for gzip). Default: 3'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the output Parquet file after creation'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Validate input
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find files
    print(f"Scanning for JSON.gz files in {input_dir}...")
    json_gz_files = find_json_gz_files(input_dir)

    if not json_gz_files:
        print(f"Error: No .json.gz files found!")
        return 1

    print(f"Found {len(json_gz_files)} .json.gz files")

    # Process files
    process_records_to_parquet(
        json_gz_files,
        output_path,
        batch_size=args.batch_size,
        compression=args.compression,
        compression_level=args.compression_level
    )

    # Verify if requested
    if args.verify:
        verify_parquet_file(output_path)

    return 0


if __name__ == '__main__':
    exit(main())
