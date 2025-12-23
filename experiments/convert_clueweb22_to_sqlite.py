#!/usr/bin/env python3
"""
Convert ClueWeb22-B JSON.gz files to SQLite database format.
Optimized for fast random access by ClueWeb22_ID with optional compression.

python experiments/convert_clueweb22_to_sqlite.py --input /home/eh6/E128356/scratch/collections/ClueWeb22-B/txt/en/ --output data/clueweb-docs-db/clueweb22b_en.db --use-compression --compression-level 5

(minicpmembed) [e128356@sctsresap21 large-scale-embeddings]$ python experiments/convert_clueweb22_to_sqlite.py --input /home/eh6/E128356/scratch/collections/ClueWeb22-B/txt/en/ --output data/clueweb-docs-db/clueweb
22b_en.db --use-compression --compression-level 5 --num-threads 128 --verify

Verifying database: data/clueweb-docs-db/clueweb22b_en.db

Total records: 87,208,655

Sample records (first 5):

[1] ID: clueweb22-en0000-01-01649
    URL: https://www.amazon.com/litter-locker-ii/s?k=litter+locker+ii

    Language: en
    Words: 1,981
    JSON size: 12,066 bytes
    Compressed size: 4,166 bytes
    Compression ratio: 2.90x

[2] ID: clueweb22-en0000-01-03142
    URL: https://www.buildzoom.com/contractor/f-j-drywall-llc

    Language: en
    Words: 362
    JSON size: 2,531 bytes
    Compressed size: 1,272 bytes
    Compression ratio: 1.99x

[3] ID: clueweb22-en0000-00-13593
    URL: https://www.cnywrestling.com/iv/teams/Oxford/

    Language: en
    Words: 897
    JSON size: 5,530 bytes
    Compressed size: 2,460 bytes
    Compression ratio: 2.25x

[4] ID: clueweb22-en0000-00-13710
    URL: https://www.reddit.com/r/tiktokthots/comments/okq43m/uluvmi/

    Language: en
    Words: 462
    JSON size: 2,892 bytes
    Compressed size: 1,444 bytes
    Compression ratio: 2.00x

[5] ID: clueweb22-en0000-01-01648
    URL: https://www.dess-usa.com/biohorizons-hex-compatible-intraoral-scan-body/

    Language: en
    Words: 216
    JSON size: 1,536 bytes
    Compressed size: 561 bytes
    Compression ratio: 2.74x

Testing random access with ID: clueweb22-en0000-00-00100
✓ Successfully retrieved document
  URL: https://www.stepheneinhorn.co.uk/bespoke-rings.asp
"""

import json
import gzip
import sqlite3
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def find_json_gz_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise ValueError(f"Directory not found: {root_dir}")

    json_gz_files = sorted(root_dir.rglob("*.json.gz"))

    return json_gz_files


def read_json_gz_lines(file_path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parse JSON at {file_path}:{line_num} - {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


def serialize_and_compress(id: str, record: Any, use_compression: bool, compression_level: int):
    """
    Serialize record to JSON and optionally compress.
    This function is used for parallel processing.
    """
    if not id:
        return None

    # Serialize entire record to JSON
    json_str = json.dumps(record, ensure_ascii=False)

    # Optionally compress
    if use_compression:
        import zstd
        json_data = zstd.compress(
            json_str.encode('utf-8'),
            compression_level
        )
    else:
        json_data = json_str

    return (id, json_data)


def create_database(db_path: Path, cache_size_mb: int = 20000) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create table with simple schema: ID + JSON data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            ClueWeb22_ID TEXT PRIMARY KEY,
            json_data BLOB NOT NULL
        )
    """)

    # Optimize SQLite settings for bulk insert and later read performance
    cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still safe
    # Negative = KB (e.g., -20000*1024 = 20GB)
    cursor.execute(f"PRAGMA cache_size=-{cache_size_mb * 1024}")
    cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
    # 30GB memory map for large datasets
    cursor.execute("PRAGMA mmap_size=30000000000")

    conn.commit()
    return conn


def process_records_to_sqlite(
    json_gz_files: list[Path],
    output_path: Path,
    use_compression: bool = False,
    compression_level: int = 3,
    batch_size: int = 50000,
    num_threads: Optional[int] = None,
):
    """
    Process JSON.gz files and write to SQLite database with parallel compression.

    Args:
        json_gz_files: List of JSON.gz files to process
        output_path: Output SQLite database file path
        use_compression: Whether to compress json_data with zstd
        compression_level: Zstd compression level (1-22, higher = smaller but slower)
        batch_size: Number of records to batch before committing
        num_threads: Number of threads for parallel processing (default: CPU count)
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    # Create database
    print(f"Creating database: {output_path}")
    print(f"Using {num_threads} threads for parallel compression")
    conn = create_database(output_path)
    cursor = conn.cursor()

    batch = []
    total_records = 0
    skipped_records = 0
    records_buffer = []

    try:
        with tqdm(total=len(json_gz_files), desc="Processing files", unit="file") as pbar:
            for file_path in json_gz_files:
                for record in read_json_gz_lines(file_path):
                    clueweb_id = record.get('ClueWeb22-ID', '')
                    if not clueweb_id:
                        skipped_records += 1
                        continue

                    records_buffer.append((clueweb_id, record))

                    # Process buffer in parallel when it reaches a good size
                    if len(records_buffer) >= batch_size:
                        # Parallel serialization and compression
                        with ThreadPoolExecutor(max_workers=num_threads) as executor:
                            futures = {
                                executor.submit(serialize_and_compress, clueweb_id, record, use_compression, compression_level): clueweb_id
                                for clueweb_id, record in records_buffer
                            }

                            for future in as_completed(futures):
                                result = future.result()
                                if result:
                                    batch.append(result)
                                    total_records += 1

                        # Insert batch
                        cursor.executemany(
                            "INSERT OR REPLACE INTO documents (ClueWeb22_ID, json_data) VALUES (?, ?)",
                            batch
                        )
                        conn.commit()
                        tqdm.write(f"Processed {total_records:,} records...")

                        # Clear buffers
                        records_buffer = []
                        batch = []

                pbar.update(1)

        # Process remaining records
        if records_buffer:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {
                    executor.submit(serialize_and_compress, clueweb_id, record, use_compression, compression_level): clueweb_id
                    for clueweb_id, record in records_buffer
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch.append(result)
                        total_records += 1

            cursor.executemany(
                "INSERT OR REPLACE INTO documents (ClueWeb22_ID, json_data) VALUES (?, ?)",
                batch
            )
            conn.commit()
            print(f"Inserted final batch. Total records: {total_records:,}")

    finally:
        conn.close()

    # Print summary
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Conversion complete!")
        print(f"Total records inserted: {total_records:,}")
        if skipped_records > 0:
            print(f"Skipped records (missing ID): {skipped_records:,}")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
        print(
            f"Compression: {'zstd (level ' + str(compression_level) + ')' if use_compression else 'none'}")
        print(f"{'='*60}")

    return total_records


def verify_database(db_path: Path, use_compression: bool, sample_size: int = 5):
    """
    Verify the database and show sample records.

    Args:
        db_path: Path to database file
        use_compression: Whether data is compressed
        sample_size: Number of sample records to display
    """
    print(f"\nVerifying database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Count records
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_records = cursor.fetchone()[0]
    print(f"\nTotal records: {total_records:,}")

    # Show sample records
    print(f"\nSample records (first {sample_size}):")
    cursor.execute(
        f"SELECT ClueWeb22_ID, json_data FROM documents LIMIT {sample_size}")

    for i, (doc_id, json_data) in enumerate(cursor.fetchall(), 1):
        # Decompress if needed
        if use_compression:
            import zstd
            json_str = zstd.decompress(json_data).decode('utf-8')
        else:
            json_str = json_data

        doc = json.loads(json_str)
        clean_text = doc.get('Clean-Text', '')
        word_count = len(clean_text.split()) if clean_text else 0

        print(f"\n[{i}] ID: {doc_id}")
        print(f"    URL: {doc.get('URL', 'N/A')}")
        print(f"    Language: {doc.get('Language', 'N/A')}")
        print(f"    Words: {word_count:,}")
        print(f"    JSON size: {len(json_str):,} bytes")
        if use_compression:
            print(f"    Compressed size: {len(json_data):,} bytes")
            print(
                f"    Compression ratio: {len(json_str)/len(json_data):.2f}x")

    # Test random access
    cursor.execute("SELECT ClueWeb22_ID FROM documents LIMIT 1 OFFSET 100")
    result = cursor.fetchone()
    if result:
        test_id = result[0]
        print(f"\nTesting random access with ID: {test_id}")

        cursor.execute(
            "SELECT json_data FROM documents WHERE ClueWeb22_ID = ?", (test_id,))
        json_data = cursor.fetchone()[0]

        if use_compression:
            json_str = zstd.decompress(json_data).decode('utf-8')
        else:
            json_str = json_data

        doc = json.loads(json_str)
        print(f"✓ Successfully retrieved document")
        print(f"  URL: {doc.get('URL', 'N/A')}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Convert ClueWeb22-B JSON.gz files to SQLite database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert without compression
  python convert_clueweb22_to_sqlite.py \\
      --input /path/to/ClueWeb22-B/txt/en \\
      --output clueweb22_en.db

  # Convert with zstd compression (recommended for production)
  python convert_clueweb22_to_sqlite.py \\
      --input /path/to/ClueWeb22-B/txt/en \\
      --output clueweb22_en.db \\
      --use-compression \\
      --compression-level 5

  # Verify after conversion
  python convert_clueweb22_to_sqlite.py \\
      --input /path/to/ClueWeb22-B/txt/en \\
      --output clueweb22_en.db \\
      --use-compression \\
      --verify

Compression levels:
  1-3:  Fast compression, larger files (good for testing)
  3-6:  Balanced (recommended for production)
  7-22: High compression, slower (diminishing returns)
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory to recursively search for .json.gz files'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output SQLite database file path (.db)'
    )

    parser.add_argument(
        '--use-compression',
        action='store_true',
        help='Enable zstd compression for json_data field (requires: pip install zstd)'
    )

    parser.add_argument(
        '--compression-level',
        type=int,
        default=3,
        help='Zstd compression level (1-22). Default: 3 (good balance of speed/size)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='Number of records per commit. Default: 50000'
    )

    parser.add_argument(
        '--num-threads',
        type=int,
        default=None,
        help='Number of threads for parallel compression (default: CPU count)'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify an existing database and exit (skip conversion), only uses --output and --use-compression'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input)
    output_path = Path(args.output)

    # If verify mode, just verify and exit
    if args.verify:
        if not output_path.exists():
            print(f"Error: Database file does not exist: {output_path}")
            return 1
        verify_database(output_path, args.use_compression)
        return 0

    # Validate compression level
    if args.compression_level < 1 or args.compression_level > 22:
        print("Error: compression-level must be between 1 and 22")
        return 1

    # Validate input
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Check if output already exists
    if output_path.exists():
        response = input(
            f"Warning: {output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        output_path.unlink()

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
    process_records_to_sqlite(
        json_gz_files,
        output_path,
        use_compression=args.use_compression,
        compression_level=args.compression_level,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )

    return 0


if __name__ == '__main__':
    exit(main())
