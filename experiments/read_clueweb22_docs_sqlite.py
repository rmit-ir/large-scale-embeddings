#!/usr/bin/env python3
"""
Read ClueWeb22 documents from SQLite database using ID lookup.

For loading clueweb22-b documents
"""

import json
import sqlite3
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def decompress_and_parse(doc_id: str, json_data: bytes, use_compression: bool):
    """
    Decompress (if needed) and parse JSON data.
    This function is used for parallel processing.
    """
    if use_compression:
        import zstd
        json_str = zstd.decompress(json_data).decode('utf-8')
    else:
        json_str = json_data

    doc = json.loads(json_str)
    return doc_id, doc


def read_docs_by_ids(db_path: Path, query_ids: list[str], use_compression: bool = False, num_threads: int = None):
    """
    Read documents from SQLite database by ClueWeb22_IDs.

    Args:
        db_path: Path to SQLite database file
        query_ids: List of ClueWeb22_IDs to retrieve
        use_compression: Whether the json_data is compressed with zstd
        num_threads: Number of threads for parallel decompression (default: CPU count)

    Returns:
        List of document dictionaries (None for not found), preserving query_ids order
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Batch query with IN clause - much faster than individual queries
    placeholders = ','.join('?' * len(query_ids))
    query = f"SELECT ClueWeb22_ID, json_data FROM documents WHERE ClueWeb22_ID IN ({placeholders})"
    cursor.execute(query, query_ids)

    # Fetch all results
    results = cursor.fetchall()
    conn.close()

    # Parallel decompression and parsing
    result_dict = {}
    if use_compression and len(results) > 1:
        # Use ThreadPoolExecutor for parallel decompression
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(decompress_and_parse, doc_id, json_data, use_compression): doc_id
                for doc_id, json_data in results
            }
            # Process results as they complete
            for future in as_completed(futures):
                doc_id, doc = future.result()
                result_dict[doc_id] = doc
    else:
        # Single-threaded for small batches or uncompressed data
        for doc_id, json_data in results:
            _, doc = decompress_and_parse(doc_id, json_data, use_compression)
            result_dict[doc_id] = doc

    # Return docs in original query_ids order, None for not found
    return [result_dict.get(qid) for qid in query_ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read ClueWeb22 documents from SQLite database using ID lookup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up specific documents by ID (uncompressed database)
  python read_clueweb22_docs_sqlite.py \\
      --db_path clueweb22_en.db \\
      --query_ids clueweb22-en0000-00-00000 clueweb22-en0000-00-00001

  # Look up from compressed database
  python read_clueweb22_docs_sqlite.py \\
      --db_path clueweb22_en.db \\
      --query_ids clueweb22-en0000-00-00000 \\
      --use-compression
        """
    )
    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to SQLite database file")
    parser.add_argument("--query_ids", type=str, nargs='+', required=True,
                        help="ClueWeb22_IDs to look up")
    parser.add_argument("--use-compression", action='store_true',
                        help="Database uses zstd compression for json_data")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        exit(1)

    query_ids = args.query_ids
    print(f"Looking up {len(query_ids)} document(s)...")
    print("=" * 80)

    docs = read_docs_by_ids(db_path, query_ids, args.use_compression)

    print("\n" + "=" * 80)
    print("Documents")
    print("=" * 80)

    for qid, doc in zip(query_ids, docs):
        loc = f"{query_ids.index(qid) + 1}/{len(query_ids)}"
        print(f"\n[Document {loc}]")
        if doc is None:
            print(f"  ID: {qid} -> NOT FOUND")
        else:
            clean_text = doc.get('Clean-Text', '')
            word_count = len(clean_text.split()) if clean_text else 0
            print(f"  ID: {doc.get('ClueWeb22-ID', 'N/A')}")
            print(f"  URL: {doc.get('URL', 'N/A')}")
            print(f"  Language: {doc.get('Language', 'N/A')}")
            print(f"  Words: {word_count:,}")

    print("-" * 80)
    print(
        f"\nTotal documents retrieved: {sum(1 for d in docs if d is not None)}/{len(docs)}")
