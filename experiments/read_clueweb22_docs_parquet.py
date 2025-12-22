import argparse
import pyarrow as pa
import pyarrow.parquet as pq


def load_id_index(index_path):
    """Load the index file that maps ClueWeb22_ID to row positions."""
    table = pq.read_table(index_path, memory_map=True)
    ids = table['ClueWeb22_ID'].to_numpy()
    rows = table['row_position'].to_numpy()
    return ids, rows


def build_id_to_row_dict(ids, rows):
    """Build a dictionary mapping ClueWeb22_ID to row position."""
    return dict(zip(ids.tolist(), rows.tolist()))


def lookup_rows_dict(id_to_row, query_ids):
    """Look up row positions for given ClueWeb22_IDs."""
    return [id_to_row.get(qid, -1) for qid in query_ids]


def read_docs_by_rows(db_table: pa.Table, row_positions):
    """
    Read documents from the main parquet file at specific row positions.

    Args:
        parquet_path: Path to the main parquet file
        row_positions: List of row positions to read

    Returns:
        List of document dictionaries
    """
    docs = []
    for pos in row_positions:
        if pos < 0 or pos >= len(db_table):
            docs.append(None)
        else:
            # Extract row as dictionary
            row_dict = {
                'URL': db_table['URL'][pos].as_py(),
                'URL_hash': db_table['URL_hash'][pos].as_py(),
                'Language': db_table['Language'][pos].as_py(),
                'ClueWeb22_ID': db_table['ClueWeb22_ID'][pos].as_py(),
                'Clean_Text': db_table['Clean_Text'][pos].as_py(),
            }
            docs.append(row_dict)

    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read ClueWeb22 documents from Parquet files using ID lookup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up specific documents by ID
  python read_clueweb22_docs_parquet.py \\
      --parquet_path clueweb22_en.parquet \\
      --query_ids clueweb22-en0000-00-00000 clueweb22-en0000-00-00001
        """
    )
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--query_ids", type=str, nargs='+', required=True)
    args = parser.parse_args()

    # Derive index path from main parquet path
    parquet_path = args.parquet_path
    index_path = parquet_path.replace('.parquet', '.index.parquet')
    if not parquet_path.endswith('.parquet'):
        raise ValueError("Parquet path must end with .parquet extension")

    print(f"Loading index from: {index_path}")
    ids, rows = load_id_index(index_path)
    id_to_row = build_id_to_row_dict(ids, rows)
    print(f"Index loaded: {len(ids):,} document IDs")

    query_ids = args.query_ids
    print(f"\nLooking up {len(query_ids)} document(s)...")
    print("=" * 80)

    result_rows = lookup_rows_dict(id_to_row, query_ids)
    # Display row positions
    for qid, row in zip(query_ids, result_rows):
        if row == -1:
            print(f"ID: {qid} -> NOT FOUND")
        else:
            print(f"ID: {qid} -> Row: {row}")

    print(f"Loading docs db from: {parquet_path}")
    db_table = pq.read_table(parquet_path, memory_map=True)
    print(f"Docs db loaded: {len(db_table):,} documents")

    print(f"\nLoading documents from docs db")
    docs = read_docs_by_rows(db_table, result_rows)
    print("\n" + "=" * 80)
    print("Documents")
    print("=" * 80)

    for qid, row_pos, doc in zip(query_ids, result_rows, docs):
        _loc = f"{query_ids.index(qid) + 1}/{len(query_ids)}"
        print(f"[Document {_loc}]", end="")
        if doc is None:
            print(f", {qid} NOT FOUND)")
        else:
            print(f", {len(doc['Clean_Text'].split())} words")
    print("-" * 80)

    print(
        f"\nTotal documents retrieved: {sum(1 for d in docs if d is not None)}/{len(docs)}")
