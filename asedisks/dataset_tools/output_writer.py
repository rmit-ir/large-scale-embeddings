"""
Output writer for creating search-ready files.

Creates:
- SQLite database with full documents
- DiskANN binary format files (embeds.bin, docids.pkl)
- Config.json with metadata
"""

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Callable, Awaitable, Union
import numpy as np
import sqlite3
import pickle
import json

from .types import DataRecord
from asedisks.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OutputConfig:
    """Configuration for output writing."""

    sqlite_compression: bool = True
    compression_level: int = 5
    batch_size: int = 50
    sqlite_cache_size_mb: int = 2000


async def output_to_idx(
    output_dir: Path,
    records: Union[AsyncIterator[list[DataRecord]], Iterator[list[DataRecord]]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    config: Optional[OutputConfig] = None,
):
    """
    Write records and embeddings to search-ready files.

    This function processes records asynchronously, calls the embedding function
    for each batch, and writes outputs:
    1. SQLite database (documents.db) with full documents
    2. DiskANN binary file (embeds.bin) with embeddings
    3. Document ID mappings (docids.pkl)
    4. Metadata config (config.json)

    Args:
        output_dir: Directory to write output files
        records: Async or sync iterator of DataRecord batches
        embed_fn: Async function that embeds a batch of texts.
                  Signature: async def embed_fn(texts: list[str]) -> np.ndarray, shape (len(texts), 1024)
        config: Output configuration options

    The function handles:
    - SQLite database creation with optimized settings
    - Streaming write to minimize memory usage
    - Progress tracking

    Output files:
        output_dir/documents.db: SQLite database with documents
        output_dir/embeds.bin: DiskANN binary format embeddings
        output_dir/docids.pkl: Pickled list of document IDs
        output_dir/config.json: Metadata (num_docs, embedding_dim, etc.)
    """
    config = config or OutputConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize outputs
    embedding_dim: Optional[int] = None
    num_records = 0

    # File paths
    embeds_path = output_dir / "embeds.bin"
    docids_path = output_dir / "docids.pkl"
    db_path = output_dir / "documents.db"
    config_path = output_dir / "config.json"

    doc_ids: list[str] = []

    # Initialize SQLite
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA cache_size=-{config.sqlite_cache_size_mb * 1024}")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL
        )
    """
    )
    conn.commit()

    # Open binary file for streaming writes
    pbar = None
    try:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(
            total=None,
            unit=" records",
            desc="Embedding",
            leave=True,
            position=1,
            dynamic_ncols=True,
        )
    except Exception:
        pbar = None
    with open(embeds_path, "wb") as embeds_file:
        # Write placeholder header (will update later)
        embeds_file.write(np.uint32(0).tobytes())  # num_vectors placeholder
        embeds_file.write(np.uint32(0).tobytes())  # dimensions placeholder

        async for batch in _ensure_async(records):
            if not batch:
                continue
            embedding_dim = await _process_batch(
                batch, embed_fn, embeds_file, conn, doc_ids, embedding_dim
            )
            num_records += len(batch)
            if pbar is not None:
                pbar.update(len(batch))
            else:
                logger.info("Processed %s records...", f"{num_records:,}")

        # Update header with actual counts
        embeds_file.seek(0)
        embeds_file.write(np.uint32(num_records).tobytes())
        embeds_file.write(np.uint32(embedding_dim or 0).tobytes())

    # Save docids
    with open(docids_path, "wb") as f:
        pickle.dump(doc_ids, f)

    # Commit and close SQLite
    conn.commit()
    conn.close()
    if pbar is not None:
        pbar.close()

    # Write config
    with open(config_path, "w") as f:
        json.dump(
            {
                "num_docs": num_records,
                "embedding_dim": embedding_dim,
            },
            f,
            indent=2,
        )

    logger.info("Output written to %s", output_dir)
    logger.info("  - %s documents", f"{num_records:,}")
    logger.info("  - %s dimensions", embedding_dim)


async def _process_batch(
    batch: list[DataRecord],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    embeds_file,
    conn: sqlite3.Connection,
    doc_ids: list[str],
    embedding_dim: Optional[int],
) -> int:
    """
    Process a single batch of records.

    Args:
        batch: List of DataRecord objects
        embed_fn: Embedding function
        embeds_file: Open binary file handle
        conn: SQLite connection
        doc_ids: List to append document IDs
        embedding_dim: Current embedding dimension (or None if not yet set)

    Returns:
        Embedding dimension
    """
    # Get embeddings
    texts = [r["content"] for r in batch]
    embeddings = await embed_fn(texts)

    # Validate/set embedding dimension
    if embedding_dim is None:
        embedding_dim = embeddings.shape[1]
    elif embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {embedding_dim}, "
            f"got {embeddings.shape[1]}"
        )

    # Write embeddings to binary (DiskANN format)
    embeds_file.write(embeddings.astype(np.float32).tobytes())

    # Write to SQLite
    cursor = conn.cursor()
    for record in batch:
        doc_ids.append(record["id"])
        json_data = json.dumps(record["metadata"])
        cursor.execute(
            "INSERT OR REPLACE INTO documents (doc_id, json_data) VALUES (?, ?)",
            (record["id"], json_data),
        )
    conn.commit()

    return embedding_dim


async def _ensure_async(
    iterable: Union[AsyncIterator[list[DataRecord]], Iterator[list[DataRecord]]]
) -> AsyncIterator[list[DataRecord]]:
    """
    Convert sync iterator to async if needed.

    Args:
        iterable: Sync or async iterator

    Yields:
        DataRecord objects
    """
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:  # type: ignore
            yield item
    else:
        for item in iterable:  # type: ignore
            yield item
