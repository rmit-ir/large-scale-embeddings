"""
Output writer for creating search-ready files.

Creates:
- SQLite database with full documents
- DiskANN binary format files (embeds.bin, docids.pkl)
- Config.json with metadata
"""

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from .types import Document


# TODO: use TypedDict instead
@dataclass
class OutputConfig:
    """Configuration for output writing."""
    sqlite_compression: bool = True
    compression_level: int = 5
    batch_size: int = 1000
    sqlite_cache_size_mb: int = 20000


async def output_to_idx(
    output_dir: Path,
    documents: AsyncIterator[Document],
    embeddings: AsyncIterator[np.ndarray],
    config: Optional[OutputConfig] = None,
):
    """
    Write documents and embeddings to search-ready files.
    
    This function processes documents and embeddings asynchronously and writes
    them to four outputs:
    1. SQLite database (documents.db) with full documents
    2. DiskANN binary file (embeds.bin) with embeddings
    3. Document ID mappings (docids.pkl)
    4. Metadata config (config.json)
    
    Args:
        output_dir: Directory to write output files
        documents: Async iterator of Document objects
        embeddings: Async iterator of numpy arrays (embeddings)
        config: Output configuration options
        
    The function handles:
    - SQLite database creation with optimized settings
    - Parallel compression if enabled (zstd)
    - Streaming write to minimize memory usage
    - Progress tracking and error handling
    
    Output files:
        output_dir/documents.db: SQLite database with documents
        output_dir/embeds.bin: DiskANN binary format embeddings
        output_dir/docids.pkl: Pickled list of document IDs
        output_dir/config.json: Metadata (num_docs, embedding_dim, etc.)
    """
    pass


async def zip_async(*iterables):
    """
    Zip async iterators together.
    
    Args:
        *iterables: Async iterators to zip together
        
    Yields:
        Tuples containing items from each iterator
    """
    pass


def _write_embeddings_binary(output_path: Path, embeddings: list[np.ndarray]):
    """
    Write embeddings to DiskANN binary format.
    
    Args:
        output_path: Path to write binary file
        embeddings: List of embedding arrays
        
    The binary format is:
    - 4 bytes: number of dimensions (int32, little-endian)
    - For each vector:
        - For each dimension: 4 bytes (float32, little-endian)
    """
    pass
