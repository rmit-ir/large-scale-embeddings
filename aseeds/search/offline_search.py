"""
Offline search for indexed datasets.
"""

from pathlib import Path
from typing import Callable, Awaitable, Optional, AsyncIterator
import numpy as np


async def search(
    index_dir: Path,
    queries_file: Path,
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    top_k: int = 10,
    output_file: Optional[Path] = None,
):
    """
    Generic offline search program.
    
    This function performs batch search on an indexed dataset. It reads queries
    from a file, embeds them using the provided embedding function, and searches
    the DiskANN index for similar documents.
    
    Args:
        index_dir: Directory containing:
            - index/ (DiskANN index files)
            - config.json (metadata with embedding_dim, num_docs)
            - documents.db (SQLite database for document content)
        queries_file: File with queries. Format:
            - One query per line (text file)
            - Or JSON file with query objects
        embed_fn: Async function that takes a list of query strings and returns
                  embeddings as numpy array. Signature:
                  async def embed_fn(queries: list[str]) -> np.ndarray
        top_k: Number of results to return per query (default: 10)
        output_file: Optional path to write results in JSON format
        
    Returns:
        Async iterator of search results, where each result is:
        {
            "query": str,
            "results": [
                {
                    "doc_id": str,
                    "score": float,
                    "rank": int,
                    "document": dict  # Full document from SQLite
                },
                ...
            ]
        }
        
    The function handles:
    - Loading DiskANN index
    - Loading document ID mappings
    - Batch embedding of queries
    - Parallel search execution
    - Document retrieval from SQLite
    - Progress tracking
    """
    pass


def _load_index(index_dir: Path):
    """
    Load DiskANN index from directory.
    
    Args:
        index_dir: Directory containing index files
        
    Returns:
        Loaded index object
    """
    pass


def _load_docids(index_dir: Path) -> list[str]:
    """
    Load document ID mappings.
    
    Args:
        index_dir: Directory containing docids.pkl
        
    Returns:
        List of document IDs in index order
    """
    pass
