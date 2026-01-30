"""
Offline search for indexed datasets.
"""

from pathlib import Path
from typing import Callable, Awaitable, Optional, Union
import os
import numpy as np
import pickle
import json
import sqlite3

from asedisks.logging_utils import get_logger

logger = get_logger(__name__)

async def search(
    index_dir: Path,
    queries: Union[list[str], Path],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    top_k: int = 10,
    complexity: int = 100,
    include_documents: bool = False,
    output_file: Optional[Path] = None,
    num_threads: Optional[int] = None,
    num_nodes_to_cache: Optional[int] = None,
    cache_mechanism: Optional[int] = None,
    distance_metric: Optional[str] = None,
    index_prefix: Optional[str] = None,
    beam_width: Optional[int] = None,
) -> list[dict]:
    """
    Perform offline search on an indexed dataset.

    This function performs batch search on an indexed dataset. It reads queries,
    embeds them using the provided embedding function, and searches the DiskANN
    index for similar documents.

    Args:
        index_dir: Directory containing:
            - index/ (DiskANN index files)
            - config.json (metadata with embedding_dim, num_docs)
            - documents.db (SQLite database for document content)
            - docids.pkl (document ID mappings)
        queries: List of query strings, or Path to file with queries
            - Text file: one query per line
            - JSON file: list of query strings or objects with "query" key
        embed_fn: Async function that takes a list of query strings and returns
                  embeddings as numpy array. Signature:
                  async def embed_fn(queries: list[str]) -> np.ndarray
        top_k: Number of results to return per query (default: 10)
        complexity: DiskANN search complexity (default: 100)
        include_documents: Whether to include full document content (default: False)
        output_file: Optional path to write results in JSON format

    Returns:
        List of search results, where each result is:
        {
            "query": str,
            "results": [
                {
                    "doc_id": str,
                    "score": float,
                    "rank": int,
                    "document": dict  # Only if include_documents=True
                },
                ...
            ]
        }

    The function handles:
    - Loading DiskANN index
    - Loading document ID mappings
    - Batch embedding of queries
    - Search execution
    - Optional document retrieval from SQLite
    """
    index_dir = Path(index_dir)

    # Load config
    config = _load_config(index_dir / "config.json")
    logger.info(
        "Index: %s documents, %s dims",
        f"{config['num_docs']:,}",
        config["embedding_dim"],
    )

    # Load index and docids
    index = _load_index(
        index_dir / "index",
        config["embedding_dim"],
        num_threads=num_threads,
        num_nodes_to_cache=num_nodes_to_cache,
        cache_mechanism=cache_mechanism,
        distance_metric=distance_metric,
        index_prefix=index_prefix,
    )
    docids = _load_docids(index_dir / "docids.pkl")

    # Load documents database if needed
    docs_db = None
    if include_documents:
        docs_db = sqlite3.connect(str(index_dir / "documents.db"))

    # Load queries
    if isinstance(queries, Path):
        queries = _load_queries(queries)

    logger.info("Searching %s queries...", len(queries))

    # Embed queries
    query_embeddings = await embed_fn(queries)

    # Search
    results = []
    if beam_width is None:
        beam_width = _env_int("DISKANN_BEAM_WIDTH", 1)
    if num_threads is None:
        num_threads = _env_int("DISKANN_NUM_THREADS", 4)

    for i, (query, q_emb) in enumerate(zip(queries, query_embeddings)):
        _queries = q_emb.reshape(1, -1).astype(np.float32)
        indices, distances = index.batch_search(
            queries=_queries,
            k_neighbors=top_k,
            complexity=complexity,
            num_threads=num_threads,
            beam_width=beam_width,
        )

        # Build result
        result = {
            "query": query,
            "results": [],
        }

        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(docids):
                continue  # Skip invalid indices

            item = {
                "doc_id": docids[idx],
                "score": float(dist),
                "rank": rank,
            }

            if include_documents and docs_db:
                doc = _load_document(docs_db, docids[idx])
                if doc:
                    item["document"] = doc

            result["results"].append(item)

        results.append(result)

    # Close database
    if docs_db:
        docs_db.close()

    # Write output file if specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", output_file)

    return results


def _load_index(
    index_dir: Path,
    embedding_dim: int,
    num_threads: Optional[int] = None,
    num_nodes_to_cache: Optional[int] = None,
    cache_mechanism: Optional[int] = None,
    distance_metric: Optional[str] = None,
    index_prefix: Optional[str] = None,
):
    """
    Load DiskANN index from directory.

    Args:
        index_dir: Directory containing index files
        embedding_dim: Embedding dimension

    Returns:
        Loaded DiskANN index object
    """
    try:
        import diskannpy
    except ImportError:
        raise ImportError(
            "diskannpy not installed. Install with: pip install diskannpy"
        )

    if num_threads is None:
        num_threads = _env_int("DISKANN_NUM_THREADS", 4)
    if num_nodes_to_cache is None:
        num_nodes_to_cache = _env_int("DISKANN_NUM_NODES_TO_CACHE", 10000)
    if cache_mechanism is None:
        cache_mechanism = _env_int("DISKANN_CACHE_MECHANISM", 1)
    if distance_metric is None:
        distance_metric = os.environ.get("DISKANN_DISTANCE_METRIC", "mips")
    if index_prefix is None:
        index_prefix = os.environ.get("DISKANN_INDEX_PREFIX", "index_")

    return diskannpy.StaticDiskIndex(
        index_directory=str(index_dir),
        num_threads=num_threads,
        num_nodes_to_cache=num_nodes_to_cache,
        cache_mechanism=cache_mechanism,
        distance_metric=distance_metric,
        vector_dtype=np.float32,
        dimensions=embedding_dim,
        index_prefix=index_prefix,
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_docids(path: Path) -> list[str]:
    """
    Load document ID mappings.

    Args:
        path: Path to docids.pkl file

    Returns:
        List of document IDs in index order
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_config(path: Path) -> dict:
    """
    Load configuration file.

    Args:
        path: Path to config.json

    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


def _load_queries(path: Path) -> list[str]:
    """
    Load queries from file.

    Supports:
    - Text file: one query per line
    - JSON file: list of strings or objects with "query" key

    Args:
        path: Path to queries file

    Returns:
        List of query strings
    """
    path = Path(path)

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                return [item.get("query", item.get("text", "")) for item in data]
            return data
        return [data]

    # Text file - one query per line
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _load_document(conn: sqlite3.Connection, doc_id: str) -> Optional[dict]:
    """
    Load a document from SQLite database.

    Args:
        conn: SQLite connection
        doc_id: Document ID

    Returns:
        Document dictionary or None if not found
    """
    cursor = conn.cursor()
    cursor.execute("SELECT json_data FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()

    if row:
        return json.loads(row[0])
    return None
