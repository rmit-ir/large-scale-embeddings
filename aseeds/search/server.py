"""
Search router server for indexed datasets.
"""

from pathlib import Path
from typing import Callable, Awaitable, Optional


async def run_search_router(
    index_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8001,
    embed_fn: Optional[Callable[[str], Awaitable]] = None,
):
    """
    Generic search router server.
    
    This function runs a FastAPI-based search router that provides REST API
    endpoints for searching an indexed dataset. It can either run an inline
    embedding node (using embed_fn) or connect to an external embedding service.
    
    Args:
        index_dir: Directory containing:
            - index/ (DiskANN index files)
            - config.json (metadata with embedding_dim, num_docs)
            - documents.db (SQLite database for document content)
            - docids.pkl (document ID mappings)
        host: Server host address (default: "0.0.0.0")
        port: Server port (default: 8001)
        embed_fn: Optional async function that embeds a single query string.
                  If provided, runs embed node inline. Signature:
                  async def embed_fn(query: str) -> np.ndarray
                  If None, expects external embed node via config.json
        
    API Endpoints:
        POST /search
            Request body:
                {
                    "query": str,
                    "top_k": int (default: 10)
                }
            Response:
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
        
        GET /health
            Response: {"status": "ok"}
    
    The function handles:
    - Loading DiskANN index on startup
    - Loading SQLite database connection
    - Running FastAPI server with uvicorn
    - Embedding (inline or via external service)
    - Search request processing
    - Document retrieval
    - Error handling and logging
    """
    pass


def _create_fastapi_app(index_dir: Path, embed_fn: Optional[Callable] = None):
    """
    Create FastAPI application for search router.
    
    Args:
        index_dir: Directory containing index files
        embed_fn: Optional embedding function
        
    Returns:
        FastAPI application instance
    """
    pass
