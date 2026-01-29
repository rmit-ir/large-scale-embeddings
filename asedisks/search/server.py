"""
Search router server for indexed datasets.
"""

from pathlib import Path
from typing import Callable, Awaitable, Optional
import numpy as np
import pickle
import json
import sqlite3


async def run_search_router(
    index_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8001,
    embed_fn: Optional[Callable[[str], Awaitable[np.ndarray]]] = None,
):
    """
    Run FastAPI-based search router server.

    This function runs a FastAPI-based search router that provides REST API
    endpoints for searching an indexed dataset. It uses the provided embedding
    function to embed queries inline.

    Args:
        index_dir: Directory containing:
            - index/ (DiskANN index files)
            - config.json (metadata with embedding_dim, num_docs)
            - documents.db (SQLite database for document content)
            - docids.pkl (document ID mappings)
        host: Server host address (default: "0.0.0.0")
        port: Server port (default: 8001)
        embed_fn: Async function that embeds a single query string.
                  Signature: async def embed_fn(query: str) -> np.ndarray

    API Endpoints:
        POST /search
            Request body:
                {
                    "query": str,
                    "top_k": int (default: 10),
                    "complexity": int (default: 100),
                    "include_document": bool (default: false)
                }
            Response:
                {
                    "query": str,
                    "results": [
                        {
                            "doc_id": str,
                            "score": float,
                            "rank": int,
                            "document": dict  # Only if include_document=True
                        },
                        ...
                    ]
                }

        GET /health
            Response: {"status": "ok", "num_docs": int, "embedding_dim": int}

    The function handles:
    - Loading DiskANN index on startup
    - Loading SQLite database connection
    - Running FastAPI server with uvicorn
    - Embedding queries via embed_fn
    - Search request processing
    - Document retrieval
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Install with: pip install uvicorn")

    app = _create_fastapi_app(index_dir, embed_fn)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


def _create_fastapi_app(index_dir: Path, embed_fn: Optional[Callable] = None):
    """
    Create FastAPI application for search router.

    Args:
        index_dir: Directory containing index files
        embed_fn: Optional embedding function

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI and pydantic not installed. "
            "Install with: pip install fastapi pydantic"
        )

    index_dir = Path(index_dir)

    app = FastAPI(
        title="ASEDISKS Search",
        description="Search API for indexed datasets",
        version="1.0.0",
    )

    # State stored in app
    app.state.index = None
    app.state.docids = None
    app.state.config = None
    app.state.docs_db = None
    app.state.embed_fn = embed_fn

    @app.on_event("startup")
    async def startup():
        """Load index and resources on startup."""
        print(f"Loading index from {index_dir}...")

        # Load config
        with open(index_dir / "config.json", "r") as f:
            app.state.config = json.load(f)

        print(
            f"  Config: {app.state.config['num_docs']:,} docs, "
            f"{app.state.config['embedding_dim']} dims"
        )

        # Load DiskANN index
        try:
            import diskannpy
        except ImportError:
            raise ImportError(
                "diskannpy not installed. Install with: pip install diskannpy"
            )

        app.state.index = diskannpy.StaticDiskIndex(
            index_directory=str(index_dir / "index"),
            num_threads=4,
            num_nodes_to_cache=10000,
            cache_mechanism=1,
            distance_metric="mips",
            vector_dtype=np.float32,
            dimensions=app.state.config["embedding_dim"],
            index_prefix="index_",
        )
        print("  Index loaded")

        # Load docids
        with open(index_dir / "docids.pkl", "rb") as f:
            app.state.docids = pickle.load(f)
        print(f"  Docids loaded: {len(app.state.docids):,}")

        # Open documents database
        app.state.docs_db = sqlite3.connect(str(index_dir / "documents.db"))
        print("  Documents database connected")

        print("Server ready!")

    @app.on_event("shutdown")
    async def shutdown():
        """Close resources on shutdown."""
        if app.state.docs_db:
            app.state.docs_db.close()

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 10
        complexity: int = 100
        include_document: bool = False

    class SearchResult(BaseModel):
        doc_id: str
        score: float
        rank: int
        document: Optional[dict] = None

    class SearchResponse(BaseModel):
        query: str
        results: list[SearchResult]

    class HealthResponse(BaseModel):
        status: str
        num_docs: int
        embedding_dim: int

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """Search the index."""
        if app.state.embed_fn is None:
            raise HTTPException(
                status_code=500,
                detail="No embedding function configured"
            )

        # Embed query
        q_emb = await app.state.embed_fn(request.query)
        q_emb = np.array(q_emb, dtype=np.float32).reshape(1, -1)

        # Search
        indices, distances = app.state.index.batch_search(
            q_emb,
            request.top_k,
            request.complexity,
        )

        # Build results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(app.state.docids):
                continue

            result = SearchResult(
                doc_id=app.state.docids[idx],
                score=float(dist),
                rank=rank,
            )

            if request.include_document:
                cursor = app.state.docs_db.cursor()
                cursor.execute(
                    "SELECT json_data FROM documents WHERE doc_id = ?",
                    (app.state.docids[idx],)
                )
                row = cursor.fetchone()
                if row:
                    result.document = json.loads(row[0])

            results.append(result)

        return SearchResponse(query=request.query, results=results)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            num_docs=app.state.config["num_docs"],
            embedding_dim=app.state.config["embedding_dim"],
        )

    return app
