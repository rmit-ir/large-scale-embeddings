"""
Search router server for indexed datasets.
"""

from pathlib import Path
from typing import Callable, Awaitable, Optional
import os
import numpy as np
import pickle
import json
import sqlite3

from asedisks.logging_utils import get_logger

logger = get_logger(__name__)

async def run_search_router(
    index_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8001,
    embed_fn: Optional[Callable[[list[str]], Awaitable[np.ndarray]]] = None,
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
        embed_fn: Async function that embeds a batch of query strings.
                  Signature: async def embed_fn(queries: list[str]) -> np.ndarray

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


def _create_fastapi_app(
    index_dir: Path,
    embed_fn: Optional[Callable[[list[str]], Awaitable[np.ndarray]]] = None,
):
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
        logger.info("Loading index from %s...", index_dir)

        # Load config
        with open(index_dir / "config.json", "r") as f:
            app.state.config = json.load(f)

        logger.info(
            "  Config: %s docs, %s dims",
            f"{app.state.config['num_docs']:,}",
            app.state.config["embedding_dim"],
        )

        # Load DiskANN index
        try:
            import diskannpy
        except ImportError:
            raise ImportError(
                "diskannpy not installed. Install with: pip install diskannpy"
            )

        app.state.num_threads = _env_int("DISKANN_NUM_THREADS", 4)
        app.state.num_nodes_to_cache = _env_int("DISKANN_NUM_NODES_TO_CACHE", 10000)
        app.state.cache_mechanism = _env_int("DISKANN_CACHE_MECHANISM", 1)
        app.state.distance_metric = os.environ.get("DISKANN_DISTANCE_METRIC", "mips")
        app.state.index_prefix = os.environ.get("DISKANN_INDEX_PREFIX", "index_")
        app.state.beam_width = _env_int("DISKANN_BEAM_WIDTH", 1)

        app.state.index = diskannpy.StaticDiskIndex(
            index_directory=str(index_dir / "index"),
            num_threads=app.state.num_threads,
            num_nodes_to_cache=app.state.num_nodes_to_cache,
            cache_mechanism=app.state.cache_mechanism,
            distance_metric=app.state.distance_metric,
            vector_dtype=np.float32,
            dimensions=app.state.config["embedding_dim"],
            index_prefix=app.state.index_prefix,
        )
        logger.info("  Index loaded")

        # Load docids
        with open(index_dir / "docids.pkl", "rb") as f:
            app.state.docids = pickle.load(f)
        logger.info("  Docids loaded: %s", f"{len(app.state.docids):,}")

        # Open documents database
        app.state.docs_db = sqlite3.connect(str(index_dir / "documents.db"))
        logger.info("  Documents database connected")

        logger.info("Server ready!")

    @app.on_event("shutdown")
    async def shutdown():
        """Close resources on shutdown."""
        if app.state.docs_db:
            app.state.docs_db.close()

    class SearchRequest(BaseModel):
        query: Optional[str] = None
        queries: Optional[list[str]] = None
        top_k: int = 10
        complexity: int = 100
        include_document: bool = False
        beam_width: Optional[int] = None
        num_threads: Optional[int] = None

    class SearchResult(BaseModel):
        doc_id: str
        score: float
        rank: int
        document: Optional[dict] = None

    class SearchQueryResults(BaseModel):
        query: str
        results: list[SearchResult]

    class SearchResponse(BaseModel):
        query: str
        results: list[SearchResult]
        queries_results: list[SearchQueryResults] = []

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

        queries: list[str] = []
        if request.queries:
            queries = [q for q in request.queries if q]
        elif request.query:
            queries = [request.query]

        if not queries:
            raise HTTPException(
                status_code=400,
                detail="Provide either query or queries.",
            )

        embeddings = await app.state.embed_fn(queries)
        _queries = np.array(embeddings, dtype=np.float32)
        if _queries.shape[0] != len(queries):
            raise HTTPException(
                status_code=500,
                detail="embed_fn returned unexpected shape.",
            )

        k_neighbors = request.top_k
        num_queries = _queries.shape[0]
        complexity = request.complexity
        beam_width = (
            request.beam_width
            if request.beam_width is not None
            else app.state.beam_width
        )
        num_threads = (
            request.num_threads
            if request.num_threads is not None
            else app.state.num_threads
        )

        neighbors, distances = app.state.index.batch_search(
            queries=_queries,
            k_neighbors=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
            beam_width=beam_width,
        )

        queries_results: list[SearchQueryResults] = []
        all_doc_ids: set[str] = set()
        for q_idx, query in enumerate(queries):
            results = []
            for rank, (idx, dist) in enumerate(zip(neighbors[q_idx], distances[q_idx])):
                if idx >= len(app.state.docids):
                    continue
                doc_id = app.state.docids[idx]
                all_doc_ids.add(doc_id)
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        score=float(dist),
                        rank=rank,
                    )
                )
            queries_results.append(SearchQueryResults(query=query, results=results))

        doc_map: dict[str, dict] = {}
        if request.include_document and all_doc_ids:
            cursor = app.state.docs_db.cursor()
            placeholders = ",".join("?" for _ in all_doc_ids)
            cursor.execute(
                f"SELECT doc_id, json_data FROM documents WHERE doc_id IN ({placeholders})",
                tuple(all_doc_ids),
            )
            for doc_id, json_data in cursor.fetchall():
                doc_map[doc_id] = json.loads(json_data)

            for response in queries_results:
                for item in response.results:
                    doc = doc_map.get(item.doc_id)
                    if doc:
                        item.document = doc

        if len(queries_results) == 1:
            return SearchResponse(
                query=queries_results[0].query,
                results=queries_results[0].results,
                queries_results=[],
            )

        return SearchResponse(
            query="",
            results=[],
            queries_results=queries_results,
        )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            num_docs=app.state.config["num_docs"],
            embedding_dim=app.state.config["embedding_dim"],
        )

    return app
