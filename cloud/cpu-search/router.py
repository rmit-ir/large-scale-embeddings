#!/usr/bin/env python3
"""
FastAPI router for search API using DiskANN and Tevatron

Use conda environment: minicpmembed
"""

import os
import pickle
import sys
import uvicorn
import httpx
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
from docs_loader import DocsLoader
from logging_util import logging

logger = logging.getLogger('cpu-search-router')


DOC_ID_MAPPING_PATH = os.environ.get("DOC_ID_MAPPING_PATH",
                                     "../../data/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light-diskann/docids.pkl")
DOC_DB_PATH = os.environ.get(
    "DOC_DB_PATH", "../../data/clueweb-docs-db/clueweb22b_en.db")
USE_COMPRESSION = os.environ.get("USE_COMPRESSION", "true").lower() == "true"
PORT = int(os.environ.get("PORT", 8000))
DOCS_LOADER_NUM_THREADS = int(os.environ.get("DOCS_LOADER_NUM_THREADS", -1))
SEARCH_NODE_URL = os.environ.get("SEARCH_NODE_URL",
                                 "http://localhost:51001/search")
EMBED_NODE_URL = os.environ.get("EMBED_NODE_URL",
                                "http://localhost:51003/embed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    num_docs: int = Field(..., description="Number of documents in mapping")
    timestamps: dict = Field(..., description="Timestamps for health check")


# Search request/response models
class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., description="Search query text")
    k: int = Field(default=10, description="Number of results to return")
    complexity: int = Field(
        default=50, description="Search complexity parameter")


class SearchResult(BaseModel):
    """Single search result"""
    docid: str = Field(..., description="ClueWeb22-B document ID")
    distance: Optional[float] = Field(
        None, description="Similarity score")
    doc: Optional[Dict[str, Any]] = Field(
        None, description="Raw JSON body of the document")


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    k: int = Field(..., description="Number of results requested")


# Global instances
docid_mapping: Optional[List[str]] = None
docs_loader: Optional[DocsLoader] = None


def load_docid_mapping():
    global docid_mapping
    docid_path = Path(DOC_ID_MAPPING_PATH)
    if docid_path.exists():
        logger.info(f"Loading docid mapping from {docid_path}...")
        try:
            with open(docid_path, "rb") as f:
                docid_mapping = pickle.load(f)
            logger.info(
                f"Loaded {len(docid_mapping or [])} document ID mappings")
        except Exception as e:
            logger.error(f"Failed to load docid mapping: {str(e)}")
            raise
    else:
        logger.error(f"Docid mapping file not found at {docid_path}")
        raise Exception(f"Docid mapping file not found at {docid_path}")


def load_docs_loader():
    global docs_loader
    if DOC_DB_PATH:
        logger.info(
            f"Initializing document loader with database: {DOC_DB_PATH}, compression={USE_COMPRESSION}, threads={DOCS_LOADER_NUM_THREADS}")
        try:
            docs_loader = DocsLoader(
                db_path=DOC_DB_PATH,
                use_compression=USE_COMPRESSION,
                num_threads=DOCS_LOADER_NUM_THREADS)
            logger.info("Document loader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document loader: {str(e)}")
            raise
    else:
        logger.warning(
            "DOC_DB_PATH not set, document loader will not be available")
        docs_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for initialization"""
    app.state.startup_time = time.time()
    logger.info("Application startup initiated")
    load_docid_mapping()
    load_docs_loader()
    # set startup complete time
    app.state.startup_complete_time = time.time()
    logger.info(
        f"Application startup complete in {app.state.startup_complete_time - app.state.startup_time:.2f}s")
    yield
    logger.info("Application shutdown initiated")


# FastAPI app
app = FastAPI(
    title="Search API",
    description="Search API for ClueWeb22-B using DiskANN and Tevatron",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
        Health status and model loading state
    """
    num_docs = len(docid_mapping) if docid_mapping is not None else 0
    logger.debug(f"Health check: status=healthy, num_docs={num_docs}")
    return HealthResponse(
        status="healthy",
        num_docs=num_docs,
        timestamps=({"timestamp": time.time(),
                     "startup_time": app.state.startup_time,
                     "startup_complete_time": app.state.startup_complete_time})
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, response: Response):
    """
    Search endpoint that performs semantic search over ClueWeb22-B corpus

    This endpoint:
    1. Calls the embedding service to embed the query
    2. Sends the embedding to the DiskANN search node at localhost:51001
    3. Translates raw internal IDs to ClueWeb22-B document IDs
    4. Returns the search results

    Returns:
        SearchResponse with results containing ClueWeb22-B document IDs
    """
    start_time = time.perf_counter()

    logger.info(
        f"Received search request: query='{request.query}', k={request.k}, complexity={request.complexity}")

    try:
        # Step 1: Call embedding service to embed the query
        embed_start = time.perf_counter()
        embed_payload = {
            "input": request.query,
            "model": "openbmb/MiniCPM-Embedding-Light"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            embed_response = await client.post(EMBED_NODE_URL, json=embed_payload)

        if embed_response.status_code != 200:
            logger.error(
                f"Embed service error: status={embed_response.status_code}, response={embed_response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Embed service returned error: {embed_response.text}"
            )

        embed_data = embed_response.json()
        query_embedding = embed_data["data"][0]["embedding"]
        embed_time = (time.perf_counter() - embed_start) * 1000

        logger.debug(f"Query embedded successfully in {embed_time:.2f}ms")

        # Step 2: Send request to DiskANN search node at localhost:51001
        search_start = time.perf_counter()
        payload = {
            "q_emb": query_embedding,
            "k": request.k,
            "complexity": request.complexity
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            http_response = await client.post(SEARCH_NODE_URL, json=payload)

        if http_response.status_code != 200:
            logger.error(
                f"Search node error: status={http_response.status_code}, response={http_response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Search node returned error: {http_response.text}"
            )

        search_data = http_response.json()
        search_time = (time.perf_counter() - search_start) * 1000

        logger.debug(f"DiskANN search completed in {search_time:.2f}ms")

        raw_indices = search_data["indices"]
        distances = search_data["distances"]

        # Step 3: Translate raw indices to ClueWeb22-B document IDs
        expand_start = time.perf_counter()
        # translate_start = time.perf_counter()
        docids = []
        results = []
        for idx, (raw_id, distance) in enumerate(zip(raw_indices, distances)):
            if docid_mapping is not None and 0 <= raw_id < len(docid_mapping):
                docid = docid_mapping[raw_id]
                docids.append(docid)
                results.append(SearchResult(
                    docid=docid,
                    distance=distance,
                    doc=None))
            else:
                logger.warning(
                    f"No docid mapping for raw_id {raw_id}, query: {request.query}")
                continue  # Skip if mapping is not available
        # translate_time = (time.perf_counter() - translate_start) * 1000

        # Step 4: Load document bodies if requested
        if docs_loader is not None and docids:
            docs = docs_loader.load_docs_by_ids(docids)
            for result, doc in zip(results, docs):
                result.doc = doc
        expand_time = (time.perf_counter() - expand_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000

        # Add Server-Timing header
        response.headers["Server-Timing"] = f"embed;dur={embed_time:.2f}, search;dur={search_time:.2f}, expand;dur={expand_time:.2f}, total;dur={total_time:.2f}"

        logger.info(f"Search completed: query='{request.query}', results={len(results)}, "
                    f"embed_time={embed_time:.2f}ms, search_time={search_time:.2f}ms, "
                    f"expand_time={expand_time:.2f}ms, total_time={total_time:.2f}ms")

        return SearchResponse(
            results=results,
            query=request.query,
            k=request.k
        )

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to service: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing search request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )


if __name__ == "__main__":
    logger.info(f"Starting search router on port {PORT}")
    logger.info(f"Search node URL: {SEARCH_NODE_URL}")
    logger.info(f"Embed node URL: {EMBED_NODE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
