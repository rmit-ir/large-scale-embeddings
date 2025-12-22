#!/usr/bin/env python3
"""
FastAPI router for embedding API following OpenAI's specification

Use conda environment: minicpmembed
"""

import os
import pickle
import uvicorn
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Literal
import time
from embed import EmbeddingModel

DOC_ID_MAPPING_PATH = os.environ.get("DOC_ID_MAPPING_PATH",
                                     "../../data/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light-diskann/docids.pkl")
PORT = int(os.environ.get("PORT", 8000))
SEARCH_NODE_URL = os.environ.get("SEARCH_NODE_URL",
                                 "http://localhost:51001/search")


class EmbeddingRequest(BaseModel):
    """Request model following OpenAI embeddings API specification"""
    input: Union[str, List[str]] = Field(...,
                                         description="Input text(s) to embed")
    model: str = Field(default="openbmb/MiniCPM-Embedding-Light",
                       description="Model identifier, not used, fixed to openbmb/MiniCPM-Embedding-Light")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="Encoding format")
    dimensions: Optional[int] = Field(
        default=None, description="Number of dimensions (not used)")
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingData(BaseModel):
    """Single embedding response data"""
    object: str = Field(default="embedding", description="Object type")
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="Index of the embedding in the list")


class EmbeddingUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = Field(...,
                               description="Number of tokens in the prompt")
    total_tokens: int = Field(..., description="Total tokens used")


class EmbeddingResponse(BaseModel):
    """Response model following OpenAI embeddings API specification"""
    object: str = Field(default="list", description="Object type")
    data: List[EmbeddingData] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used")
    usage: EmbeddingUsage = Field(..., description="Token usage")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
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


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    k: int = Field(..., description="Number of results requested")


# Global model instance
embedding_model: Optional[EmbeddingModel] = None
docid_mapping: Optional[List[str]] = None


def load_embedding_model():
    global embedding_model
    print("Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_path="openbmb/MiniCPM-Embedding-Light")
    print("Embedding model loaded successfully")


def load_docid_mapping():
    global docid_mapping
    docid_path = Path(DOC_ID_MAPPING_PATH)
    if docid_path.exists():
        print(f"Loading docid mapping from {docid_path}...")
        with open(docid_path, "rb") as f:
            docid_mapping = pickle.load(f)
        print(f"Loaded {len(docid_mapping or [])} document ID mappings")
    else:
        raise Exception(f"Docid mapping file not found at {docid_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model initialization"""
    app.state.startup_time = time.time()
    load_embedding_model()
    load_docid_mapping()
    # set startup complete time
    app.state.startup_complete_time = time.time()
    yield


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
    return HealthResponse(
        status="healthy" if embedding_model is not None else "initializing",
        model_loaded=embedding_model is not None,
        num_docs=len(docid_mapping) if docid_mapping is not None else 0,
        timestamps=({"timestamp": time.time(),
                     "startup_time": app.state.startup_time,
                     "startup_complete_time": app.state.startup_complete_time})
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Handle both single string and list of strings
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input

        if not inputs:
            raise HTTPException(
                status_code=400, detail="Input cannot be empty")

        # Get embeddings
        if len(inputs) == 1:
            embeddings = [embedding_model.embed(inputs[0])]
        else:
            embeddings = embedding_model.embed_docs(inputs)

        # Build response data
        data = [
            EmbeddingData(
                object="embedding",
                embedding=emb,
                index=idx
            )
            for idx, emb in enumerate(embeddings)
        ]

        # Approximate token count (simple word count * 1.3)
        total_tokens = sum(len(text.split()) for text in inputs)
        total_tokens = int(total_tokens * 1.3)

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search endpoint that performs semantic search over ClueWeb22-B corpus

    This endpoint:
    1. Embeds the query using the embedding model
    2. Sends the embedding to the DiskANN search node at localhost:51001
    3. Translates raw internal IDs to ClueWeb22-B document IDs
    4. Returns the search results

    Returns:
        SearchResponse with results containing ClueWeb22-B document IDs
    """
    if embedding_model is None:
        raise HTTPException(
            status_code=503, detail="Embedding model not loaded yet")

    try:
        # Step 1: Embed the query
        query_embedding = embedding_model.embed(request.query)

        # Step 2: Send request to DiskANN search node at localhost:51001
        payload = {
            "q_emb": query_embedding,
            "k": request.k,
            "complexity": request.complexity
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(SEARCH_NODE_URL, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Search node returned error: {response.text}"
            )

        search_data = response.json()
        raw_indices = search_data["indices"]
        distances = search_data["distances"]

        # Step 3: Translate raw indices to ClueWeb22-B document IDs
        results = []
        for idx, (raw_id, distance) in enumerate(zip(raw_indices, distances)):
            if docid_mapping is not None and 0 <= raw_id < len(docid_mapping):
                docid = docid_mapping[raw_id]
            else:
                print(
                    f"Warning: No docid mapping for raw_id {raw_id}, query: {request.query}")
                continue  # Skip if mapping is not available

            results.append(SearchResult(docid=docid, distance=distance))

        return SearchResponse(
            results=results,
            query=request.query,
            k=request.k
        )

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to search node: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
