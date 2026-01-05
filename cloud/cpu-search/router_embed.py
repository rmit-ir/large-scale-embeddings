#!/usr/bin/env python3
"""
FastAPI router for embedding API following OpenAI's specification

Use conda environment: minicpmembed
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Literal
import time
from embed import EmbeddingModel
from logging_util import logging

logger = logging.getLogger('embed-router')


PORT = int(os.environ.get("PORT", 51003))


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
    timestamps: dict = Field(..., description="Timestamps for health check")


# Global instances
embedding_model: Optional[EmbeddingModel] = None


def load_embedding_model():
    global embedding_model
    print("Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_path="openbmb/MiniCPM-Embedding-Light")
    print("Embedding model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model initialization"""
    app.state.startup_time = time.time()
    load_embedding_model()
    # set startup complete time
    app.state.startup_complete_time = time.time()
    yield


# FastAPI app
app = FastAPI(
    title="Embedding API",
    description="Embedding API for text following OpenAI's specification",
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
        timestamps=({"timestamp": time.time(),
                     "startup_time": app.state.startup_time,
                     "startup_complete_time": app.state.startup_complete_time})
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, response: Response):
    start_time = time.perf_counter()

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
        embed_start = time.perf_counter()
        if len(inputs) == 1:
            embeddings = [embedding_model.embed(inputs[0])]
        else:
            embeddings = embedding_model.embed_docs(inputs)
        embed_time = (time.perf_counter() - embed_start) * 1000

        # Build response data
        build_start = time.perf_counter()
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
        build_time = (time.perf_counter() - build_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000

        # Add Server-Timing header
        response.headers["Server-Timing"] = f"embed;dur={embed_time:.2f}, build;dur={build_time:.2f}, total;dur={total_time:.2f}"

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
