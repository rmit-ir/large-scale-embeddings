"""
Example: How to run a search router server.

This example shows how to:
1. Define your async embed function for inline embedding
2. Run the search router server
3. Query the server via REST API
"""

import asyncio
from pathlib import Path
import numpy as np

from search import run_search_router
from aseann.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Step 1: Define your async embed function
# ============================================================================


async def my_embed(queries: list[str]) -> np.ndarray:
    """
    Embed a batch of query texts.

    The search router uses this to embed queries before searching.

    Args:
        queries: List of query strings to embed

    Returns:
        numpy array of shape (len(queries), embedding_dim)
    """
    # Option 1: Call an embedding API (e.g., your embed router)
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "http://localhost:51003/embed",
    #         json={"input": [query]},
    #         timeout=60.0
    #     )
    #     data = response.json()
    #     return np.array(data["data"][0]["embedding"], dtype=np.float32)

    # Option 2: Use local model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    # return embedding

    # Placeholder: Return random embeddings for demonstration
    # Replace with actual embedding logic!
    embedding_dim = 1024
    return np.random.randn(len(queries), embedding_dim).astype(np.float32)


# ============================================================================
# Step 2: Run the search router server
# ============================================================================


async def main():
    """Run the search router server."""

    # Configuration
    index_dir = Path("data/my_dataset")
    host = "0.0.0.0"
    port = 8001

    logger.info("Starting search router...")
    logger.info("Index directory: %s", index_dir)
    logger.info("Server: http://%s:%s", host, port)
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST http://%s:%s/search", host, port)
    logger.info("  GET  http://%s:%s/health", host, port)
    logger.info("")

    await run_search_router(
        index_dir=index_dir,
        host=host,
        port=port,
        embed_fn=my_embed,
    )


# ============================================================================
# Step 3: Example client to query the server
# ============================================================================


async def query_server(queries: list[str], top_k: int = 10):
    """
    Example client to query the search router server.

    Run this in a separate terminal after starting the server.

    Args:
        queries: Query strings
        top_k: Number of results to return
    """
    import httpx

    url = "http://localhost:8001/search"

    payload = {"queries": queries, "top_k": top_k, "include_document": True}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()


async def demo_client():
    """Demo client showing how to query the server."""
    query = "What is machine learning?"

    logger.info("Querying: %s", query)
    results = await query_server([query], top_k=5)

    if results.get("queries_results"):
        for result in results["queries_results"]:
            logger.info("Results for: %s", result["query"])
            for item in result["results"]:
                logger.info(
                    "  [%s] %s (score: %.4f)",
                    item["rank"],
                    item["doc_id"],
                    item["score"],
                )
    else:
        logger.info("Results for: %s", results["query"])
        for item in results["results"]:
            logger.info(
                "  [%s] %s (score: %.4f)",
                item["rank"],
                item["doc_id"],
                item["score"],
            )


if __name__ == "__main__":
    # To run the server:
    asyncio.run(main())

    # To test the client (run in separate terminal after server is started):
    # asyncio.run(demo_client())
