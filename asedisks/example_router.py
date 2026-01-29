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


# ============================================================================
# Step 1: Define your async embed function
# ============================================================================


async def my_embed(query: str) -> np.ndarray:
    """
    Embed a single query text.

    The search router uses this to embed queries before searching.

    Args:
        query: Single query string to embed

    Returns:
        numpy array of shape (embedding_dim,)
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

    # Placeholder: Return random embedding for demonstration
    # Replace with actual embedding logic!
    embedding_dim = 384
    return np.random.randn(embedding_dim).astype(np.float32)


# ============================================================================
# Step 2: Run the search router server
# ============================================================================


async def main():
    """Run the search router server."""

    # Configuration
    index_dir = Path("data/my_dataset")
    host = "0.0.0.0"
    port = 8001

    print(f"Starting search router...")
    print(f"Index directory: {index_dir}")
    print(f"Server: http://{host}:{port}")
    print()
    print("Endpoints:")
    print(f"  POST http://{host}:{port}/search")
    print(f"  GET  http://{host}:{port}/health")
    print()

    await run_search_router(
        index_dir=index_dir,
        host=host,
        port=port,
        embed_fn=my_embed,
    )


# ============================================================================
# Step 3: Example client to query the server
# ============================================================================


async def query_server(query: str, top_k: int = 10):
    """
    Example client to query the search router server.

    Run this in a separate terminal after starting the server.

    Args:
        query: Query string
        top_k: Number of results to return
    """
    import httpx

    url = "http://localhost:8001/search"

    payload = {"query": query, "top_k": top_k, "include_document": True}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()


async def demo_client():
    """Demo client showing how to query the server."""
    query = "What is machine learning?"

    print(f"Querying: {query}")
    results = await query_server(query, top_k=5)

    print(f"\nResults for: {results['query']}")
    for item in results["results"]:
        print(f"  [{item['rank']}] {item['doc_id']} (score: {item['score']:.4f})")


if __name__ == "__main__":
    # To run the server:
    asyncio.run(main())

    # To test the client (run in separate terminal after server is started):
    # asyncio.run(demo_client())
