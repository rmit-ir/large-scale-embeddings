"""
Example: How to run a search router server.

This example shows how to:
1. Define your async embed function (optional, for inline embed node)
2. Run the search router server
3. Query the server via REST API
"""

import asyncio
from pathlib import Path
from typing import Optional
import numpy as np

from search import run_search_router


# ============================================================================
# Step 1: Define your async embed function (optional)
# ============================================================================

async def my_embed(query: str) -> np.ndarray:
    """
    Embed a single query text.
    
    If provided, the search router will run an inline embed node.
    If not provided (None), it will expect an external embed node.
    
    Args:
        query: Single query string to embed
        
    Returns:
        numpy array of shape (embedding_dim,)
    """
    # Example: Call an embedding API
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "https://api.example.com/embed",
    #         json={"texts": [query]}
    #     )
    #     return np.array(response.json()["embeddings"][0])
    
    # Example: Use local model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embedding = model.encode(query, convert_to_numpy=True)
    # return embedding
    
    # Placeholder
    pass


# ============================================================================
# Step 2: Run the search router server
# ============================================================================

async def main():
    """Run the search router server."""
    
    # Configuration
    index_dir = Path("data/my_dataset")
    host = "0.0.0.0"
    port = 8001
    
    # Option 1: Run with inline embed node
    print(f"Starting search router with inline embed node...")
    print(f"Index directory: {index_dir}")
    print(f"Server: http://{host}:{port}")
    
    await run_search_router(
        index_dir=index_dir,
        host=host,
        port=port,
        embed_fn=my_embed,  # Provide embed function for inline embed node
    )
    
    # Option 2: Run with external embed node (comment out above, use this instead)
    # print(f"Starting search router with external embed node...")
    # print(f"Index directory: {index_dir}")
    # print(f"Server: http://{host}:{port}")
    # print(f"Note: Configure external embed node URL in {index_dir}/config.json")
    #
    # await run_search_router(
    #     index_dir=index_dir,
    #     host=host,
    #     port=port,
    #     embed_fn=None,  # No inline embed node, use external
    # )


# ============================================================================
# Step 3: Query the server (example client)
# ============================================================================

async def query_server(query: str, top_k: int = 10):
    """
    Example client to query the search router server.
    
    Args:
        query: Query string
        top_k: Number of results to return
    """
    import httpx
    
    url = "http://localhost:8001/search"
    
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # To run the server:
    asyncio.run(main())
    
    # To query the server (run in separate terminal):
    # asyncio.run(query_server("What is machine learning?"))
