"""
Example: How to perform offline search on an indexed dataset.

This example shows how to:
1. Define your async embed function (same as during indexing)
2. Search the index with queries
3. Retrieve and display document results
"""

import asyncio
from pathlib import Path
import numpy as np

from search import search


# ============================================================================
# Step 1: Define your async embed function
# ============================================================================


async def my_batch_embed(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of query texts.

    This function should match the one used during indexing.

    Args:
        texts: List of query strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    # Option 1: Call an embedding API (e.g., your embed router)
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "http://localhost:51003/embed",
    #         json={"input": texts},
    #         timeout=60.0
    #     )
    #     data = response.json()
    #     return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)

    # Option 2: Use local model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # return embeddings

    # Placeholder: Return random embeddings for demonstration
    # Replace with actual embedding logic!
    embedding_dim = 384
    return np.random.randn(len(texts), embedding_dim).astype(np.float32)


# ============================================================================
# Step 2: Perform offline search
# ============================================================================


async def main():
    """Main search execution."""

    # Configuration
    index_dir = Path("data/my_dataset")

    # Example queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
    ]

    # Alternative: Load queries from file
    # queries = Path("queries.txt")

    print(f"Searching index: {index_dir}")
    print(f"Number of queries: {len(queries)}")

    # Perform search
    results = await search(
        index_dir=index_dir,
        queries=queries,
        embed_fn=my_batch_embed,
        top_k=10,
        complexity=100,
        include_documents=True,  # Include full document content
        output_file=Path("search_results.json"),  # Optional: save results
    )

    # Display results
    for result in results:
        query = result["query"]
        hits = result["results"]

        print(f"\nQuery: {query}")
        print(f"Found {len(hits)} results:")

        for item in hits:
            doc_id = item["doc_id"]
            score = item["score"]
            rank = item["rank"]
            document = item.get("document", {})

            print(f"  [{rank}] {doc_id} (score: {score:.4f})")
            if document:
                print(f"      Title: {document.get('title', 'N/A')}")

    print("\nResults saved to: search_results.json")


if __name__ == "__main__":
    asyncio.run(main())
