"""
Example: How to perform offline search on an indexed dataset.

This example shows how to:
1. Load queries from a file
2. Embed queries using your embedding function
3. Search the index
4. Retrieve and return document results
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator
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
    # Example: Call an embedding API
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "https://api.example.com/embed",
    #         json={"texts": texts}
    #     )
    #     return np.array(response.json()["embeddings"])
    
    # Example: Use local model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(texts, convert_to_numpy=True)
    # return embeddings
    
    # Placeholder
    pass


async def read_queries(queries_file: Path) -> AsyncIterator[str]:
    """
    Read queries from file.
    
    Supports two formats:
    1. Plain text: One query per line
    2. JSON: Array of query objects
    
    Args:
        queries_file: Path to queries file
        
    Yields:
        Query strings
    """
    if queries_file.suffix == '.json':
        import json
        with open(queries_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, dict) and 'query' in item:
                        yield item['query']
    else:
        # Plain text: one query per line
        with open(queries_file, 'r') as f:
            for line in f:
                query = line.strip()
                if query:
                    yield query


async def batch_embed_queries(
    queries_file: Path,
    batch_size: int = 10
) -> AsyncIterator[np.ndarray]:
    """
    Batch embed queries from file.
    
    Args:
        queries_file: Path to queries file
        batch_size: Number of queries to embed at once
        
    Yields:
        Embeddings for each batch
    """
    batch = []
    
    async for query in read_queries(queries_file):
        batch.append(query)
        
        if len(batch) >= batch_size:
            yield await my_batch_embed(batch)
            batch = []
    
    # Process remaining queries
    if batch:
        yield await my_batch_embed(batch)


# ============================================================================
# Step 2: Perform offline search
# ============================================================================

async def main():
    """Main search execution."""
    
    # Configuration
    index_dir = Path("data/my_dataset")
    queries_file = Path("queries.txt")
    output_file = Path("search_results.json")
    
    # Perform search
    print(f"Searching index: {index_dir}")
    print(f"Queries file: {queries_file}")
    
    async for result in search(
        index_dir=index_dir,
        queries_file=queries_file,
        embed_fn=my_batch_embed,
        top_k=10,
        output_file=output_file,
    ):
        # Process each search result
        query = result['query']
        results = result['results']
        
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} results:")
        
        for item in results:
            doc_id = item['doc_id']
            score = item['score']
            rank = item['rank']
            document = item['document']
            
            print(f"  [{rank}] {doc_id} (score: {score:.4f})")
            print(f"      Title: {document.get('title', 'N/A')}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
