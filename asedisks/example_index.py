"""
Example: How to create an index for a new dataset.

This example shows the complete workflow:
1. Define async data generator (yields DataRecord)
2. Define async batch embed function
3. Use output_to_idx() to write search-ready files
4. Build DiskANN index
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator
import numpy as np

from dataset_tools import (
    output_to_idx,
    build_index,
    DataRecord,
    OutputConfig,
    DiskANNConfig,
)


# ============================================================================
# Step 1: Define your async data generator
# ============================================================================


async def my_data_generator() -> AsyncIterator[DataRecord]:
    """
    Async generator that yields DataRecord objects from your data source.

    This is where you implement your data loading logic. Examples:
    - Read from files on disk
    - Fetch from an API
    - Query a database
    - Stream from a web service

    Each record has:
    - id: Unique document identifier
    - content: Text to embed (you format title/body/etc into this)
    - metadata: Full document data for storage/retrieval
    """
    # Example: Read from JSON files
    data_dir = Path("my_data")

    # Simulated data for example
    sample_data = [
        {
            "id": "doc-001",
            "title": "Introduction to Machine Learning",
            "body": "Machine learning is a subset of artificial intelligence...",
            "url": "https://example.com/ml-intro",
        },
        {
            "id": "doc-002",
            "title": "Deep Learning Fundamentals",
            "body": "Deep learning uses neural networks with many layers...",
            "url": "https://example.com/dl-fundamentals",
        },
        {
            "id": "doc-003",
            "title": "Natural Language Processing",
            "body": "NLP enables computers to understand human language...",
            "url": "https://example.com/nlp-overview",
        },
    ]

    for data in sample_data:
        # Format content - user's responsibility to combine fields
        content = f"{data['title']}\n\n{data['body']}"

        yield DataRecord(
            id=data["id"],
            content=content,  # Text that will be embedded
            metadata=data,  # Full document for storage
        )


# ============================================================================
# Step 2: Define your async batch embed function
# ============================================================================


async def my_batch_embed(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts.

    This function receives a list of texts and returns embeddings.
    You can implement this by:
    - Calling an embedding API (OpenAI, Cohere, etc.)
    - Running a local model (HuggingFace, sentence-transformers, etc.)
    - Using any other embedding service

    Args:
        texts: List of strings to embed

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    # Option 1: Use external API (e.g., your embed router)
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
    embedding_dim = 384  # e.g., all-MiniLM-L6-v2 dimension
    return np.random.randn(len(texts), embedding_dim).astype(np.float32)


# ============================================================================
# Step 3: Run the pipeline
# ============================================================================


async def main():
    """Main pipeline execution."""

    # Configuration
    output_dir = Path("data/my_dataset")

    output_config = OutputConfig(
        sqlite_compression=True,
        compression_level=5,
        batch_size=100,  # Adjust based on your embed function's capacity
    )

    # Step 3a: Write search-ready files (SQLite + DiskANN binary)
    print("Writing search-ready files...")
    await output_to_idx(
        output_dir=output_dir,
        records=my_data_generator(),
        embed_fn=my_batch_embed,
        config=output_config,
    )
    print(f"Files written to {output_dir}")

    # Step 3b: Build DiskANN index
    print("\nBuilding DiskANN index...")
    build_index(
        binary_file=output_dir / "embeds.bin",
        output_dir=output_dir / "index",
        config=DiskANNConfig(
            metric="mips",  # Use MIPS for normalized embeddings
            R=64,
            L=100,
            build_memory_gb=8,
            search_memory_gb=4,
        ),
    )
    print(f"Index built at {output_dir / 'index'}")


if __name__ == "__main__":
    asyncio.run(main())
