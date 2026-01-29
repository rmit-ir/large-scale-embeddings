"""
Example: How to create an index for a new dataset.

This example shows the complete workflow:
1. Define async data generator (ingester)
2. Define async batch embed function
3. Use output_to_idx() to write search-ready files
4. Build DiskANN index
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Any
import numpy as np

from dataset_tools import (
    output_to_idx,
    build_index,
    Document,
    OutputConfig,
    DiskANNConfig,
)


# ============================================================================
# Step 1: Define your async data generator
# ============================================================================

async def my_data_generator() -> AsyncIterator[Document]:
    """
    Async generator that yields Document objects from your data source.
    
    This is where you implement your data loading logic. Examples:
    - Read from files on disk
    - Fetch from an API
    - Query a database
    - Stream from a web service
    """
    # Example: Read from files
    for file_path in Path("my_data/*.json").glob("*.json"):
        # Load document (replace with your logic)
        data = load_document(file_path)
        
        # Yield Document object
        yield Document(
            id=data["id"],
            title=data["title"],
            body=data["body"],
            whole_doc=data,  # Full document for SQLite storage
            url=data.get("url"),
            fetched_at=data.get("fetched_at"),
            # Add any other fields you need
        )


def load_document(file_path: Path) -> dict:
    """
    Load a single document from file.
    
    Replace this with your actual document loading logic.
    """
    import json
    with open(file_path, 'r') as f:
        return json.load(f)


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
    # Example: Call an embedding API
    # Replace with your actual embedding logic
    
    # Option 1: Use external API
    # import httpx
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "https://api.example.com/embed",
    #         json={"texts": texts}
    #     )
    #     return np.array(response.json()["embeddings"])
    
    # Option 2: Use local model
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(texts, convert_to_numpy=True)
    # return embeddings
    
    # Placeholder
    pass


async def my_embed_async_generator(documents: AsyncIterator[Document]) -> AsyncIterator[np.ndarray]:
    """
    Wrapper that embeds documents as they arrive.
    
    This helper function creates embeddings from documents.
    """
    batch = []
    batch_size = 100  # Adjust based on your needs
    
    async for doc in documents:
        batch.append(doc.get_search_text())
        
        if len(batch) >= batch_size:
            yield await my_batch_embed(batch)
            batch = []
    
    # Process remaining documents
    if batch:
        yield await my_batch_embed(batch)


# ============================================================================
# Step 3: Run the pipeline
# ============================================================================

async def main():
    """Main pipeline execution."""
    
    # Configuration
    output_dir = Path("data/my_dataset")
    documents = my_data_generator()
    embeddings = my_embed_async_generator(documents)
    
    output_config = OutputConfig(
        sqlite_compression=True,
        compression_level=5,
        batch_size=1000,
    )
    
    # Step 3a: Write search-ready files (SQLite + DiskANN binary)
    print("Writing search-ready files...")
    await output_to_idx(
        output_dir=output_dir,
        documents=documents,
        embeddings=embeddings,
        config=output_config,
    )
    print(f"Files written to {output_dir}")
    
    # Step 3b: Build DiskANN index
    print("Building DiskANN index...")
    build_index(
        binary_file=output_dir / "embeds.bin",
        output_dir=output_dir / "index",
        config=DiskANNConfig(
            metric="L2",
            index_build_threads=32,
            R=64,
            L=100,
        ),
    )
    print(f"Index built at {output_dir / 'index'}")


if __name__ == "__main__":
    asyncio.run(main())
