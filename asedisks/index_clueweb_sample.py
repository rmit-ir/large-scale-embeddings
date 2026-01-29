"""
Index ClueWeb22-B sample dataset using ASEDISKS.

This script:
1. Reads documents from ClueWeb22-B sample data
2. Embeds them using MiniCPM-Embedding-Light locally
3. Builds a DiskANN index
4. Runs test searches
"""

import asyncio
import gzip
import json
import os
from pathlib import Path
from typing import AsyncIterator
import numpy as np
import torch

from dataset_tools import output_to_idx, build_index, DataRecord, OutputConfig, DiskANNConfig
from search import search


# ============================================================================
# Configuration
# ============================================================================

CLUEWEB_ROOT = Path("/Users/kun/Projects/rmit/research/ase2.0/large-scale-embeddings/data/datasets/clueweb22-b")
OUTPUT_DIR = Path("/Users/kun/Projects/rmit/research/ase2.0/large-scale-embeddings/data/asedisks_test/clueweb22-sample")
MODEL_NAME = "openbmb/MiniCPM-Embedding-Light"
MAX_DOCS = 1000  # Limit for testing (set to None for all ~19k docs)
BATCH_SIZE = 32  # Embedding batch size


# ============================================================================
# ClueWeb22 Document Reader
# ============================================================================

class ClueWeb22Reader:
    """Read documents from ClueWeb22-B format."""

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def iter_documents(self, max_docs: int | None = None):
        """
        Iterate over all documents in the dataset.

        Yields:
            Tuple of (doc_id, doc_data) where doc_data is parsed JSON
        """
        txt_path = self.root_path / "txt"

        # Find all .json.gz files
        for json_gz in txt_path.rglob("*.json.gz"):
            offset_path = json_gz.with_suffix("").with_suffix(".offset")

            if not offset_path.exists():
                continue

            # Parse shard info from filename: en0000-00.json.gz -> en0000, 00
            shard_name = json_gz.stem.replace(".json", "")  # en0000-00

            # Count documents in this shard
            with open(offset_path, "r") as f:
                offsets = f.readlines()
            num_docs = len(offsets) - 1  # Last line is end offset

            # Read documents
            count = 0
            with open(json_gz, "rb") as f_json:
                for doc_idx in range(num_docs):
                    if max_docs and count >= max_docs:
                        return

                    # Read offsets
                    start_bytes = int(offsets[doc_idx].strip())
                    end_bytes = int(offsets[doc_idx + 1].strip())

                    # Read and decompress record
                    f_json.seek(start_bytes)
                    record = f_json.read(end_bytes - start_bytes)
                    record = gzip.decompress(record).decode("utf-8")

                    try:
                        doc_data = json.loads(record)
                        doc_id = f"clueweb22-{shard_name}-{doc_idx:05d}"
                        yield doc_id, doc_data
                        count += 1
                    except json.JSONDecodeError:
                        continue


# ============================================================================
# Embedding Model
# ============================================================================

class MiniCPMEmbedder:
    """
    MiniCPM-Embedding-Light model for local embedding.

    Uses transformers library with CUDA acceleration.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto"):
        from transformers import AutoModel

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")

        # Determine dtype based on device
        if self.device == "cuda":
            self.dtype = torch.float16
        elif self.device == "mps":
            self.dtype = torch.float32  # MPS doesn't support float16 well
        else:
            self.dtype = torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        print(f"Model loaded. Embedding dimension: 1024")

    @torch.no_grad()
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Encode passages/documents.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        # Use the model's encode_corpus method
        embeddings, _ = self.model.encode_corpus(texts, return_sparse_vectors=False)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings.astype(np.float32)

    @torch.no_grad()
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """
        Encode queries.

        Args:
            texts: List of query strings

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        # Use the model's encode_query method
        embeddings, _ = self.model.encode_query(texts, return_sparse_vectors=False)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings.astype(np.float32)


# Global embedder instance (loaded once)
_embedder: MiniCPMEmbedder | None = None


def get_embedder() -> MiniCPMEmbedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = MiniCPMEmbedder()
    return _embedder


# ============================================================================
# Data Generator
# ============================================================================


async def clueweb_records(max_docs: int | None = MAX_DOCS) -> AsyncIterator[DataRecord]:
    """
    Async generator yielding DataRecord from ClueWeb22-B sample.

    Args:
        max_docs: Maximum documents to process (None for all)

    Yields:
        DataRecord objects
    """
    reader = ClueWeb22Reader(CLUEWEB_ROOT)

    for doc_id, doc_data in reader.iter_documents(max_docs=max_docs):
        # Extract clean text
        clean_text = doc_data.get("Clean-Text", "")
        clueweb_id = doc_data.get("ClueWeb22-ID", doc_id)

        if not clean_text.strip():
            continue

        # Extract title from first line
        lines = clean_text.split("\n", 1)
        title = lines[0].strip()[:500] if lines else ""
        body = lines[1].strip() if len(lines) > 1 else clean_text

        # Format content for embedding
        content = f"{title}\n\n{body}"

        # Truncate to reasonable length (model max is 8192 tokens)
        content = content[:8000]

        yield DataRecord(
            id=clueweb_id,
            content=content,
            metadata={
                "id": clueweb_id,
                "title": title,
                "text": clean_text[:2000],  # Truncate for storage
                "url": doc_data.get("URL", ""),
            },
        )


# ============================================================================
# Embedding Function
# ============================================================================


async def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts using MiniCPM-Embedding-Light.

    Args:
        texts: List of text strings

    Returns:
        numpy array of shape (len(texts), 1024)
    """
    embedder = get_embedder()
    return embedder.encode_passages(texts)


async def embed_queries_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of queries using MiniCPM-Embedding-Light.

    Args:
        texts: List of query strings

    Returns:
        numpy array of shape (len(texts), 1024)
    """
    embedder = get_embedder()
    return embedder.encode_queries(texts)


# ============================================================================
# Main Pipeline
# ============================================================================


async def run_indexing():
    """Run the indexing pipeline."""
    print("=" * 60)
    print("ASEDISKS - ClueWeb22-B Sample Indexing")
    print("=" * 60)
    print(f"Source: {CLUEWEB_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max docs: {MAX_DOCS}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Initialize embedder (loads model)
    get_embedder()

    # Run indexing
    print("Starting indexing...")
    await output_to_idx(
        output_dir=OUTPUT_DIR,
        records=clueweb_records(max_docs=MAX_DOCS),
        embed_fn=embed_batch,
        config=OutputConfig(
            batch_size=BATCH_SIZE,
            sqlite_compression=False,  # Keep it simple for testing
        ),
    )

    print("\nIndexing complete!")
    print(f"Output files in: {OUTPUT_DIR}")


async def run_index_build():
    """Build DiskANN index."""
    print("\n" + "=" * 60)
    print("Building DiskANN Index")
    print("=" * 60)

    try:
        build_index(
            binary_file=OUTPUT_DIR / "embeds.bin",
            output_dir=OUTPUT_DIR / "index",
            config=DiskANNConfig(
                metric="mips",
                R=32,  # Smaller for test dataset
                L=50,
                build_memory_gb=4,
                search_memory_gb=2,
            ),
        )
        print("Index build complete!")
        return True
    except FileNotFoundError as e:
        print(f"Warning: DiskANN not found - skipping index build")
        print(f"  {e}")
        print("  You can still search using brute force if needed.")
        return False


async def run_search_test():
    """Run test searches."""
    print("\n" + "=" * 60)
    print("Running Test Searches")
    print("=" * 60)

    # Test queries
    queries = [
        "machine learning artificial intelligence",
        "climate change global warming",
        "healthy food nutrition diet",
        "programming software development",
        "travel vacation destinations",
    ]

    print(f"Testing {len(queries)} queries...")

    try:
        results = await search(
            index_dir=OUTPUT_DIR,
            queries=queries,
            embed_fn=embed_queries_batch,
            top_k=5,
            complexity=50,
            include_documents=True,
        )

        print("\n" + "-" * 60)
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Results:")
            for item in result["results"][:3]:  # Show top 3
                doc = item.get("document", {})
                title = doc.get("title", "N/A")[:60]
                print(f"  [{item['rank']}] {item['doc_id']}")
                print(f"      Score: {item['score']:.4f}")
                print(f"      Title: {title}...")

        print("\n" + "-" * 60)
        print("Search test complete!")

    except Exception as e:
        print(f"Search failed: {e}")
        print("This may be because DiskANN index was not built.")


# ============================================================================
# Entry Point
# ============================================================================


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "index":
            await run_indexing()
        elif command == "build":
            await run_index_build()
        elif command == "search":
            await run_search_test()
        elif command == "all":
            await run_indexing()
            has_index = await run_index_build()
            if has_index:
                await run_search_test()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python index_clueweb_sample.py [index|build|search|all]")
    else:
        # Default: run full pipeline
        await run_indexing()
        has_index = await run_index_build()
        if has_index:
            await run_search_test()


if __name__ == "__main__":
    asyncio.run(main())
