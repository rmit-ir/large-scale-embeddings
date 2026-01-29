"""
Index ClueWeb22-B sample dataset using ASEDISKS.

This script:
1. Reads documents from ClueWeb22-B sample data
2. Embeds them using MiniCPM-Embedding-Light locally (Transformers)
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

from asedisks.dataset_tools import (
    output_to_idx,
    build_index,
    DataRecord,
    OutputConfig,
    DiskANNConfig,
)
from asedisks.search import search


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

CLUEWEB_ROOT = Path(
    os.environ.get(
        "CLUEWEB_ROOT",
        "/home/ubuntu/projects/large-scale-embeddings/data/datasets/clueweb22-b",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CLUEWEB_OUTPUT_DIR",
        str(REPO_ROOT / "data/asedisks_test/clueweb22-sample"),
    )
)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "openbmb/MiniCPM-Embedding-Light")
MAX_DOCS = int(os.environ.get("MAX_DOCS", "1000"))  # Limit for testing (set to 0 for all)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "0") == "1"


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

    Uses transformers with the model's encode_query/encode_corpus helpers.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto"):
        from transformers import AutoModel

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        if USE_FLASH_ATTN and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()

        print("Model loaded.")

    def _to_numpy(self, embeddings) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    @torch.inference_mode()
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Encode passages/documents.

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        if not hasattr(self.model, "encode_corpus"):
            raise RuntimeError("Model does not implement encode_corpus().")
        embeddings, _ = self.model.encode_corpus(texts, return_sparse_vectors=False)
        return self._to_numpy(embeddings)

    @torch.inference_mode()
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """
        Encode queries.

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        if not hasattr(self.model, "encode_query"):
            raise RuntimeError("Model does not implement encode_query().")
        embeddings, _ = self.model.encode_query(texts, return_sparse_vectors=False)
        return self._to_numpy(embeddings)


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
        max_docs: Maximum documents to process (0 for all)

    Yields:
        DataRecord objects
    """
    reader = ClueWeb22Reader(CLUEWEB_ROOT)
    effective_max = None if max_docs == 0 else max_docs

    for doc_id, doc_data in reader.iter_documents(max_docs=effective_max):
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
                "text": clean_text[:2000],
                "url": doc_data.get("URL", ""),
            },
        )


# ============================================================================
# Embedding Functions
# ============================================================================


async def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts using MiniCPM-Embedding-Light."""
    embedder = get_embedder()
    return embedder.encode_passages(texts)


async def embed_queries_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of queries using MiniCPM-Embedding-Light."""
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
    print(f"Max docs: {MAX_DOCS if MAX_DOCS != 0 else 'ALL'}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    if not CLUEWEB_ROOT.exists():
        raise FileNotFoundError(f"Dataset path not found: {CLUEWEB_ROOT}")

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

    build_index(
        binary_file=OUTPUT_DIR / "embeds.bin",
        output_dir=OUTPUT_DIR / "index",
        config=DiskANNConfig(
            metric="mips",
            R=32,
            L=50,
            build_memory_gb=4,
            search_memory_gb=2,
        ),
    )
    print("Index build complete!")


async def run_search_test():
    """Run test searches."""
    print("\n" + "=" * 60)
    print("Running Test Searches")
    print("=" * 60)

    queries = [
        "machine learning artificial intelligence",
        "climate change global warming",
        "healthy food nutrition diet",
        "programming software development",
        "travel vacation destinations",
    ]

    print(f"Testing {len(queries)} queries...")

    if not (OUTPUT_DIR / "index").exists():
        raise FileNotFoundError("DiskANN index not found. Run the build step first.")

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
        print("Results:")
        for item in result["results"][:3]:
            doc = item.get("document", {})
            title = doc.get("title", "N/A")[:60]
            print(f"  [{item['rank']}] {item['doc_id']}")
            print(f"      Score: {item['score']:.4f}")
            print(f"      Title: {title}...")

    print("\n" + "-" * 60)
    print("Search test complete!")


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
            await run_index_build()
            await run_search_test()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python index_clueweb_sample.py [index|build|search|all]")
    else:
        await run_indexing()
        await run_index_build()
        await run_search_test()


if __name__ == "__main__":
    asyncio.run(main())
