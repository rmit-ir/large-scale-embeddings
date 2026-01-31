"""
Index ClueWeb22-B sample dataset using ASEDISKS.

This script:
1. Reads documents from ClueWeb22-B sample data
2. Embeds them using MiniCPM-Embedding-Light locally (Transformers)
3. Builds a DiskANN index
4. Runs test searches

docker run --rm --gpus all \
    -v /home/ubuntu/projects/large-scale-embeddings:/app/large-scale-embeddings \
    -w /app/large-scale-embeddings \
    -e CLUEWEB_ROOT=/app/large-scale-embeddings/data/datasets/clueweb22-b \
    -e BATCH_SIZE=8 \
    -e MAX_WORDS=1024 \
    docker.io/rankun203/diskann-ase:latest \
    bash -lc "python3 index_clueweb_sample.py all" \
    | tee worklogs/clueweb22_full.4gpu.log
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
from asedisks.logging_utils import get_logger
from asedisks.utils import (
    truncate_first_n_words,
    MultiGPUEmbedder,
    EmbedderProtocol,
    resolve_embed_gpus,
)

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent

CLUEWEB_ROOT = Path(
    os.environ.get(
        "CLUEWEB_ROOT",
        "/home/ubuntu/projects/large-scale-embeddings/data/datasets/clueweb22-b/txt",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CLUEWEB_OUTPUT_DIR",
        str(REPO_ROOT / "data/asedisks_test/clueweb22-sample"),
    )
)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "openbmb/MiniCPM-Embedding-Light")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
MODEL_BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", '32'))
MAX_WORDS = int(os.environ.get("MAX_WORDS", "1024"))
EMBED_GPUS = os.environ.get("EMBED_GPUS", "auto")


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

        if self.device.startswith("cuda"):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }

        logger.info("Loading %s on %s...", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()

        logger.info("Model loaded.")

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
        embeddings, _ = self.model.encode_corpus(
            texts,
            batch_size=MODEL_BATCH_SIZE,
            return_sparse_vectors=False,
            show_progress_bar=False,
        )
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
        embeddings, _ = self.model.encode_query(
            texts,
            batch_size=MODEL_BATCH_SIZE,
            return_sparse_vectors=False,
            show_progress_bar=False,
        )
        return self._to_numpy(embeddings)


# Global embedder instance (loaded once)
_embedder: EmbedderProtocol | None = None


def get_embedder() -> EmbedderProtocol:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        gpu_ids = resolve_embed_gpus(EMBED_GPUS, logger=logger)
        if len(gpu_ids) > 1:
            _embedder = MultiGPUEmbedder(
                device_ids=gpu_ids,
                embedder_factory=lambda device_id: MiniCPMEmbedder(
                    model_name=MODEL_NAME, device=f"cuda:{device_id}"
                ),
            )
        elif len(gpu_ids) == 1:
            _embedder = MiniCPMEmbedder(device=f"cuda:{gpu_ids[0]}")
        else:
            _embedder = MiniCPMEmbedder()
    return _embedder


# ============================================================================
# Data Generator
# ============================================================================


async def clueweb_records(batch_size: int = BATCH_SIZE) -> AsyncIterator[list[DataRecord]]:
    """
    Async generator yielding batches of DataRecord from ClueWeb22-B sample.

    Yields:
        Batches of DataRecord objects
    """
    batch: list[DataRecord] = []

    json_files = sorted(CLUEWEB_ROOT.rglob("*.json.gz"))
    try:
        from tqdm import tqdm  # type: ignore

        file_iter = tqdm(
            json_files,
            unit=" file",
            desc="Files",
            leave=True,
            position=0,
            dynamic_ncols=True,
        )
    except Exception:
        file_iter = json_files

    for json_gz in file_iter:
        shard_name = json_gz.stem.replace(".json", "")

        with open(json_gz, "rb") as f_json:
            decompressed = gzip.decompress(f_json.read()).decode(
                "utf-8",
                errors="ignore",
            )

        lines = list(decompressed.splitlines())
        lines.sort(key=len, reverse=True)

        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                doc_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            fake_id = f"clueweb22-{shard_name}-{line_idx:05d}"
            clean_text = doc_data.get("Clean-Text", "")
            clueweb_id = doc_data.get("ClueWeb22-ID", fake_id)

            if not clean_text.strip():
                continue

            content = truncate_first_n_words(clean_text, MAX_WORDS)

            batch.append(
                DataRecord(
                    id=clueweb_id,
                    content=content,
                    metadata={
                        "id": clueweb_id,
                        "raw": doc_data,
                    },
                )
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch

    if hasattr(file_iter, "close"):
        file_iter.close() # type: ignore


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
    logger.info("=" * 60)
    logger.info("ASEDISKS - ClueWeb22-B Sample Indexing")
    logger.info("=" * 60)
    logger.info("Source: %s", CLUEWEB_ROOT)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Max docs: ALL")
    logger.info("Batch size: %s", BATCH_SIZE)
    logger.info("")

    if not CLUEWEB_ROOT.exists():
        raise FileNotFoundError(f"Dataset path not found: {CLUEWEB_ROOT}")

    # Initialize embedder (loads model)
    get_embedder()

    # Run indexing
    logger.info("Starting indexing...")
    await output_to_idx(
        output_dir=OUTPUT_DIR,
        records=clueweb_records(BATCH_SIZE),
        embed_fn=embed_batch,
        config=OutputConfig(
            batch_size=BATCH_SIZE,
            sqlite_compression=False,  # Keep it simple for testing
        ),
    )

    logger.info("Indexing complete!")
    logger.info("Output files in: %s", OUTPUT_DIR)


async def run_diskann_build():
    """Build DiskANN index."""
    logger.info("=" * 60)
    logger.info("Building DiskANN Index")
    logger.info("=" * 60)

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
    logger.info("Index build complete!")


async def run_search_test():
    """Run test searches."""
    logger.info("=" * 60)
    logger.info("Running Test Searches")
    logger.info("=" * 60)

    queries = [
        "machine learning artificial intelligence",
        "climate change global warming",
        "healthy food nutrition diet",
        "programming software development",
        "travel vacation destinations",
    ]

    logger.info("Testing %s queries...", len(queries))

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

    logger.info("-" * 60)
    for result in results:
        logger.info("Query: %s", result["query"])
        logger.info("Results:")
        for item in result["results"][:3]:
            doc = item.get("document", {})
            raw_text = ""
            if isinstance(doc, dict):
                raw = doc.get("raw", {})
                if isinstance(raw, dict):
                    raw_text = raw.get("Clean-Text", "")
            preview = " ".join(raw_text.split()[:10]) if raw_text else "N/A"
            logger.info("  [%s] %s", item["rank"], item["doc_id"])
            logger.info("      Score: %.4f", item["score"])
            logger.info("      Preview: %s", preview)

    logger.info("-" * 60)
    logger.info("Search test complete!")


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
            await run_diskann_build()
        elif command == "search":
            await run_search_test()
        elif command == "all":
            await run_indexing()
            await run_diskann_build()
            await run_search_test()
        else:
            logger.error("Unknown command: %s", command)
            logger.error("Usage: python index_clueweb_sample.py [index|build|search|all]")
    else:
        await run_indexing()
        await run_diskann_build()
        await run_search_test()


if __name__ == "__main__":
    asyncio.run(main())
