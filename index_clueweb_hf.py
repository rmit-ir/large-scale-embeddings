"""
Index ClueWeb22-B sample dataset using Hugging Face embedding models (SentenceTransformer).

Default model: Qwen/Qwen3-Embedding-0.6B

Notes for Qwen/Qwen3-Embedding-0.6B:
  - SentenceTransformer usage with prompt_name="query" for query embeddings.
  - Recommended (optional) flash_attention_2 with left padding.

Example:
docker run --rm --gpus all \
    -v /home/ubuntu/projects/large-scale-embeddings:/app/large-scale-embeddings \
    -w /app/large-scale-embeddings \
    -e CLUEWEB_ROOT=/app/large-scale-embeddings/data/datasets/clueweb22-b \
    -e EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    -e QUERY_PROMPT_NAME=query \
    -e BATCH_SIZE=8 \
    -e MAX_WORDS=1024 \
    docker.io/rankun203/diskann-ase:latest \
    bash -lc "python3 index_clueweb_hf.py all" \
    | tee worklogs/clueweb22_hf_qwen3_0.6.log
"""

import asyncio
import gzip
import json
import os
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch

from aseann.dataset_tools import (
    output_to_idx,
    build_index,
    DataRecord,
    OutputConfig,
    DiskANNConfig,
)
from aseann.search import search
from aseann.logging_utils import get_logger
from aseann.utils import (
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
        str(REPO_ROOT / "data/aseann_test/clueweb22-sample-hf"),
    )
)

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MODEL_BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
MAX_WORDS = int(os.environ.get("MAX_WORDS", "1024"))
EMBED_GPUS = os.environ.get("EMBED_GPUS", "auto")

QUERY_PROMPT_NAME = os.environ.get("QUERY_PROMPT_NAME", "query")
QUERY_PROMPT = os.environ.get("QUERY_PROMPT", "")
USE_FLASH_ATTENTION = os.environ.get("USE_FLASH_ATTENTION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
TOKENIZER_PADDING_SIDE = os.environ.get("TOKENIZER_PADDING_SIDE", "").strip()
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}


# ============================================================================
# Embedding Model
# ============================================================================


class SentenceTransformerEmbedder:
    """
    SentenceTransformer-based embedder for Hugging Face models.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "auto"):
        from sentence_transformers import SentenceTransformer

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

        model_kwargs = {"torch_dtype": self.dtype}
        tokenizer_kwargs: dict[str, str] = {}

        if USE_FLASH_ATTENTION and self.device.startswith("cuda"):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            if not TOKENIZER_PADDING_SIDE:
                tokenizer_kwargs["padding_side"] = "left"

        if TOKENIZER_PADDING_SIDE:
            tokenizer_kwargs["padding_side"] = TOKENIZER_PADDING_SIDE

        st_kwargs = {
            "device": self.device,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "model_kwargs": model_kwargs,
        }
        if tokenizer_kwargs:
            st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

        logger.info("Loading %s on %s...", model_name, self.device)
        self.model = SentenceTransformer(model_name, **st_kwargs)
        self.model.eval()
        logger.info("Model loaded.")

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        encode_kwargs = {
            "batch_size": MODEL_BATCH_SIZE,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }

        if is_query:
            prompt_text = QUERY_PROMPT.strip()
            prompt_name = QUERY_PROMPT_NAME.strip()
            if prompt_text:
                encode_kwargs["prompt"] = prompt_text
            elif prompt_name:
                prompts = getattr(self.model, "prompts", None)
                if isinstance(prompts, dict) and prompt_name in prompts:
                    encode_kwargs["prompt_name"] = prompt_name
                else:
                    available = list(prompts.keys()) if isinstance(prompts, dict) else "n/a"
                    logger.warning(
                        "QUERY_PROMPT_NAME=%s not found in model.prompts (available=%s). "
                        "Encoding queries without a prompt.",
                        prompt_name,
                        available,
                    )

        embeddings = self.model.encode(texts, **encode_kwargs)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)


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
                embedder_factory=lambda device_id: SentenceTransformerEmbedder(
                    model_name=MODEL_NAME,
                    device=f"cuda:{device_id}",
                ),
            )
        elif len(gpu_ids) == 1:
            _embedder = SentenceTransformerEmbedder(
                model_name=MODEL_NAME,
                device=f"cuda:{gpu_ids[0]}",
            )
        else:
            _embedder = SentenceTransformerEmbedder(model_name=MODEL_NAME)
    return _embedder


# ============================================================================
# Data Generator
# ============================================================================


async def clueweb_records(batch_size: int = BATCH_SIZE) -> AsyncIterator[list[DataRecord]]:
    """
    Async generator yielding batches of DataRecord from ClueWeb22-B sample.
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

        with gzip.open(json_gz, "rt", encoding="utf-8", errors="ignore") as f_json:
            for line_idx, line in enumerate(f_json):
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
        file_iter.close()  # type: ignore


# ============================================================================
# Embedding Functions
# ============================================================================


async def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts using SentenceTransformer."""
    embedder = get_embedder()
    return embedder.encode_passages(texts)


async def embed_queries_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of queries using SentenceTransformer."""
    embedder = get_embedder()
    return embedder.encode_queries(texts)


# ============================================================================
# Main Pipeline
# ============================================================================


async def run_indexing():
    """Run the indexing pipeline."""
    logger.info("=" * 60)
    logger.info("ASEANN - ClueWeb22-B Sample Indexing (HF SentenceTransformer)")
    logger.info("=" * 60)
    logger.info("Source: %s", CLUEWEB_ROOT)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Embedding model: %s", MODEL_NAME)
    logger.info("Query prompt name: %s", QUERY_PROMPT_NAME or "(none)")
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
            sqlite_compression=False,
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
        elif command == "build-index":
            await run_indexing()
            await run_diskann_build()
        elif command == "search":
            await run_search_test()
        elif command == "all":
            await run_indexing()
            await run_diskann_build()
            await run_search_test()
        else:
            logger.error("Unknown command: %s", command)
            logger.error("Usage: python index_clueweb_hf.py [index|build|build-index|search|all]")
    else:
        await run_indexing()
        await run_diskann_build()
        await run_search_test()


if __name__ == "__main__":
    asyncio.run(main())
