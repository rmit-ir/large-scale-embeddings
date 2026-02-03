"""
Output writer for creating search-ready files.

Creates:
- SQLite database with full documents
- DiskANN binary format files (embeds.bin, docids.pkl)
- Config.json with metadata
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Callable, Awaitable, Union
import asyncio
import json
import pickle
import queue
import sqlite3
import threading
import numpy as np

from .types import DataRecord
from aseann.utils import ensure_async
from aseann.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OutputConfig:
    """Configuration for output writing."""

    sqlite_compression: bool = True
    compression_level: int = 5
    batch_size: int = 50
    sqlite_cache_size_mb: int = 2000
    prefetch_batches: Optional[int] = None
    sqlite_commit_every: Optional[int] = None


def _infer_num_gpus() -> int:
    embed_gpus = os.environ.get("EMBED_GPUS")
    if embed_gpus is not None:
        normalized = embed_gpus.strip().lower()
        if normalized in ("", "auto", "all"):
            pass
        elif normalized in ("cpu", "none", "off"):
            return 0
        else:
            return len([part for part in embed_gpus.split(",") if part.strip()])

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        normalized = cuda_visible.strip()
        if normalized == "" or normalized.lower() in ("none", "void", "off", "cpu"):
            return 0
        return len([part for part in normalized.split(",") if part.strip()])

    return 1


def _apply_output_defaults(config: OutputConfig) -> None:
    if config.prefetch_batches is None or config.sqlite_commit_every is None:
        num_gpus = _infer_num_gpus()
        default_value = config.batch_size * num_gpus * 100
        if config.prefetch_batches is None:
            config.prefetch_batches = default_value
        if config.sqlite_commit_every is None:
            config.sqlite_commit_every = default_value


async def output_to_idx(
    output_dir: Path,
    records: Union[AsyncIterator[list[DataRecord]], Iterator[list[DataRecord]]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    config: Optional[OutputConfig] = None,
):
    """
    Write records and embeddings to search-ready files.

    This function processes records asynchronously, calls the embedding function
    for each batch, and writes outputs:
    1. SQLite database (documents.db) with full documents
    2. DiskANN binary file (embeds.bin) with embeddings
    3. Document ID mappings (docids.pkl)
    4. Metadata config (config.json)

    Args:
        output_dir: Directory to write output files
        records: Async or sync iterator of DataRecord batches
        embed_fn: Async function that embeds a batch of texts.
                  Signature: async def embed_fn(texts: list[str]) -> np.ndarray, shape (len(texts), 1024)
        config: Output configuration options

    The function handles:
    - SQLite database creation with optimized settings
    - Streaming write to minimize memory usage
    - Progress tracking

    Output files:
        output_dir/documents.db: SQLite database with documents
        output_dir/embeds.bin: DiskANN binary format embeddings
        output_dir/docids.pkl: Pickled list of document IDs
        output_dir/config.json: Metadata (num_docs, embedding_dim, etc.)
    """
    config = config or OutputConfig()
    _apply_output_defaults(config)
    await asyncio.to_thread(
        _output_to_idx_threaded,
        output_dir,
        records,
        embed_fn,
        config,
    )


def _output_to_idx_threaded(
    output_dir: Path,
    records: Union[AsyncIterator[list[DataRecord]], Iterator[list[DataRecord]]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    config: OutputConfig,
):
    _apply_output_defaults(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_queue: queue.Queue[object] = queue.Queue(maxsize=max(1, config.prefetch_batches))
    write_queue: queue.Queue[object] = queue.Queue(maxsize=max(1, config.prefetch_batches))
    error_queue: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    sentinel = object()
    def _queue_put(q: queue.Queue[object], item: object) -> bool:
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _queue_get(q: queue.Queue[object]) -> object:
        while True:
            if stop_event.is_set() and q.empty():
                return sentinel
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                continue

    def _signal_error(exc: BaseException):
        if not stop_event.is_set():
            stop_event.set()
            error_queue.put(exc)
            _queue_put(read_queue, sentinel)
            _queue_put(write_queue, sentinel)

    def _run_reader():
        try:
            async def _read():
                async for batch in ensure_async(records):
                    if stop_event.is_set():
                        break
                    if batch:
                        if not _queue_put(read_queue, batch):
                            break
                _queue_put(read_queue, sentinel)

            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_read())
            finally:
                loop.close()
        except BaseException as exc:
            _signal_error(exc)

    def _run_embedder():
        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                while True:
                    item = _queue_get(read_queue)
                    if item is sentinel:
                        break
                    batch = item  # type: ignore[assignment]
                    texts = [r["content"] for r in batch]  # type: ignore[index]
                    embeddings = loop.run_until_complete(embed_fn(texts))
                    if not _queue_put(write_queue, (batch, embeddings)):
                        break
                _queue_put(write_queue, sentinel)
            finally:
                loop.close()
        except BaseException as exc:
            _signal_error(exc)

    def _run_writer():
        try:
            embedding_dim: Optional[int] = None
            num_records = 0
            doc_ids: list[str] = []

            embeds_path = output_dir / "embeds.bin"
            docids_path = output_dir / "docids.pkl"
            db_path = output_dir / "documents.db"
            config_path = output_dir / "config.json"

            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA cache_size=-{config.sqlite_cache_size_mb * 1024}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    json_data TEXT NOT NULL
                )
            """
            )
            conn.commit()

            pbar = None
            try:
                from tqdm import tqdm  # type: ignore

                pbar = tqdm(
                    total=None,
                    unit=" records",
                    desc="Embedding",
                    leave=True,
                    position=1,
                    dynamic_ncols=True,
                )
            except Exception:
                pbar = None

            batches_since_commit = 0
            with open(embeds_path, "wb") as embeds_file:
                embeds_file.write(np.uint32(0).tobytes())
                embeds_file.write(np.uint32(0).tobytes())

                while True:
                    item = _queue_get(write_queue)
                    if item is sentinel:
                        break
                    batch, embeddings = item  # type: ignore[misc]

                    if embedding_dim is None:
                        embedding_dim = embeddings.shape[1]
                    elif embeddings.shape[1] != embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {embedding_dim}, "
                            f"got {embeddings.shape[1]}"
                        )

                    embeds_file.write(embeddings.astype(np.float32).tobytes())

                    cursor = conn.cursor()
                    for record in batch:
                        doc_ids.append(record["id"])
                        json_data = json.dumps(record["metadata"])
                        cursor.execute(
                            "INSERT OR REPLACE INTO documents (doc_id, json_data) VALUES (?, ?)",
                            (record["id"], json_data),
                        )
                    batches_since_commit += 1
                    if config.sqlite_commit_every <= 1 or (
                        batches_since_commit >= config.sqlite_commit_every
                    ):
                        conn.commit()
                        batches_since_commit = 0

                    num_records += len(batch)
                    if pbar is not None:
                        pbar.update(len(batch))
                        pbar.set_postfix(
                            read_q=read_queue.qsize(),
                            write_q=write_queue.qsize(),
                        )
                    else:
                        logger.info("Processed %s records...", f"{num_records:,}")

                embeds_file.seek(0)
                embeds_file.write(np.uint32(num_records).tobytes())
                embeds_file.write(np.uint32(embedding_dim or 0).tobytes())

            if batches_since_commit:
                conn.commit()
            conn.close()
            if pbar is not None:
                pbar.close()

            with open(docids_path, "wb") as f:
                pickle.dump(doc_ids, f)

            with open(config_path, "w") as f:
                json.dump(
                    {
                        "num_docs": num_records,
                        "embedding_dim": embedding_dim,
                    },
                    f,
                    indent=2,
                )

            logger.info("Output written to %s", output_dir)
            logger.info("  - %s documents", f"{num_records:,}")
            logger.info("  - %s dimensions", embedding_dim)
        except BaseException as exc:
            _signal_error(exc)

    reader_thread = threading.Thread(target=_run_reader, name="ase-reader", daemon=True)
    embed_thread = threading.Thread(target=_run_embedder, name="ase-embed", daemon=True)
    writer_thread = threading.Thread(target=_run_writer, name="ase-writer", daemon=True)

    reader_thread.start()
    embed_thread.start()
    writer_thread.start()

    reader_thread.join()
    embed_thread.join()
    writer_thread.join()

    if not error_queue.empty():
        raise error_queue.get()
