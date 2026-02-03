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
import multiprocessing as mp
import pickle
import queue
import sqlite3
import numpy as np

from .types import DataRecord
from aseann.utils import ensure_async, resolve_embed_gpus
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
        num_workers = _infer_num_gpus()
        if num_workers == 0:
            cpu_workers = int(os.environ.get("EMBED_CPU_WORKERS", "1"))
            if cpu_workers < 1:
                cpu_workers = 1
            num_workers = cpu_workers
        default_value = config.batch_size * num_workers * 100
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

    Note:
        When multiple workers are used, batches are embedded in parallel and written
        as they complete. The output order will not necessarily match input order,
        but docids.pkl always aligns with embeds.bin.
    """
    config = config or OutputConfig()
    _apply_output_defaults(config)
    device_ids = resolve_embed_gpus(os.environ.get("EMBED_GPUS", "auto"), logger=logger)

    await _output_to_idx_multiprocess(
        output_dir=output_dir,
        records=records,
        embed_fn=embed_fn,
        config=config,
        device_ids=device_ids,
    )


def _device_id_to_cuda_visible(device_id: int) -> str:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None or cuda_visible.strip() == "":
        return str(device_id)
    parts = [part.strip() for part in cuda_visible.split(",") if part.strip()]
    if device_id < 0 or device_id >= len(parts):
        raise ValueError(
            f"Resolved device id {device_id} is outside CUDA_VISIBLE_DEVICES={cuda_visible}."
        )
    return parts[device_id]


def _queue_put_mp(
    q: mp.Queue,
    item: object,
    stop_event: mp.Event,
    timeout: float = 0.5,
) -> bool:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _signal_error(exc: BaseException, error_queue: mp.Queue, stop_event: mp.Event) -> None:
    try:
        error_queue.put(exc)
    except Exception:
        pass
    stop_event.set()


def _run_embed_worker(
    worker_id: int,
    cuda_visible: str,
    embed_gpus_value: str,
    read_queue: mp.Queue,
    write_queue: mp.Queue,
    error_queue: mp.Queue,
    stop_event: mp.Event,
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        os.environ["EMBED_GPUS"] = embed_gpus_value

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            while True:
                try:
                    item = read_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if item is None:
                    break
                batch = item  # type: ignore[assignment]
                texts = [record["content"] for record in batch]  # type: ignore[index]
                embeddings = loop.run_until_complete(embed_fn(texts))
                if not _queue_put_mp(write_queue, (batch, embeddings), stop_event):
                    break
        finally:
            loop.close()
    except BaseException as exc:
        _signal_error(exc, error_queue, stop_event)
    finally:
        _queue_put_mp(write_queue, None, stop_event)


def _run_writer_process(
    output_dir: Path,
    config: OutputConfig,
    write_queue: mp.Queue,
    worker_queues: list[mp.Queue],
    error_queue: mp.Queue,
    stop_event: mp.Event,
    num_workers: int,
):
    try:
        embedding_dim: Optional[int] = None
        num_records = 0
        doc_ids: list[str] = []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        sentinels = 0

        with open(embeds_path, "wb") as embeds_file:
            embeds_file.write(np.uint32(0).tobytes())
            embeds_file.write(np.uint32(0).tobytes())

            while True:
                try:
                    item = write_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if item is None:
                    sentinels += 1
                    if sentinels >= num_workers:
                        break
                    continue

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
                    queue_postfix = {
                        f"q{idx}": q.qsize() for idx, q in enumerate(worker_queues)
                    }
                    queue_postfix["wq"] = write_queue.qsize()
                    pbar.set_postfix(**queue_postfix)
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
        _signal_error(exc, error_queue, stop_event)


def _select_queue_index(queues: list[mp.Queue], rr_index: int) -> int:
    sizes = [q.qsize() for q in queues]
    min_size = min(sizes)
    start = rr_index % len(queues)
    for offset in range(len(queues)):
        idx = (start + offset) % len(queues)
        if sizes[idx] == min_size:
            return idx
    return start


async def _output_to_idx_multiprocess(
    output_dir: Path,
    records: Union[AsyncIterator[list[DataRecord]], Iterator[list[DataRecord]]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    config: OutputConfig,
    device_ids: list[int],
):
    _apply_output_defaults(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device_ids:
        ctx = mp.get_context("spawn")
    else:
        ctx = (
            mp.get_context("fork")
            if "fork" in mp.get_all_start_methods()
            else mp.get_context()
        )

    if device_ids:
        worker_specs = [
            {
                "name": f"ase-embed-{device_id}",
                "cuda_visible": _device_id_to_cuda_visible(device_id),
                "embed_gpus": "0",
            }
            for device_id in device_ids
        ]
    else:
        cpu_workers = int(os.environ.get("EMBED_CPU_WORKERS", "1"))
        if cpu_workers < 1:
            cpu_workers = 1
        worker_specs = [
            {
                "name": f"ase-embed-cpu-{worker_idx}",
                "cuda_visible": "",
                "embed_gpus": "cpu",
            }
            for worker_idx in range(cpu_workers)
        ]

    num_workers = len(worker_specs)
    gpu_queues = [ctx.Queue(maxsize=100) for _ in range(num_workers)]
    write_queue: mp.Queue = ctx.Queue(maxsize=num_workers * 100)
    error_queue: mp.Queue = ctx.Queue()
    stop_event: mp.Event = ctx.Event()

    writer_proc = ctx.Process(
        target=_run_writer_process,
        args=(
            output_dir,
            config,
            write_queue,
            gpu_queues,
            error_queue,
            stop_event,
            num_workers,
        ),
        name="ase-writer",
    )
    writer_proc.start()

    worker_procs = []
    for idx, spec in enumerate(worker_specs):
        proc = ctx.Process(
            target=_run_embed_worker,
            args=(
                idx,
                spec["cuda_visible"],
                spec["embed_gpus"],
                gpu_queues[idx],
                write_queue,
                error_queue,
                stop_event,
                embed_fn,
            ),
            name=spec["name"],
        )
        proc.start()
        worker_procs.append(proc)

    exc_to_raise: Optional[BaseException] = None
    rr_index = 0
    try:
        async for batch in ensure_async(records):
            if stop_event.is_set():
                break
            if not batch:
                continue

            try:
                err = error_queue.get_nowait()
            except queue.Empty:
                err = None
            if err is not None:
                exc_to_raise = err
                stop_event.set()
                break

            queue_idx = _select_queue_index(gpu_queues, rr_index)
            rr_index = (queue_idx + 1) % num_workers
            if not _queue_put_mp(gpu_queues[queue_idx], batch, stop_event):
                break
    except BaseException as exc:
        exc_to_raise = exc
        stop_event.set()
    finally:
        for q in gpu_queues:
            _queue_put_mp(q, None, stop_event)

        for proc in worker_procs:
            proc.join()

        writer_proc.join()

        if exc_to_raise is None:
            try:
                exc_to_raise = error_queue.get_nowait()
            except queue.Empty:
                exc_to_raise = None

        if exc_to_raise is not None:
            raise exc_to_raise
