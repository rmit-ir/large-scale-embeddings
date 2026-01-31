"""
Shared utilities for dataset pipelines and embedding workers.

Simple usage examples:
    from aseann.utils import (
        ensure_async,
        split_even,
        truncate_first_n_words,
        resolve_embed_gpus,
        MultiGPUEmbedder,
        EmbedderProtocol,
    )

    # Turn a sync iterator into an async iterator
    async for batch in ensure_async(my_batches):
        ...

    # Split a list into even chunks
    chunks = split_even(texts, parts=4)

    # Truncate to N words
    text = truncate_first_n_words(text, max_words=256)

    # Resolve GPU ids from env / config
    gpu_ids = resolve_embed_gpus("0,1")     # -> [0, 1]
    gpu_ids = resolve_embed_gpus("auto")    # -> [0..N-1] if CUDA
    gpu_ids = resolve_embed_gpus("cpu")     # -> []

    # Multi-GPU embedding with a factory
    mg = MultiGPUEmbedder(
        device_ids=gpu_ids,
        embedder_factory=lambda device_id: MyEmbedder(device=f"cuda:{device_id}"),
    )
    embeddings = mg.encode_passages(texts)
"""

from typing import AsyncIterator, Iterator, Protocol, TypeVar, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

T = TypeVar("T")


async def ensure_async(iterable: Union[AsyncIterator[T], Iterator[T]]) -> AsyncIterator[T]:
    """Yield from a sync or async iterator as an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:  # type: ignore
            yield item
    else:
        for item in iterable:  # type: ignore
            yield item


def split_even(items: list[T], parts: int) -> list[list[T]]:
    """Split a list into roughly even chunks.

    Example:
        split_even([1, 2, 3, 4, 5], parts=2) -> [[1, 2, 3], [4, 5]]
    """
    if parts <= 0:
        return [items]
    total = len(items)
    if total == 0:
        return []
    base = total // parts
    remainder = total % parts
    chunks: list[list[T]] = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < remainder else 0)
        if size == 0:
            continue
        end = start + size
        chunks.append(items[start:end])
        start = end
    return chunks


def truncate_first_n_words(text: str, max_words: int) -> str:
    """Truncate a string to the first N words (space-delimited).

    Example:
        truncate_first_n_words("a b c d", max_words=3) -> "a b c"
    """
    if not text:
        return text
    if max_words <= 0:
        return text
    count = 0
    for idx, ch in enumerate(text):
        if ch == " ":
            count += 1
            if count == max_words - 1:
                return text[:idx]
    return text


class EmbedderProtocol(Protocol):
    """Protocol for embedders used by MultiGPUEmbedder."""
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        ...

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        ...


class MultiGPUEmbedder:
    """Run embedding across multiple GPUs in parallel (one model per GPU).

    Example:
        mg = MultiGPUEmbedder(
            device_ids=[0, 1],
            embedder_factory=lambda device_id: MyEmbedder(device=f"cuda:{device_id}"),
        )
    """

    def __init__(
        self,
        device_ids: list[int],
        embedder_factory: Callable[[int], EmbedderProtocol],
    ):
        if not device_ids:
            raise ValueError("MultiGPUEmbedder requires at least one device id.")
        self.device_ids = device_ids
        self.embedders = [embedder_factory(device_id) for device_id in device_ids]
        self.pool = ThreadPoolExecutor(max_workers=len(self.embedders))

    def _encode_chunks(self, texts: list[str], encoder_name: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        chunks = split_even(texts, len(self.embedders))
        if not chunks:
            return np.zeros((0, 0), dtype=np.float32)

        futures = []
        for embedder, chunk in zip(self.embedders, chunks):
            encoder = getattr(embedder, encoder_name)
            futures.append(self.pool.submit(encoder, chunk))

        results = [future.result() for future in futures]
        return np.vstack(results)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode_chunks(texts, "encode_passages")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_chunks(texts, "encode_queries")


def resolve_embed_gpus(value: str, logger=None) -> list[int]:
    """Resolve GPU ids from a string like 'auto', 'all', '0,1', or 'cpu'.

    Example:
        resolve_embed_gpus("0,1") -> [0, 1]
        resolve_embed_gpus("auto") -> [0..N-1] if CUDA, else []
        resolve_embed_gpus("cpu") -> []
    """
    normalized = value.strip().lower()
    if normalized in ("", "auto", "all"):
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    if normalized in ("cpu", "none", "off"):
        return []

    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))

    if not torch.cuda.is_available():
        if logger is not None:
            logger.warning("EMBED_GPUS=%s but CUDA is unavailable. Falling back to CPU.", value)
        return []

    max_id = torch.cuda.device_count() - 1
    invalid = [gpu_id for gpu_id in ids if gpu_id < 0 or gpu_id > max_id]
    if invalid:
        raise ValueError(
            f"EMBED_GPUS includes invalid CUDA device ids {invalid}. "
            f"Available range: 0..{max_id}"
        )
    return ids
