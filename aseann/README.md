# ASEANN - ASE Disk Search

A flexible framework for processing datasets and building search indexes with customizable embedding functions.

## Overview

ASEANN provides tools to:

- Process any dataset format through customizable async batch generators
- Embed documents using any embedding service (API, local model, etc.)
- Build search-ready indexes (SQLite + DiskANN)
- Perform offline search
- Run search router servers with REST API

## Architecture

### Core Components

1. **dataset_tools**: Data processing and index building tools
2. **search**: Offline search and server tools

### Workflow

```
Your Data → Async Batch Generator → output_to_idx(records, embed_fn)
                                     ↓
                           Search-Ready Files:
                             - documents.db (SQLite)
                             - embeds.bin (DiskANN)
                             - docids.pkl
                             - config.json
                                     ↓
                           build_index()
                                     ↓
                           DiskANN Index
                                     ↓
                           Search (offline) or Server
```

## Quick Start

### 1. Index Your Dataset

Create `my_pipeline.py`:

```python
import asyncio
from pathlib import Path
import numpy as np
from dataset_tools import output_to_idx, build_index, DataRecord, OutputConfig

# Step 1: Define async batch data generator
async def my_data_generator(batch_size: int = 100):
    """Yield batches of DataRecord objects from your data source."""
    batch = []
    for item in my_data:  # Your data source
        batch.append(
            DataRecord(
                id=item["id"],
                content=f"{item['title']}\n{item['text']}",  # You format content
                metadata=item,  # Full document for storage
            )
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Step 2: Define async batch embed function
async def my_batch_embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts."""
    # Your embedding logic here (API call, local model, etc.)
    # Return np.array of shape (len(texts), embedding_dim)
    pass

# Step 3: Run pipeline (threaded writer + embedder pipeline)
async def main():
    await output_to_idx(
        output_dir=Path("data/my_dataset"),
        records=my_data_generator(batch_size=100),
        embed_fn=my_batch_embed,
        config=OutputConfig(batch_size=100),
    )
    build_index(
        binary_file=Path("data/my_dataset/embeds.bin"),
        output_dir=Path("data/my_dataset/index"),
    )

asyncio.run(main())
```

Run:
```bash
python my_pipeline.py
```

### 2. Perform Offline Search

Create `my_search.py`:

```python
import asyncio
from pathlib import Path
import numpy as np
from search import search

async def my_batch_embed(texts: list[str]) -> np.ndarray:
    """Embed queries (same as during indexing)."""
    # Your embedding logic here
    pass

async def main():
    results = await search(
        index_dir=Path("data/my_dataset"),
        queries=["What is machine learning?", "How do neural networks work?"],
        embed_fn=my_batch_embed,
        top_k=10,
        include_documents=True,
    )

    for result in results:
        print(f"Query: {result['query']}")
        for item in result['results']:
            print(f"  {item['rank']}: {item['doc_id']} ({item['score']:.4f})")

asyncio.run(main())
```

Run:
```bash
python my_search.py
```

### 3. Run Search Router Server

Create `my_server.py`:

```python
import asyncio
from pathlib import Path
import numpy as np
from search import run_search_router

async def my_embed(query: str) -> np.ndarray:
    """Embed single query (for inline embed node)."""
    # Your embedding logic here
    pass

async def main():
    await run_search_router(
        index_dir=Path("data/my_dataset"),
        host="0.0.0.0",
        port=8001,
        embed_fn=my_embed,
    )

asyncio.run(main())
```

Run:
```bash
python my_server.py
```

Query the server:
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "top_k": 10}'
```

## API Reference

### dataset_tools

#### `DataRecord`
```python
class DataRecord(TypedDict):
    id: str                    # Unique document identifier
    content: str               # Text to embed (user formats title/body/etc)
    metadata: dict[str, Any]   # Full document for storage
```

#### `output_to_idx()`
```python
async def output_to_idx(
    output_dir: Path,
    records: AsyncIterator[list[DataRecord]] | Iterator[list[DataRecord]],
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    config: Optional[OutputConfig] = None,
)
```

Runs a threaded pipeline for reading batches, embedding, and writing outputs to keep the GPU busy while SQLite and IO work runs in parallel.

Creates:
- `output_dir/documents.db`: SQLite database with full documents
- `output_dir/embeds.bin`: DiskANN binary format embeddings
- `output_dir/docids.pkl`: Document ID mappings
- `output_dir/config.json`: Metadata

#### `build_index()`
```python
def build_index(
    binary_file: Path,
    output_dir: Path,
    config: Optional[DiskANNConfig] = None,
)
```

Builds DiskANN index from binary file.

#### `OutputConfig`
```python
@dataclass
class OutputConfig:
    sqlite_compression: bool = True
    compression_level: int = 5
    batch_size: int = 50
    sqlite_cache_size_mb: int = 2000
    prefetch_batches: Optional[int] = None  # defaults to batch_size * num_gpus * 100
    sqlite_commit_every: Optional[int] = None  # defaults to batch_size * num_gpus * 100
```
Defaults for `prefetch_batches` and `sqlite_commit_every` are computed at runtime as `batch_size * num_gpus * 100`, where `num_gpus` is inferred from `EMBED_GPUS` or `CUDA_VISIBLE_DEVICES`.

#### `DiskANNConfig`
```python
@dataclass
class DiskANNConfig:
    metric: str = "mips"           # "l2", "mips", "cosine"
    R: int = 64                    # Max node degree (60-150)
    L: int = 100                   # Build complexity (>= R)
    build_threads: int = 32
    build_memory_gb: int = 64      # RAM for building
    search_memory_gb: int = 24     # RAM for search
    diskann_bin_path: Optional[Path] = None
```

### search

#### `search()`
```python
async def search(
    index_dir: Path,
    queries: list[str] | Path,
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    top_k: int = 10,
    complexity: int = 100,
    include_documents: bool = False,
    output_file: Optional[Path] = None,
) -> list[dict]
```

Performs offline search on indexed dataset.

#### `run_search_router()`
```python
async def run_search_router(
    index_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8001,
    embed_fn: Optional[Callable[[str], Awaitable[np.ndarray]]] = None,
)
```

Runs FastAPI search router server.

## Output Directory Structure

```
data/my_dataset/
├── documents.db          # SQLite: id -> metadata (full document)
├── embeds.bin            # DiskANN: [num_vectors][dim][vectors...]
├── docids.pkl            # List: [id1, id2, id3, ...]
├── config.json           # Metadata: num_docs, embedding_dim
└── index/                # DiskANN index
    ├── index_*           # Index files
    └── build_stats.json  # Build statistics
```

## Examples

- `example_index.py`: Complete indexing pipeline
- `example_search_ds.py`: Offline search example
- `example_router.py`: Search router server example

## Features

- Async-first design for I/O efficiency
- Streaming processing (minimal memory usage)
- Flexible embedding (API, local model, custom)
- Optimized for massive scale (billions of documents)
- Compatible with existing cloud/cpu-search infrastructure

## Requirements

- Python 3.10+
- numpy
- sqlite3
- Optional: diskannpy (for search)
- Optional: fastapi, uvicorn (for server)
- Optional: httpx (for API embedding)

## Design Philosophy

ASEANN provides **tools, not solutions**. You write:
- Async data generator for your dataset
- Async embed function for your model

We handle:
- Efficient batch processing
- SQLite optimization
- DiskANN format compatibility
- Search infrastructure

## Project Structure

```
aseann/
├── dataset_tools/
│   ├── __init__.py
│   ├── types.py           # DataRecord TypedDict
│   ├── output_writer.py   # output_to_idx()
│   └── index_builder.py   # build_index()
├── search/
│   ├── __init__.py
│   ├── offline_search.py  # search()
│   └── server.py          # run_search_router()
├── example_index.py
├── example_search_ds.py
└── example_router.py
```

## Integration with cloud/cpu-search

After building an index with ASEANN, you can deploy it with the existing `cloud/cpu-search` infrastructure:

```bash
# 1. Start DiskANN search node
uv run search_api/cw22_search_api/cw22_node_generic.py \
    --index-dir ./data/my_dataset/index \
    --port 51001 \
    --dimensions YOUR_EMBEDDING_DIM

# 2. Start embed router (if using external embedding)
PORT=51003 python cloud/cpu-search/router_embed.py

# 3. Start search router
DOC_ID_MAPPING_PATH=./data/my_dataset/docids.pkl \
DOC_DB_PATH=./data/my_dataset/documents.db \
PORT=51002 python cloud/cpu-search/router.py
```
