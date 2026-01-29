# ASE Dataset Search (ASEEDS)

A flexible framework for processing datasets and building search indexes with customizable embedding functions.

## Overview

ASEEDS provides tools to:

- Process any dataset format through customizable async generators
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
Your Data → Async Generator → Embeddings → output_to_idx()
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
from dataset_tools import output_to_idx, build_index, Document, OutputConfig

# Step 1: Define async data generator
async def my_data_generator():
    """Yield Document objects from your data source."""
    for file_path in Path("my_data/*.json").glob("*.json"):
        data = load_document(file_path)
        yield Document(
            id=data["id"],
            title=data["title"],
            body=data["body"],
            whole_doc=data,
            url=data.get("url"),
        )

# Step 2: Define async batch embed function
async def my_batch_embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts."""
    # Your embedding logic here
    # Return np.array of shape (len(texts), embedding_dim)
    pass

async def my_embed_async_generator(documents):
    """Wrapper to embed documents."""
    batch = []
    async for doc in documents:
        batch.append(f"{doc.title} {doc.body}")
        if len(batch) >= 100:
            yield await my_batch_embed(batch)
            batch = []
    if batch:
        yield await my_batch_embed(batch)

# Step 3: Run pipeline
async def main():
    await output_to_idx(
        output_dir=Path("data/my_dataset"),
        documents=my_data_generator(),
        embeddings=my_embed_async_generator(my_data_generator()),
        config=OutputConfig(sqlite_compression=True, batch_size=1000),
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
    async for result in search(
        index_dir=Path("data/my_dataset"),
        queries_file=Path("queries.txt"),
        embed_fn=my_batch_embed,
        top_k=10,
    ):
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
        embed_fn=my_embed,  # Or None for external embed node
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

#### `Document`
```python
@dataclass
class Document:
    id: str
    title: str
    body: str
    whole_doc: dict
    url: str | None = None
    fetched_at: str | None = None
```

#### `output_to_idx()`
```python
async def output_to_idx(
    output_dir: Path,
    documents: AsyncIterator[Document],
    embeddings: AsyncIterator[np.ndarray],
    config: Optional[OutputConfig] = None,
)
```

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
    batch_size: int = 1000
    sqlite_cache_size_mb: int = 20000
```

#### `DiskANNConfig`
```python
@dataclass
class DiskANNConfig:
    metric: str = "L2"
    index_build_threads: int = 32
    R: int = 64
    L: int = 100
    search_window_size: int = 100
```

### search

#### `search()`
```python
async def search(
    index_dir: Path,
    queries_file: Path,
    embed_fn: Callable[[list[str]], Awaitable[np.ndarray]],
    top_k: int = 10,
    output_file: Optional[Path] = None,
) -> AsyncIterator[dict]
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
├── documents.db          # SQLite: id -> whole_doc (compressed)
├── embeds.bin            # DiskANN: embeddings
├── docids.pkl            # List: [id1, id2, id3, ...]
├── config.json           # Metadata: dim, num_docs, etc.
└── index/                # DiskANN index
    ├── index.bin
    └── index_metadata.bin
```

## Examples

- `example_index.py`: Complete indexing pipeline
- `example_search_ds.py`: Offline search example
- `example_router.py`: Search router server example

## Features

- ✅ Async-first design for I/O efficiency
- ✅ Streaming processing (minimal memory usage)
- ✅ Python 3.14 free threading support
- ✅ Flexible embedding (API, local model, custom)
- ✅ Optimized for massive scale (billions of documents)
- ✅ Zero external dependencies (except what you need)

## Requirements

- Python 3.10+
- numpy
- sqlite3
- Optional: zstd (for compression)
- Optional: DiskANN (for index building)

## Design Philosophy

ASEEDS provides **tools, not solutions**. You write:
- Async data generator for your dataset
- Async embed function for your model
- Pipeline orchestration

We handle:
- Efficient multi-threaded processing
- SQLite optimization
- DiskANN format compatibility
- Search infrastructure

## Project Structure

```
aseeds/
├── dataset_tools/
│   ├── __init__.py
│   ├── types.py
│   ├── output_writer.py
│   └── index_builder.py
├── search/
│   ├── __init__.py
│   ├── offline_search.py
│   └── server.py
├── example_index.py
├── example_search_ds.py
└── example_router.py
```
