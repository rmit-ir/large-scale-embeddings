Plan: Streamlined Dataset Indexing Toolset (diskann_search/)

 Goal

 Create a complete toolset under diskann_search/ that streamlines adding new datasets to the search pipeline. The toolset is an orchestration layer -
  users provide:
 1. Data generator: Yields (id, content, metadata) records
 2. Embed function: Takes text batch, returns embeddings (can be API call or local model)

 The toolset handles batching, binary format conversion, index building, and output structure compatible with cloud/cpu-search.

 ---
 Design Principles

 1. Protocol-based: Users implement simple Python protocols
 2. No built-in models: User provides embed function (API or local)
 3. Streaming: Memory-efficient processing of large datasets
 4. Resumable: Checkpoint support for interrupted builds
 5. Compatible: Output works with existing cloud/cpu-search infrastructure

 ---
 Project Structure

 diskann_search/
     __init__.py

     # Core protocols and types
     protocols.py              # DataGenerator, EmbedFunction protocols
     types.py                  # IndexConfig, OutputConfig dataclasses

     # Pipeline
     pipeline.py               # Main build_index() orchestrator

     # Output writers
     output/
         __init__.py
         binary_writer.py      # DiskANN binary format (embeds.bin)
         docid_writer.py       # Document ID mapping (docids.pkl)
         sqlite_writer.py      # Document storage (documents.db)
         config_writer.py      # Metadata (config.json)

     # Index building
     index/
         __init__.py
         builder.py            # DiskANN build_disk_index wrapper

     # Utilities
     utils/
         __init__.py
         checkpoint.py         # Checkpoint/resume support
         progress.py           # Progress reporting

     # CLI
     cli.py                    # Optional CLI interface

     # Examples
     examples/
         fineweb.py            # FineWeb dataset example
         local_embed.py        # Local embedding example
         api_embed.py          # API embedding example

 ---
 Core Protocols

 diskann_search/protocols.py

 from typing import Protocol, Iterator, AsyncIterator, TypedDict, Any
 import numpy as np


 class DataRecord(TypedDict):
     """Single record from data source."""
     id: str                      # Unique document identifier
     content: str                 # Text to embed (user formats title/body/etc)
     metadata: dict[str, Any]     # Full document for storage/retrieval


 class DataGenerator(Protocol):
     """Sync data generator - user implements this."""
     def __iter__(self) -> Iterator[DataRecord]: ...
     def __len__(self) -> int: ...  # Optional, for progress bar


 class AsyncDataGenerator(Protocol):
     """Async data generator - for streaming from APIs/databases."""
     def __aiter__(self) -> AsyncIterator[DataRecord]: ...


 class EmbedFunction(Protocol):
     """
     Embedding function - user provides this.

     Can be:
     - Local model inference
     - API call to embedding service
     - Call to cloud/cpu-search embed router
     """
     async def __call__(self, texts: list[str]) -> np.ndarray:
         """
         Embed a batch of texts.

         Args:
             texts: List of text strings

         Returns:
             np.ndarray of shape (len(texts), embedding_dim)
         """
         ...

 ---
 Configuration Types

 diskann_search/types.py

 from dataclasses import dataclass, field
 from pathlib import Path
 from typing import Optional
 from enum import Enum


 class DistanceMetric(Enum):
     L2 = "l2"
     MIPS = "mips"
     COSINE = "cosine"


 @dataclass
 class IndexConfig:
     """DiskANN index building configuration."""
     metric: DistanceMetric = DistanceMetric.MIPS
     R: int = 64                    # Max node degree (60-150)
     L_build: int = 100             # Build complexity (>= R)
     build_memory_gb: int = 64      # RAM for building
     search_memory_gb: int = 24     # RAM for search
     build_threads: int = 32
     diskann_bin_path: Optional[Path] = None  # Path to DiskANN binaries


 @dataclass
 class OutputConfig:
     """Output writing configuration."""
     include_documents_db: bool = True   # Generate SQLite for doc retrieval
     checkpoint_interval: int = 10000    # Records between checkpoints


 @dataclass
 class BuildConfig:
     """Complete build configuration."""
     output_dir: Path
     embedding_dim: int                  # Must be known upfront
     index_config: IndexConfig = field(default_factory=IndexConfig)
     output_config: OutputConfig = field(default_factory=OutputConfig)
     batch_size: int = 256               # Batch size for embed function
     build_index: bool = True            # Whether to build DiskANN index
     resume: bool = True                 # Resume from checkpoint

 ---
 Main Pipeline

 diskann_search/pipeline.py

 async def build_index(
     data_generator: DataGenerator | AsyncDataGenerator,
     embed_fn: EmbedFunction,
     config: BuildConfig,
 ) -> Path:
     """
     Build search index from data generator.

     Args:
         data_generator: Yields DataRecord objects
         embed_fn: User-provided embedding function
         config: Build configuration

     Returns:
         Path to output directory

     Pipeline:
     1. Initialize output writers (binary, docids, sqlite)
     2. Load checkpoint if resuming
     3. Iterate data_generator in batches
     4. Call embed_fn for each batch
     5. Write embeddings to binary format
     6. Write docids to pickle
     7. Write documents to SQLite (optional)
     8. Save checkpoint periodically
     9. Build DiskANN index (optional)
     10. Write config.json
     """
     ...

 ---
 Output Format

 Compatible with cloud/cpu-search/router.py:

 output_dir/
     embeds.bin          # DiskANN binary: [num_vectors:u32][dim:u32][vectors:f32...]
     docids.pkl          # Pickle: list[str] mapping index → doc_id
     documents.db        # SQLite: doc_id → full document JSON (optional)
     config.json         # Metadata: num_docs, embedding_dim, model info
     index/              # DiskANN index files (after build)
         index_*

 ---
 Example Usage

 Adding FineWeb Dataset

 # fineweb_index.py
 import asyncio
 import numpy as np
 import httpx
 from pathlib import Path
 from datasets import load_dataset

 from diskann_search import (
     build_index,
     BuildConfig,
     IndexConfig,
     DataRecord,
 )


 # 1. Data Generator - yields records from FineWeb
 def fineweb_generator(max_records: int = 1_000_000):
     """Yield DataRecord from FineWeb-Edu."""
     dataset = load_dataset(
         "HuggingFaceFW/fineweb-edu",
         name="sample-10BT",
         split="train",
         streaming=True
     )

     for i, item in enumerate(dataset):
         if i >= max_records:
             break

         # Format content (user's responsibility)
         text = item['text']
         title = text.split('\n')[0][:200]

         yield DataRecord(
             id=item['id'],
             content=f"{title}\n\n{text}",  # User formats as needed
             metadata={
                 "id": item['id'],
                 "url": item.get('url', ''),
                 "text": text,
                 "score": item.get('score', 0),
             }
         )

     def __len__(self):
         return 1_000_000  # For progress bar


 # 2. Embed Function - calls existing embed router API
 async def embed_via_api(texts: list[str]) -> np.ndarray:
     """Call cloud/cpu-search embed router."""
     async with httpx.AsyncClient() as client:
         response = await client.post(
             "http://localhost:51003/embed",
             json={"input": texts, "model": "openbmb/MiniCPM-Embedding-Light"},
             timeout=60.0
         )
         data = response.json()
         embeddings = [item["embedding"] for item in data["data"]]
         return np.array(embeddings, dtype=np.float32)


 # 3. Alternative: Local embedding function
 async def embed_local(texts: list[str]) -> np.ndarray:
     """Run embedding locally with sentence-transformers."""
     from sentence_transformers import SentenceTransformer
     model = SentenceTransformer("all-MiniLM-L6-v2")
     embeddings = model.encode(texts, normalize_embeddings=True)
     return embeddings


 # 4. Build index
 async def main():
     config = BuildConfig(
         output_dir=Path("./data/fineweb_index"),
         embedding_dim=1024,  # MiniCPM-Embedding-Light dimension
         index_config=IndexConfig(
             R=100,
             L_build=150,
             build_memory_gb=64,
         ),
         batch_size=256,
     )

     await build_index(
         data_generator=fineweb_generator(max_records=1_000_000),
         embed_fn=embed_via_api,  # Or embed_local
         config=config,
     )

     print("Index built at:", config.output_dir)


 if __name__ == "__main__":
     asyncio.run(main())

 ---
 Integration with cloud/cpu-search

 After building, deploy with existing infrastructure:

 # 1. Start DiskANN search node
 uv run search_api/cw22_search_api/cw22_node_generic.py \
     --index-dir ./data/fineweb_index/index \
     --port 51001 \
     --dimensions 1024

 # 2. Start embed router (if using API embedding)
 PORT=51003 python cloud/cpu-search/router_embed.py

 # 3. Start search router
 DOC_ID_MAPPING_PATH=./data/fineweb_index/docids.pkl \
 DOC_DB_PATH=./data/fineweb_index/documents.db \
 PORT=51002 python cloud/cpu-search/router.py

 ---
 Implementation Steps

 Step 1: Core Types and Protocols

 - Create diskann_search/protocols.py with DataRecord, DataGenerator, EmbedFunction
 - Create diskann_search/types.py with config dataclasses

 Step 2: Output Writers

 - output/binary_writer.py: Streaming binary writer for embeds.bin
 - output/docid_writer.py: Pickle writer for docids.pkl
 - output/sqlite_writer.py: SQLite writer for documents.db
 - output/config_writer.py: JSON metadata writer

 Step 3: Pipeline Orchestrator

 - pipeline.py: Main build_index() function
 - Batch iteration over generator
 - Call embed_fn per batch
 - Write outputs
 - Progress reporting

 Step 4: Checkpoint Support

 - utils/checkpoint.py: Save/restore progress
 - Handle resume from partial builds

 Step 5: Index Builder

 - index/builder.py: Wrapper for DiskANN build_disk_index
 - Validate binary file
 - Run build with configured parameters

 Step 6: Examples

 - examples/fineweb.py: FineWeb dataset with API embedding
 - examples/local_embed.py: Local model embedding example

 ---
 Key Files to Reference
 ┌──────────────────────────────────────────────────┬────────────────────────────────────────────┐
 │                       File                       │                  Purpose                   │
 ├──────────────────────────────────────────────────┼────────────────────────────────────────────┤
 │ diskann/utils.py                                 │ Binary format spec (write_embed_to_binary) │
 ├──────────────────────────────────────────────────┼────────────────────────────────────────────┤
 │ cloud/cpu-search/router.py                       │ Output format consumer (docids, docs db)   │
 ├──────────────────────────────────────────────────┼────────────────────────────────────────────┤
 │ cloud/cpu-search/router_embed.py                 │ Embed API interface                        │
 ├──────────────────────────────────────────────────┼────────────────────────────────────────────┤
 │ tevatron/src/tevatron/retriever/driver/encode.py │ Checkpoint pattern reference               │
 └──────────────────────────────────────────────────┴────────────────────────────────────────────┘
 ---
 Verification Plan

 1. Unit tests: Test each output writer with small data
 2. Integration test: Build small index (1000 docs) end-to-end
 3. Compatibility test: Verify output works with cloud/cpu-search/router.py
 4. Resume test: Interrupt build, verify resume works correctly
 5. Search test: Run queries against built index, verify results