"""
ASE Dataset Tools - Tools for processing datasets for search indexing.

This package provides tools to:
- Write documents and embeddings to search-ready formats
- Build DiskANN indexes
- Manage SQLite databases for document storage
"""

from .output_writer import output_to_idx, OutputConfig
from .index_builder import build_index, DiskANNConfig
from .types import Document

__all__ = [
    'output_to_idx',
    'build_index',
    'Document',
    'OutputConfig',
    'DiskANNConfig',
]
