"""
ASE Search - Generic search tools for indexed datasets.

This package provides tools to:
- Perform offline search on indexed datasets
- Run search router servers
- Load and query DiskANN indexes
- Retrieve documents from SQLite databases
"""

from .offline_search import search
from .server import run_search_router

__all__ = [
    'search',
    'run_search_router',
]
