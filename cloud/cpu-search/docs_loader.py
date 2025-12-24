#!/usr/bin/env python3
"""
Document loader service for ClueWeb22-B documents from SQLite database.

This module provides a service for loading ClueWeb22 documents by ID,
designed to work with the search API router.
"""

import json
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from typing import Optional, List, Dict


class DocsLoader:
    """
    Document loader service for ClueWeb22-B documents.

    This service maintains a connection to a SQLite database and provides
    efficient batch document retrieval by ClueWeb22 IDs.
    """

    def __init__(self, db_path: str, use_compression: bool = False, num_threads: Optional[int] = None):
        """
        Initialize the document loader.

        Args:
            db_path: Path to SQLite database file
            use_compression: Whether the json_data is compressed with zstd
            num_threads: Number of threads for parallel decompression (default: CPU count)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        self.use_compression = use_compression
        self.num_threads = num_threads or int(multiprocessing.cpu_count() / 2)

        # Test connection
        self._test_connection()
        print(f"DocsLoader initialized with database: {db_path}")
        print(f"  - Compression: {use_compression}")
        print(f"  - Threads: {self.num_threads}")

    def _test_connection(self):
        """Test database connection."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(ClueWeb22_ID) FROM documents")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"  - Documents in database: {count:,}")

    def _decompress_and_parse(self, doc_id: str, json_data: bytes) -> tuple[str, Optional[Dict]]:
        """
        Decompress (if needed) and parse JSON data.

        Args:
            doc_id: Document ID
            json_data: Raw JSON data (possibly compressed)

        Returns:
            Tuple of (doc_id, parsed document dict)
        """
        try:
            if self.use_compression:
                import zstd
                json_str = zstd.decompress(json_data).decode('utf-8')
            elif isinstance(json_data, str):
                json_str = json_data
            else:
                raise ValueError("Unsupported json_data type, not zstd or str")

            doc = json.loads(json_str)
            return doc_id, doc
        except Exception as e:
            print(f"Error parsing document {doc_id}: {e}")
            return doc_id, None

    def load_docs_by_ids(self, query_ids: List[str]) -> List[Optional[Dict]]:
        """
        Load documents from SQLite database by ClueWeb22_IDs.

        Args:
            query_ids: List of ClueWeb22_IDs to retrieve

        Returns:
            List of document dictionaries (None for not found), preserving query_ids order
        """
        if not query_ids:
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Batch query with IN clause
        placeholders = ','.join('?' * len(query_ids))
        query = f"SELECT ClueWeb22_ID, json_data FROM documents WHERE ClueWeb22_ID IN ({placeholders})"
        cursor.execute(query, query_ids)

        # Fetch all results
        results = cursor.fetchall()
        conn.close()

        # Parallel decompression and parsing
        result_dict = {}
        if self.use_compression and len(results) > 1:
            # Use ThreadPoolExecutor for parallel decompression
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._decompress_and_parse, doc_id, json_data): doc_id
                    for doc_id, json_data in results
                }
                # Process results as they complete
                for future in as_completed(futures):
                    doc_id, doc = future.result()
                    if doc is not None:
                        result_dict[doc_id] = doc
        else:
            # Single-threaded for small batches or uncompressed data
            for doc_id, json_data in results:
                _, doc = self._decompress_and_parse(doc_id, json_data)
                if doc is not None:
                    result_dict[doc_id] = doc

        # Return docs in original query_ids order, None for not found
        return [result_dict.get(qid) for qid in query_ids]

    def get_doc_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Load a single document by ClueWeb22_ID.

        Args:
            doc_id: ClueWeb22_ID to retrieve

        Returns:
            Document dictionary or None if not found
        """
        docs = self.load_docs_by_ids([doc_id])
        return docs[0] if docs else None
