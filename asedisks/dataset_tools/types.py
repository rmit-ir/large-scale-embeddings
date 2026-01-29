"""
Data types for dataset processing.
"""

from typing import TypedDict, Any


class DataRecord(TypedDict):
    """
    Single record from data source.

    Users format all fields (title, body, etc.) into the `content` string.
    The `metadata` dict is stored in SQLite for retrieval.

    Args:
        id: Unique document identifier
        content: Text to embed (user combines title/body/etc)
        metadata: Full document data for storage/retrieval
    """

    id: str
    content: str
    metadata: dict[str, Any]
