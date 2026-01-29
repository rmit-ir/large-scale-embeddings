"""
Data types for dataset processing.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

# TODO: use TypedDict instead
@dataclass
class Document:
    """
    Document representation for search indexing.
    
    Args:
        id: Unique document identifier (required)
        title: Document title (required)
        body: Document body text (required)
        whole_doc: Full document as dictionary (required, for SQLite storage)
        url: Document URL (optional)
        fetched_at: Fetch timestamp (optional)
        **kwargs: Additional metadata fields (optional)
    """
    id: str
    title: str
    body: str
    whole_doc: Dict[str, Any]
    url: Optional[str] = None
    fetched_at: Optional[str] = None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)
    
    def get_search_text(self) -> str:
        """Get text used for embedding (title + body)."""
        return f"{self.title} {self.body}".strip()
