"""
Logging utilities for ASEDISKS.
"""

from __future__ import annotations

import logging
import os
import sys


def _resolve_log_level() -> int:
    level_name = os.environ.get("ASEDISKS_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def setup_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name if name else "asedisks")
