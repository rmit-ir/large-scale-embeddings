#!/usr/bin/env python3
"""
Health check utilities for checking dependency status
"""

import httpx
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from logging_util import logging

logger = logging.getLogger('health-util')


class DependencyHealth(BaseModel):
    """Dependency health status"""
    status: str = Field(...,
                        description="Dependency status (healthy/unhealthy/unknown)")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details about the dependency")
    error: Optional[str] = Field(
        None, description="Error message if unhealthy")


async def check_embed_health(embed_url: str) -> DependencyHealth:
    """
    Check health of the embedding service

    Args:
        embed_url: The embedding service URL (e.g., http://localhost:51003/embed)

    Returns:
        DependencyHealth with status and details
    """
    try:
        # Extract base URL from embed_url (remove /embed endpoint)
        embed_base_url = embed_url.rsplit('/', 1)[0]
        health_url = f"{embed_base_url}/health"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(health_url)

        if response.status_code == 200:
            health_data = response.json()
            return DependencyHealth(
                status="healthy",
                details=health_data
            )
        else:
            return DependencyHealth(
                status="unhealthy",
                error=f"HTTP {response.status_code}: {response.text}"
            )
    except httpx.TimeoutException:
        logger.warning(f"Embed service health check timed out: {embed_url}")
        return DependencyHealth(
            status="unhealthy",
            error="Connection timeout"
        )
    except Exception as e:
        logger.warning(f"Embed service health check failed: {str(e)}")
        return DependencyHealth(
            status="unhealthy",
            error=str(e)
        )


async def check_search_health(search_url: str) -> DependencyHealth:
    """
    Check health of the search service

    Args:
        search_url: The search service URL (e.g., http://localhost:51001/search)

    Returns:
        DependencyHealth with status and details
    """
    try:
        # Extract base URL from search_url (remove /search endpoint)
        search_base_url = search_url.rsplit('/', 1)[0]
        health_url = f"{search_base_url}/health"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(health_url)

        if response.status_code == 200:
            health_data = response.json()
            return DependencyHealth(
                status="healthy",
                details=health_data
            )
        else:
            return DependencyHealth(
                status="unhealthy",
                error=f"HTTP {response.status_code}: {response.text}"
            )
    except httpx.TimeoutException:
        logger.warning(f"Search service health check timed out: {search_url}")
        return DependencyHealth(
            status="unhealthy",
            error="Connection timeout"
        )
    except Exception as e:
        logger.warning(f"Search service health check failed: {str(e)}")
        return DependencyHealth(
            status="unhealthy",
            error=str(e)
        )
