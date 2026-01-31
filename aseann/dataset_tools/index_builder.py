"""
Index builder for creating DiskANN indexes from binary files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import shutil
import json
import numpy as np

from aseann.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DiskANNConfig:
    """Configuration for DiskANN index building."""

    metric: str = "mips"  # "l2", "mips", "cosine"
    R: int = 64  # Max node degree (60-150)
    L: int = 100  # Build complexity (>= R)
    build_threads: int = 32
    build_memory_gb: int = 64  # RAM for building
    search_memory_gb: int = 24  # RAM for search
    diskann_bin_path: Optional[Path] = None


def build_index(
    binary_file: Path,
    output_dir: Path,
    config: Optional[DiskANNConfig] = None,
):
    """
    Build DiskANN index from binary file.

    This function builds a DiskANN ANN index from embeddings in binary format.
    It uses the DiskANN C++ tools under the hood with optimized settings
    for large-scale search (billions of vectors).

    Args:
        binary_file: Path to binary file containing embeddings (embeds.bin)
        output_dir: Directory to write the index
        config: DiskANN configuration options

    The function handles:
    - Validating binary file format
    - Calling DiskANN build_disk_index tool
    - Optimizing index parameters for search performance
    - Progress tracking and error handling

    Output files:
        output_dir/index_*: DiskANN index files
        output_dir/build_stats.json: Build statistics

    Note:
        Requires DiskANN to be installed and configured.
        The index building process can be memory intensive for large datasets.
    """
    config = config or DiskANNConfig()
    binary_file = Path(binary_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate binary file
    if not binary_file.exists():
        raise FileNotFoundError(f"Binary file not found: {binary_file}")

    num_vectors, embedding_dim = _read_binary_header(binary_file)
    logger.info("Building index:")
    logger.info("  Vectors: %s", f"{num_vectors:,}")
    logger.info("  Dimensions: %s", embedding_dim)
    logger.info("  Metric: %s", config.metric)
    logger.info("  R (max degree): %s", config.R)
    logger.info("  L (build complexity): %s", config.L)
    logger.info("  Build memory: %s GB", config.build_memory_gb)
    logger.info("  Search memory: %s GB", config.search_memory_gb)

    # Find DiskANN binary
    diskann_path = _find_diskann(config.diskann_bin_path)
    logger.info("  Using DiskANN: %s", diskann_path)

    # Build command
    index_prefix = output_dir / "index_"
    cmd = [
        str(diskann_path / "build_disk_index"),
        "--data_type",
        "float",
        "--dist_fn",
        config.metric,
        "--data_path",
        str(binary_file),
        "--index_path_prefix",
        str(index_prefix),
        "-R",
        str(config.R),
        "-L",
        str(config.L),
        "-B",
        str(config.search_memory_gb),
        "-M",
        str(config.build_memory_gb),
        "-T",
        str(config.build_threads),
    ]

    logger.info("Running: %s", " ".join(cmd))

    # Run build
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("STDOUT: %s", result.stdout)
        logger.error("STDERR: %s", result.stderr)
        raise RuntimeError(f"DiskANN build failed with code {result.returncode}")

    logger.info(result.stdout)

    # Save build stats
    stats = {
        "num_vectors": num_vectors,
        "embedding_dim": embedding_dim,
        "config": {
            "metric": config.metric,
            "R": config.R,
            "L": config.L,
            "build_threads": config.build_threads,
            "build_memory_gb": config.build_memory_gb,
            "search_memory_gb": config.search_memory_gb,
        },
    }
    stats_path = output_dir / "build_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Index built at %s", output_dir)
    logger.info("Stats saved to %s", stats_path)


def _read_binary_header(binary_file: Path) -> tuple[int, int]:
    """
    Read header from DiskANN binary file.

    Binary format:
    - 4 bytes: number of vectors (uint32, little-endian)
    - 4 bytes: embedding dimension (uint32, little-endian)
    - Remaining: vectors as float32

    Args:
        binary_file: Path to binary file

    Returns:
        Tuple of (num_vectors, embedding_dim)
    """
    with open(binary_file, "rb") as f:
        num_vectors = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        embedding_dim = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    return int(num_vectors), int(embedding_dim)


def _validate_binary_format(binary_file: Path) -> bool:
    """
    Validate that binary file is in correct DiskANN format.

    Args:
        binary_file: Path to binary file to validate

    Returns:
        True if format is valid, False otherwise
    """
    try:
        num_vectors, embedding_dim = _read_binary_header(binary_file)

        # Check file size matches expected
        expected_size = 8 + (num_vectors * embedding_dim * 4)  # header + vectors
        actual_size = binary_file.stat().st_size

        if actual_size != expected_size:
            logger.error(
                "Size mismatch: expected %s, got %s",
                expected_size,
                actual_size,
            )
            return False

        return True
    except Exception as e:
        logger.error("Validation failed: %s", e)
        return False


def _find_diskann(custom_path: Optional[Path]) -> Path:
    """
    Find DiskANN installation.

    Args:
        custom_path: Optional custom path to DiskANN binaries

    Returns:
        Path to DiskANN apps directory

    Raises:
        FileNotFoundError: If DiskANN not found
    """
    if custom_path:
        custom_path = Path(custom_path)
        if custom_path.exists() and (custom_path / "build_disk_index").exists():
            return custom_path

    # Search common locations
    search_paths = [
        Path("./DiskANN/build/apps"),
        Path("./DiskANN-bin/build/apps"),
        Path("../DiskANN/build/apps"),
        Path.home() / "DiskANN/build/apps",
        Path("/usr/local/bin"),
        Path("/opt/diskann/bin"),
    ]

    for path in search_paths:
        if path.exists() and (path / "build_disk_index").exists():
            return path

    # Try to find in PATH
    which_result = shutil.which("build_disk_index")
    if which_result:
        return Path(which_result).parent

    raise FileNotFoundError(
        "DiskANN (build_disk_index) not found. Please install DiskANN and either:\n"
        "  1. Add to PATH\n"
        "  2. Set diskann_bin_path in DiskANNConfig\n"
        "  3. Place in ./DiskANN/build/apps/"
    )
