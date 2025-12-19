#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate minicpmembed

set -o allexport
source local.env
set +o allexport

echo "Building DiskANN index for ClueWeb22-B with MiniCPM embeddings"
echo "HF_HOME: $HF_HOME"

# Input paths
EMBEDDING_DIR=./data/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light-diskann
CORPUS_PATH=$EMBEDDING_DIR/embeds.bin

# Output paths
INDEX_ROOT_DIR=./data/ann_index/diskann-indexes
DATASET_NAME="clueweb22b_minicpm"

# Index parameters - optimized for 2TB RAM build system, 32GB RAM final search system
# Based on DiskANN documentation:
#   R: Default 64, typical 60-150 (higher = better recall, larger index)
#   L: Default 100, typical 75-200, should be >= R (higher = better quality, slower build)
#   B: Search DRAM budget - MUST match the RAM available on the final search system
#   M: Build DRAM budget - "allocate as much memory as your RAM allows" on build system
R=150           # Max node degree (at high end of recommended range for best recall)
L_BUILD=300     # Build list length (2x R, balanced between quality and build time)
B=24            # Search DRAM budget (GB) - for 32GB search system (75% of RAM)
M=1300          # Build DRAM budget (GB) - 65% of 2TB RAM on build system

INDEX_NAME="${DATASET_NAME}_R${R}_L${L_BUILD}_B${B}_M${M}"
INDEX_DIR="${INDEX_ROOT_DIR}/${INDEX_NAME}"
INDEX_PREFIX="index_"

echo "Index parameters:"
echo "  R (max node degree): $R"
echo "  L (build list length): $L_BUILD"
echo "  B (search DRAM budget): $B GB"
echo "  M (build DRAM budget): $M GB"
echo ""
echo "Paths:"
echo "  Corpus embeddings: $CORPUS_PATH"
echo "  Index directory: $INDEX_DIR"
echo "  Index prefix: $INDEX_PREFIX"
echo ""

# Create index directory
mkdir -p $INDEX_DIR

# Check if corpus file exists
if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found at $CORPUS_PATH"
    exit 1
fi

# Check if DiskANN is installed
if [ ! -f "./DiskANN-bin/build/apps/build_disk_index" ]; then
    echo "Error: DiskANN build_disk_index not found at ./DiskANN-bin/build/apps/build_disk_index"
    echo "Please install DiskANN according to: https://github.com/microsoft/DiskANN"
    exit 1
fi

echo "Starting DiskANN index build..."
echo "This may take several hours for large datasets (333GB)..."
echo ""

# Build the DiskANN index
# Parameters:
#   --data_type: Data type of embeddings (float for float32)
#   --dist_fn: Distance function (mips for maximum inner product search)
#   --data_path: Path to the binary embedding file
#   --index_path_prefix: Output path prefix for index files
#   -R: Max node degree (higher = more accurate but slower)
#   -L: Build list length (higher = more accurate but slower build)
#   -B: Search DRAM budget in GB
#   -M: Build DRAM budget in GB

./DiskANN-bin/build/apps/build_disk_index \
    --data_type float \
    --dist_fn mips \
    --data_path $CORPUS_PATH \
    --index_path_prefix ${INDEX_DIR}/${INDEX_PREFIX} \
    -R $R \
    -L $L_BUILD \
    -B $B \
    -M $M

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "DiskANN index build completed successfully!"
    echo "============================================"
    echo ""
    echo "Index files created in: $INDEX_DIR"
    echo ""
    echo "Index files:"
    ls -lh $INDEX_DIR/
    echo ""
    echo "Next steps:"
    echo "1. Start the main search API service"
else
    echo ""
    echo "Error: DiskANN index build failed!"
    exit 1
fi
