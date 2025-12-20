#!/usr/bin/env bash
INDEX_DIR="./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300"
QUERY_BIN="./data/queries/embeds/MiniCPM-Embedding-Light/sample_queries.bin"
RESULT_DIR="${INDEX_DIR}/results/K100_L100"
mkdir -p $RESULT_DIR

./DiskANN-bin/build/apps/search_disk_index --data_type float --dist_fn mips \
    --index_path_prefix "${INDEX_DIR}/index_" --query_file $QUERY_BIN \
    -K 100 -L 100 --result_path "${RESULT_DIR}/res_" --num_nodes_to_cache 10000 \
    --num_threads 4
