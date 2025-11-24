#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate minicpmembed

INDEX_DIR=./data/ann_index/embeds/clueweb22b-small/MiniCPM-Embedding-Light
QUERIES_N_OUTPUT_DIR=./data/queries/embeds/MiniCPM-Embedding-Light

set -f && python -m tevatron.retriever.driver.search \
    --query_reps $QUERIES_N_OUTPUT_DIR/sample_queries.pkl \
    --passage_reps $INDEX_DIR/clueweb22-corpus.cweb.*.pkl \
    --depth 100 \
    --batch_size 32 \
    --save_text \
    --save_ranking_to $QUERIES_N_OUTPUT_DIR/sample.run.txt
