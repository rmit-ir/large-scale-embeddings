#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate minicpmembed

set -o allexport
source local.env
set +o allexport

echo $HF_HOME

PATH_TO_QUERIES="./data/datasets/msmarco-passage/dev.jsonl"

PATH_TO_MODEL=openbmb/MiniCPM-Embedding-Light

EMBEDDING_OUTPUT_DIR=./data/queries/embeds/MiniCPM-Embedding-Light

mkdir -p $EMBEDDING_OUTPUT_DIR

python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path $PATH_TO_MODEL \
    --bf16 \
    --pooling avg \
    --dataset_cache_dir $HF_HOME \
    --cache_dir $HF_HOME \
    --normalize \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 64 \
    --passage_max_len 512 \
    --dataset_path $PATH_TO_QUERIES \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/sample_queries.pkl \
    --device_id 0





