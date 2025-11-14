#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate minicpmembed

PATH_TO_MODEL=openbmb/MiniCPM-Embedding-Light
EMBEDDING_OUTPUT_DIR=./data/embeddings

mkdir -p $EMBEDDING_OUTPUT_DIR
    
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --bf16 \
    --model_name_or_path $PATH_TO_MODEL \
    --dataset_name "Tevatron/msmarco-passage" \
    --dataset_split "validation" \
    --dataset_cache_dir ./data/msmarco-passage \
    --cache_dir $HF_HOME \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --pooling avg \
    --normalize \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --encode_is_query \
    --add_markers False \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/queries.msmarco.dev.pkl
