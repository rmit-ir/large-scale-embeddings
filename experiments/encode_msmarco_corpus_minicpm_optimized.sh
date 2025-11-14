#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate minicpmembed

PATH_TO_MODEL=openbmb/MiniCPM-Embedding-Light
EMBEDDING_OUTPUT_DIR=./data/embeddings

mkdir -p $EMBEDDING_OUTPUT_DIR

shard=0
    
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --bf16 \
    --model_name_or_path $PATH_TO_MODEL \
    --dataset_name "Tevatron/msmarco-passage-corpus" \
    --dataset_cache_dir ./data/msmarco-passage-corpus \
    --cache_dir $HF_HOME \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --pooling avg \
    --normalize \
    --per_device_eval_batch_size 128 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --add_markers False \
    --dataset_number_of_shards 1 \
    --dataset_shard_index ${shard} \
    --dataloader_num_workers 4 \
    --inference_save_step 100 \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.msmarco.${shard}.pkl
