#!/usr/bin/env bash
# run the first shard of 8 total shards on device 0
# bash experiments/encode_clueweb_qwen3_0.6.sh 8 0 0

eval "$(conda shell.bash hook)"
conda activate minicpmembed

set -o allexport
source local.env
set +o allexport

echo $HF_HOME


# PATH_TO_MODEL=openbmb/MiniCPM-Embedding-Light
PATH_TO_MODEL=Qwen/Qwen3-Embedding-0.6B

EMBEDDING_OUTPUT_DIR=./data/ann_index/embeds/clueweb22b/Qwen3-Embedding-0.6B

mkdir -p $EMBEDDING_OUTPUT_DIR

# Check if required parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 <num_total_shard> <shard_index> [device]"
    echo "Example: $0 16 0 0"
    exit 1
fi

num_total_shard=$1
shard=$2
device=${3:-"0"}

echo "Processing shard $shard of $num_total_shard total shards"

# local_bz=2048
local_bz=512


# corpus encoding
# PATH_TO_CORPUS="/home/ubuntu/projects/index-clueweb22/ClueWeb22-B" 
PATH_TO_CORPUS="/home/eh6/E128356/scratch/collections/ClueWeb22-B" 
# TODO: langs default to all, otherwise "en" "de" .etc
python -m tevatron.retriever.driver.encode \
    --clueweb_api_dataset True \
    --langs "en" \
    --output_dir $EMBEDDING_OUTPUT_DIR \
    --bf16 \
    --model_name_or_path $PATH_TO_MODEL \
    --dataset_cache_dir $HF_HOME \
    --cache_dir $HF_HOME \
    --query_prefix "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: " \
    --passage_prefix "" \
    --pooling avg \
    --normalize \
    --per_device_eval_batch_size $local_bz \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_path $PATH_TO_CORPUS \
    --add_markers False \
    --dataset_number_of_shards $num_total_shard \
    --dataset_shard_index ${shard} \
    --inference_save_step 20 \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/clueweb22-corpus.cweb.${shard}.pkl \
    --device_id $device
