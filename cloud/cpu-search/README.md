# Hosting ClueWeb22-B with DiskANN

## Indexing process

ClueWeb22-B source files -> Embedding pickles -> DiskANN indexing input format -> DiskANN index

Steps:

### ClueWeb22-B source files -> Embedding pickles

Script `experiments/encode_clueweb.sh`

### Embedding pickles -> DiskANN indexing input format

Script `experiments/convert_clueweb_embeddings_to_diskann_input.py`

### DiskANN indexing input format -> DiskANN index

Documentation `launch_clueweb_search_api.md`

### Test search

Script `experiments/search_clueweb.sh`

## Usage

### Install dependencies

```bash
# Activate the conda environment
conda activate minicpmembed

# Update environment
conda env update --file ../../environment.yaml
```

### Run search server

```bash
# Run the server
PORT=51002 python router.py
```

The server provides two main endpoints:

#### 1. Embedding endpoint: `/embed`

Returns embeddings for text:

```bash
curl -X POST http://localhost:51002/embed \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?"}'
```

Returns:

```json
{
  "object": "list",
  "data": [{
      "object": "embedding",
      "embedding": [ 0.014239102602005005, 0.026707859709858894, ... -0.005067536141723394 ],
      "index": 0
  }],
  "model": "openbmb/MiniCPM-Embedding-Light",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

#### 2. Search endpoint: `/search`

Performs semantic search over ClueWeb22-B corpus:

```bash
curl -X POST http://localhost:51002/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 10,
    "complexity": 50,
    "with_distance": true
  }'
```

Returns:

```json
{
  "results": [
    {
      "docid": "clueweb22-en0000-00-00000",
      "distance": -5.436867237091064
    },
    ...
  ],
  "query": "What is machine learning?",
  "k": 10
}
```

**Note**: The search endpoint requires:
1. DiskANN search node running at `localhost:51001` (or set via `SEARCH_NODE_URL` env var)
2. Document ID mapping file at `./data/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light-diskann/docids.pkl`

### Run DiskANN search server

```bash
# No conda environment needed
uv run -p 3.11 --with numpy --with uvicorn --with diskannpy==0.7.0 --with fastapi --with pydantic search_api/cw22_search_api/cw22_node_generic.py \
   --node-id "0" \
   --index-dir "./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300" \
   --port 51001 \
   --index-prefix "index_" \
   --dimensions 1024 \
   --num-threads 4 \
   --num-nodes-to-cache 10000
```

Returns

```json
{
  "indices": [16009620, 58655364, 46378737, 40261862, 73973058, 36752250, 32638497, 4327366, 83243836, 76653306],
  "distances": [-5.4368672370910645, -5.443008899688721, -5.4554338455200195, -5.458445072174072, -5.482348442077637, -5.485116958618164, -5.520634651184082, -5.5278401374816895, -5.532476425170898, -5.542726516723633]
}
```

