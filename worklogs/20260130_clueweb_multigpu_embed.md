ClueWeb22 Sample Multi-GPU Embedding Update

Date
- 2026-01-30

Changes
- Added multi-GPU embedding support in `index_clueweb_sample.py` by splitting each batch across GPUs and running one MiniCPM model per GPU in parallel.
- New env var `EMBED_GPUS` controls GPU selection: `auto`/`all` (all visible GPUs), `cpu`/`none` (force CPU), or a comma list like `0,1,2`.
- `cuda:N` devices now correctly enable FP16 + Flash Attention when applicable.

Usage
- Example (2 GPUs):
  `EMBED_GPUS=0,1 BATCH_SIZE=128 python index_clueweb_sample.py index`
- Batch is split evenly across GPUs per call to `embed_batch` / `embed_queries_batch`.

Flash Attention
- Enable with `USE_FLASH_ATTN=1` (only applied on CUDA devices).
- Effect: uses `attn_implementation="flash_attention_2"` when supported by the model/torch stack.
- Expected benefits: higher throughput and lower memory on supported NVIDIA GPUs.
- Notes/risks: requires compatible GPU + PyTorch + flash-attn; if unsupported, model load can error.

Observations
- No benchmark run in this log; behavior is expected to improve embedding throughput when multiple GPUs are available and `USE_FLASH_ATTN=1` is supported.
