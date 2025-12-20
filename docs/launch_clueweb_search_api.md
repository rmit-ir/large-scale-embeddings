# Guide: Launching ClueWeb22-B Search API with DiskANN

This guide walks you through launching the ClueWeb22-B search API after building the DiskANN index.

## Prerequisites

Before starting, ensure you have:
- ✅ Completed encoding: `experiments/encode_clueweb_docs_minicpm.sh`
- ✅ Converted to DiskANN format: `experiments/convert_clueweb_pkl_to_diskann_format.py`
- ⏳ **Next:** Build DiskANN index (this guide starts here)

## Step 1: Build the DiskANN Index

Run the build script to create the searchable index:

```bash
cd /home/eh6/E128356/projects/large-scale-embeddings
bash experiments/build_clueweb_diskann_index.sh
```

**Important:**
- This will take several hours (possibly 12-24+ hours for 333GB)
- Monitor progress with `top` or `htop` to check CPU/memory usage
- The index will be created at: `./data/ann_index/diskann-indexes/clueweb22b_minicpm_R100_L100_B64_M128/`

Expected output files after build:
```
data/ann_index/diskann-indexes/clueweb22b_minicpm_R100_L100_B64_M128/
├── index_disk.index
├── index_disk.index_medoids
├── index_disk.index_centroids
├── index_sample_ids.bin
└── ... (other index files)
```

