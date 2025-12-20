This worklog file documents the progress and updates made on December 15, 2025 onwards trying to launch a search API for ASE project.



<details>
<summary>15 Dec 2025, init</summary>

Previously done:

1. Ran convert_clueweb_pkl_to_diskann_format.py to convert embeddings to diskann format.

### Building DiskANN

DiskANN release page https://github.com/microsoft/DiskANN/releases doesn't include diskann binaries, the whl files are just python bindings. So we built DiskANN from source.

```bash
[e128356@sctsresap21 diskann]$ ls
diskannpy  diskannpy-0.7.0-cp311-cp311-manylinux_2_28_x86_64.whl  diskannpy-0.7.0.dist-info  diskannpy.libs
[e128356@sctsresap21 diskann]$ find .
.
./diskannpy-0.7.0-cp311-cp311-manylinux_2_28_x86_64.whl
./diskannpy
./diskannpy/diskannpy.egg-info
./diskannpy/diskannpy.egg-info/PKG-INFO
./diskannpy/diskannpy.egg-info/SOURCES.txt
./diskannpy/diskannpy.egg-info/dependency_links.txt
./diskannpy/diskannpy.egg-info/not-zip-safe
./diskannpy/diskannpy.egg-info/top_level.txt
./diskannpy/diskannpy.egg-info/requires.txt
./diskannpy/py.typed
./diskannpy/module.cpp
./diskannpy/dynamic_memory_index.cpp
./diskannpy/_dynamic_memory_index.py
./diskannpy/static_disk_index.cpp
./diskannpy/_builder.pyi
./diskannpy/_builder.py
./diskannpy/_static_disk_index.py
./diskannpy/_static_memory_index.py
./diskannpy/_common.py
./diskannpy/static_memory_index.cpp
./diskannpy/defaults.py
./diskannpy/_diskannpy.cpython-311-x86_64-linux-gnu.so
./diskannpy/__init__.py
./diskannpy/builder.cpp
./diskannpy/_files.py
./diskannpy-0.7.0.dist-info
./diskannpy-0.7.0.dist-info/WHEEL
./diskannpy-0.7.0.dist-info/METADATA
./diskannpy-0.7.0.dist-info/top_level.txt
./diskannpy-0.7.0.dist-info/LICENSE
./diskannpy-0.7.0.dist-info/NOTICE.txt
./diskannpy-0.7.0.dist-info/RECORD
./diskannpy.libs
./diskannpy.libs/libmkl_def-c5f22b52.so
./diskannpy.libs/libiomp5-3994699c.so
./diskannpy.libs/libgomp-a97153a2.so.1.0.0
./diskannpy.libs/libaio-8c118d5e.so.1.0.1
```

### Create DiskANN index

Manually configuring diskann running environment library paths, $LD_LIBRARY_PATH to include intel libraries and diskann libraries.

```bash
export LD_LIBRARY_PATH=/home/eh6/E128356/projects/large-scale-embeddings/diskann_libs/intel64_lin_compilers_and_libraries_2020.4.304:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/eh6/E128356/projects/large-scale-embeddings/diskann_libs/opt_intel_mkl_lib_intel64_lin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/eh6/E128356/projects/large-scale-embeddings/diskann_libs/usr-virtual/lib64:$LD_LIBRARY_PATH

[e128356@sctsresap21 apps]$ pwd
/home/eh6/E128356/projects/large-scale-embeddings/DiskANN-bin/build/apps

[e128356@sctsresap21 apps]$ ./build_disk_index
the option '--build_DRAM_budget' is required but missing
```

More information on how to build DiskANN index on:

https://github.com/rmit-ir/DiskANN/blob/main/workflows/SSD_index.md

### Running search_clueweb_diskann_index.sh

6980 queries on ClueWeb22B MiniCPM embeddings, searching on DiskANN index.

```bash
[e128356@sctsresap21 large-scale-embeddings]$ bash experiments/search_clueweb_diskann_index.sh
Search parameters: #threads: 4,  beamwidth: 2.
Reading (with alignment) bin file ./data/queries/embeds/MiniCPM-Embedding-Light/sample_queries.bin ...Metadata: #pts = 6980, #dims = 1024, aligned_dim = 1024... allocating aligned memory of 28590080 bytes... done. Copying data to mem_aligned buffer... done.
Since data is floating point, we assume that it has been appropriately pre-processed (normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we shall invoke an l2 distance function.
L2: Using AVX2 distance computation DistanceL2Float
L2: Using AVX2 distance computation DistanceL2Float
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_compressed.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_compressed.bin...
Metadata: #pts = 87208655, #dims = 292...
tcmalloc: large alloc 25464930304 bytes == 0x42d4000 @  0x7f3345dc2760 0x7f3345de3c64 0x8096a8 0x84ca89 0x84ea57 0x422a71 0x40f9a3 0x7f3344c23865 0x4104be
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 4, #dims = 1...
done.
Offsets: 4096 1053704 1057812 1058992
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 256, #dims = 1025...
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 1025, #dims = 1...
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 293, #dims = 1...
done.
Loaded PQ Pivots: #ctrs: 256, #dims: 1025, #chunks: 292
Loaded PQ centroids and in-memory compressed vectors. #points: 87208655 #dim: 1025 #aligned_dim: 1032 #chunks: 292
Disk-Index File Meta-data: # nodes per sector: 0, max node len (bytes): 4704, max node degree: 150
Opened file : ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index
Setting up thread-specific contexts for nthreads: 4
allocating ctx: 0x7f334f83e000 to thread-id:139858355574912
allocating ctx: 0x7f334f82d000 to thread-id:139858162751360
allocating ctx: 0x7f334f81c000 to thread-id:139858154354688
allocating ctx: 0x7f334f80b000 to thread-id:139858145958016
Loading centroid data from medoids vector data of 1 medoid(s)
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index_max_base_norm.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index_max_base_norm.bin...
Metadata: #pts = 1, #dims = 1...
done.
Setting re-scaling factor of base vectors to 1
done..
Caching 10000 nodes around medoid(s)
Caching 10000...
Level: 1.. #nodes: 1, #nodes thus far: 1
Level: 2.. #nodes: 150, #nodes thus far: 151
Level: 3. #nodes: 9849, #nodes thus far: 10000
done
Loading the cache list into memory....done.
     L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs    Mean IO (us)         CPU (s)
===================================================================================================================================
   100           2          199.14        19883.89        39685.00          108.68        16007.48         3580.28
Done searching. Now saving results
Writing bin: ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/results/K100_L100/res__100_idx_uint32.bin
bin: #pts = 6980, #dims = 100, size = 2792008B
Finished writing bin.
Writing bin: ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/results/K100_L100/res__100_dists_float.bin
bin: #pts = 6980, #dims = 100, size = 2792008B
Finished writing bin.
Clearing scratch
```

Running on the server node 0, port 51001:

```bash
(minicpmembed) [e128356@sctsresap21 large-scale-embeddings]$ python search_api/cw22_search_api/cw22_node_generic.py \
   --node-id "0" \
   --index-dir "./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300" \
   --port 51001 \
   --index-prefix "index_" \
   --dimensions 1024 \
   --num-threads 4 \
   --num-nodes-to-cache 10000
src/tcmalloc.cc:283] Attempt to free invalid pointer 0x55a0ade11510
Aborted (core dumped)
```

Running on home server, inside Docker

```bash
sudo docker run --rm -it -p 51001:51001 -v `pwd`/large-scale-embeddings:/app/large-scale-embeddings diskann bash

apt install -y curl && curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

cd ../large-scale-embeddings

uv run -p 3.11 --with numpy --with uvicorn --with diskannpy==0.7.0 --with fastapi --with pydantic search_api/cw22_search_api/cw22_node_generic.py \
   --node-id "0" \
   --index-dir "./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300" \
   --port 51001 \
   --index-prefix "index_" \
   --dimensions 1024 \
   --num-threads 4 \
   --num-nodes-to-cache 10000

Installed 16 packages in 49ms

======== Node 0 Service Information ========
Starting search service on HPC cluster
Index Directory: ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300
The service will be accessible at: http://172.17.0.4:51001
API documentation will be available at: http://172.17.0.4:51001/docs
======================================

INFO:     Started server process [226]
INFO:     Waiting for application startup.
Since data is floating point, we assume that it has been appropriately pre-processed (normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we shall invoke an l2 distance function.
L2: Using AVX2 distance computation DistanceL2Float
L2: Using AVX2 distance computation DistanceL2Float
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_compressed.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_compressed.bin...
Metadata: #pts = 87208655, #dims = 292...
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 4, #dims = 1...
done.
Offsets: 4096 1053704 1057812 1058992
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 256, #dims = 1025...
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 1025, #dims = 1...
done.
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__pq_pivots.bin...
Metadata: #pts = 293, #dims = 1...
done.
Loaded PQ Pivots: #ctrs: 256, #dims: 1025, #chunks: 292
Loaded PQ centroids and in-memory compressed vectors. #points: 87208655 #dim: 1025 #aligned_dim: 1032 #chunks: 292
Disk-Index File Meta-data: # nodes per sector: 0, max node len (bytes): 4704, max node degree: 150
Opened file : ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index
Setting up thread-specific contexts for nthreads: 4
allocating ctx: 0x7f39d2986000 to thread-id:139860760278976
allocating ctx: 0x7f39d2975000 to thread-id:139860768675648
allocating ctx: 0x7f39cfe1d000 to thread-id:139886854684672
allocating ctx: 0x7f39cf86e000 to thread-id:139860777072320
Loading centroid data from medoids vector data of 1 medoid(s)
Reading bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index_max_base_norm.bin ...
Opening bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__disk.index_max_base_norm.bin...
Metadata: #pts = 1, #dims = 1...
done.
Setting re-scaling factor of base vectors to 1
done..
Reading (with alignment) bin file ./data/ann_index/diskann-indexes/clueweb22b_minicpm_R150_L300_B24_M1300/index__sample_data.bin ...Metadata: #pts = 100108, #dims = 1025, aligned_dim = 1032... allocating aligned memory of 413245824 bytes... done. Copying data to mem_aligned buffer... done.
```

</details>
