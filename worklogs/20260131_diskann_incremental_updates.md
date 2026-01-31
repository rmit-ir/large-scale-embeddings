# DiskANN incremental updates notes (2026-01-31)

Summary of findings (from DiskANN repo docs/issues):
- DiskANN’s SSD/disk index is effectively static; you can append to the raw `embeds.bin` file, but you **cannot** incrementally add vectors to an already-built disk index.
- The repo documents a **dynamic (in-memory) index** workflow with insertions/lazy deletes/consolidation, but that is separate from the SSD/disk index build path.
- For updates to a disk index, maintainers suggest rebuilding from full data or using a base+delta strategy; incremental disk updates aren’t supported out of the box.

Implication for our pipeline:
- Complete `_run_writer` to write the full `embeds.bin` + `docids.pkl` + `config.json`, then run the index build step in `index_builder`.
- If we need incremental ingestion, we should consider a dynamic in-memory index or a base+delta approach.
