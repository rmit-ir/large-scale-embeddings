#!/usr/bin/env python3
"""
Convert query embeddings from pickle format to DiskANN binary format.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskann.utils import convert_encoded_pkl_to_binary

INPUT_PKL = "./data/queries/embeds/MiniCPM-Embedding-Light/sample_queries.pkl"
OUTPUT_BIN = "./data/queries/embeds/MiniCPM-Embedding-Light/sample_queries.bin"
OUTPUT_DOCIDS = "./data/queries/embeds/MiniCPM-Embedding-Light/sample_queries_docids.pkl"

def main():
    print(f"Converting query embeddings to DiskANN format...")
    print(f"  Input: {INPUT_PKL}")
    print(f"  Output (embeddings): {OUTPUT_BIN}")
    print(f"  Output (doc IDs): {OUTPUT_DOCIDS}")
    
    # Convert the pickle file to binary format
    convert_encoded_pkl_to_binary(INPUT_PKL, OUTPUT_BIN, OUTPUT_DOCIDS)
    
    print("\nConversion completed successfully!")
    
    # Show file sizes
    print("\nOutput files:")
    os.system(f"ls -lh {OUTPUT_BIN} {OUTPUT_DOCIDS}")


if __name__ == "__main__":
    main()
