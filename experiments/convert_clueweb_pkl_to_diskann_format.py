#!/usr/bin/env python3
"""
Convert ClueWeb22 encoded pickle files to DiskANN binary format.

This script takes the encoded pickle files from the ClueWeb22 corpus 
and converts them to DiskANN binary format using the convert_encoded_pkls_to_binary function.
"""

import os
import sys
import glob

# Add the project root to Python path to import diskann utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskann.utils import convert_encoded_pkls_to_binary

INDEX_DIR = "./data/ann_index/embeds/clueweb22b/MiniCPM-Embedding-Light"
OUTPUT_DIR = INDEX_DIR + "-diskann"
print(f"Input directory: {INDEX_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

def main():
    # Check if input directory exists
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"Error: Input directory {INDEX_DIR} does not exist!")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all pkl files matching the pattern from search_clueweb.sh
    pkl_pattern = os.path.join(INDEX_DIR, "clueweb22-corpus.cweb.*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        raise FileNotFoundError(f"No pickle files found matching pattern: {pkl_pattern}")
    
    # Sort files to ensure consistent processing order
    pkl_files.sort()
    
    print(f"Found {len(pkl_files)} pickle files:")
    for f in pkl_files:
        print(f"  - {os.path.basename(f)}")
    
    # Extract just the filenames for the conversion function
    input_names = [os.path.basename(f) for f in pkl_files]
    
    # Define output paths
    embed_output_path = os.path.join(OUTPUT_DIR, "embeds.bin")
    docid_output_path = os.path.join(OUTPUT_DIR, "docids.pkl")
    
    print(f"\nConverting to DiskANN format...")
    print(f"  Embeddings output: {embed_output_path}")
    print(f"  DocIDs output: {docid_output_path}")
    
    # Convert the pickle files to binary format
    convert_encoded_pkls_to_binary(
        input_dir=INDEX_DIR,
        input_names=input_names,
        embed_output_path=embed_output_path,
        docid_output_path=docid_output_path
    )
    
    print("\nConversion completed successfully!")
    print(f"Binary embeddings saved to: {embed_output_path}")
    print(f"Document IDs saved to: {docid_output_path}")
    
    # Run ls -lh equivalent to show file sizes
    print("\nOutput files:")
    os.system(f"ls -lh {embed_output_path} {docid_output_path}")


if __name__ == "__main__":
    main()
