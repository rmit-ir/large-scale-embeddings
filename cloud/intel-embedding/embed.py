#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "openvino-genai==2025.4.1.0",
# ]
# ///

"""
OpenVINO GenAI Text Embedding Script

This script uses OpenVINO GenAI to run text embedding models.

IMPORTANT: HuggingFace models must be converted to OpenVINO IR format first!

To convert MiniCPM-Embedding-Light (or any HuggingFace model):
1. Install optimum-cli:
   pip install optimum[openvino]

2. Convert the model:
   uv run --with optimum[openvino] optimum-cli export openvino --trust-remote-code --model openbmb/MiniCPM-Embedding-Light ./minicpm-embedding-ov
   # note on 19 Dec, 2025: Asked to export a minicpm model for the task feature-extraction (auto-detected), but the Optimum OpenVINO exporter only supports the tasks text-generation, text-generation-with-past for minicpm. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task feature-extraction to be supported in the ONNX export for minicpm.
   
   Or with weight compression (recommended for faster inference):
   uv run --with optimum[openvino] optimum-cli export openvino --trust-remote-code --model openbmb/MiniCPM-Embedding-Light --weight-format int8 ./minicpm-embedding-ov

3. Use the converted model path with this script:
   model = EmbeddingModel("./minicpm-embedding-ov")
"""

import openvino_genai as ov_genai


class EmbeddingModel:
    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Initialize embedding model from OpenVINO IR format.
        
        Args:
            model_path: Path to OpenVINO IR model directory (not HuggingFace model ID)
            device: Device to run inference on (CPU, GPU, NPU)
        """
        print(f"Loading model from: {model_path}")
        self.pipe = ov_genai.TextEmbeddingPipeline(model_path, device)
        print("Model loaded successfully")
    
    def embed(self, query: str) -> list[float]:
        embedding = self.pipe.embed_query(query)
        return embedding.tolist()
    
    def embed_docs(self, queries: list[str]) -> list[list[float]]:
        embeddings = self.pipe.embed_documents(queries)
        return [emb.tolist() for emb in embeddings]


def main():
    # Initialize the model once
    model = EmbeddingModel(device="GPU", model_path="./minicpm-embedding-ov")
    
    # Example queries
    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is machine learning?"
    ]
    
    # Embed queries one by one (reusing the same model)
    print("\n--- Single Query Embedding ---")
    for query in queries:
        print(f"\nQuery: {query}")
        embedding = model.embed(query)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    
    # Or embed as a batch
    print("\n--- Batch Embedding ---")
    batch_embeddings = model.embed_docs(queries)
    print(f"Embedded {len(batch_embeddings)} queries")
    for i, emb in enumerate(batch_embeddings):
        print(f"Query {i+1} embedding dimension: {len(emb)}")


if __name__ == "__main__":
    main()
