#!/usr/bin/env python3
"""
CPU-Only Text Embedding Script using Tevatron

This script wraps the tevatron encoding approach for CPU-only inference.
Based on the encoding method used in encode_clueweb_example_queries.sh
"""

import sys
from pathlib import Path

# Add tevatron to path
tevatron_path = Path(__file__).parent.parent.parent / "tevatron" / "src"
sys.path.insert(0, str(tevatron_path))

import torch
from transformers import AutoTokenizer
from tevatron.retriever.modeling import DenseModel


class EmbeddingModel:
    def __init__(self, model_path: str = "openbmb/MiniCPM-Embedding-Light"):
        """
        Initialize embedding model using Tevatron's DenseModel for CPU-only inference.
        
        Args:
            model_path: Path to local model directory or HuggingFace model ID
        """
        print(f"Loading model from: {model_path}")
        print("Using CPU device for inference")
        
        # Force CPU device
        self.device = torch.device("cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        
        # Load model with CPU settings (float32 for CPU)
        self.model = DenseModel.load(
            model_path,
            pooling='avg',  # Average pooling
            normalize=True,  # L2 normalization
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Query prefix used in the encoding script
        self.query_prefix = "Instruction: Given a web search query, retrieve relevant passages that answer the query. Query: "
        
        print("Model loaded successfully")
    
    def embed(self, query: str) -> list[float]:
        """
        Embed a single query using the same approach as tevatron encode.
        
        Args:
            query: Text to embed
            
        Returns:
            Embedding as a list of floats
        """
        # Add query prefix
        text = self.query_prefix + query
        
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=64,  # query_max_len from script
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            batch = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Encode using model
            model_output = self.model(query=batch)
            embedding = model_output.q_reps.cpu().detach().numpy()[0]
            
            return embedding.tolist()
    
    def embed_docs(self, queries: list[str]) -> list[list[float]]:
        """
        Embed multiple queries in a batch.
        
        Args:
            queries: List of texts to embed
            
        Returns:
            List of embeddings, each as a list of floats
        """
        # Add query prefix to all
        texts = [self.query_prefix + q for q in queries]
        
        with torch.no_grad():
            # Tokenize batch
            encoded = self.tokenizer(
                texts,
                max_length=64,  # query_max_len from script
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            batch = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Encode using model
            model_output = self.model(query=batch)
            embeddings = model_output.q_reps.cpu().detach().numpy()
            
            return [emb.tolist() for emb in embeddings]


def main():
    # Initialize the model once (CPU-only, no GPU support)
    model = EmbeddingModel(model_path="openbmb/MiniCPM-Embedding-Light")
    
    # Example queries
    queries = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is machine learning?"
    ]
    
    # Embed queries one by one
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
