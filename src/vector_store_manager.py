import faiss
import pickle
import json
import os
import numpy as np
from typing import List, Dict, Tuple

def save_vector_store(index: faiss.Index, chunks: List[str], metadata: List[Dict], output_dir: str = "vector_store") -> None:
    """
    Save the FAISS index and associated metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    # Save text chunks
    with open(os.path.join(output_dir, "text_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    print(f"Vector store saved to {output_dir}/")

def load_vector_store(input_dir: str = "vector_store") -> Tuple[faiss.Index, List[str], List[Dict]]:
    """
    Load the FAISS index and associated metadata.
    """
    if not os.path.exists(input_dir):
        # Try finding it relative to project root if running from notebook
        if os.path.exists(f"../{input_dir}"):
            input_dir = f"../{input_dir}"
        elif os.path.exists(f"../../{input_dir}"): # deeper nesting
            input_dir = f"../../{input_dir}"
            
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Vector store directory '{input_dir}' not found.")
        
    index_path = os.path.join(input_dir, "faiss_index.bin")
    chunks_path = os.path.join(input_dir, "text_chunks.pkl")
    metadata_path = os.path.join(input_dir, "metadata.json")
    
    if not all(os.path.exists(p) for p in [index_path, chunks_path, metadata_path]):
         raise FileNotFoundError(f"Missing files in vector store directory '{input_dir}'. Required: faiss_index.bin, text_chunks.pkl, metadata.json")

    print(f"Loading vector store from {input_dir}...")
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load chunks
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    print(f"Vector store loaded. Index size: {index.ntotal}")
    return index, chunks, metadata
