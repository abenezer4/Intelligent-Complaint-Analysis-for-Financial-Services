import pandas as pd
import numpy as np
import faiss
import os
import json
import pickle
import pyarrow.parquet as pq

def ingest_prebuilt_embeddings(parquet_path: str = "complaint_embeddings.parquet", output_dir: str = "vector_store", batch_size: int = 10000):
    """
    Reads the provided parquet file in batches to save memory,
    builds a FAISS index incrementally, and saves it.
    """
    print(f"Loading pre-built embeddings from {parquet_path}...")
    
    if not os.path.exists(parquet_path):
        print(f"ERROR: File {parquet_path} not found.")
        print("Please download 'complaint_embeddings.parquet' from the Google Drive link in the assignment.")
        return

    try:
        # Initialize Parquet File Reader
        parquet_file = pq.ParquetFile(parquet_path)
        print(f"Total rows in file: {parquet_file.metadata.num_rows}")
        
        # Variables to hold state
        faiss_index = None
        all_chunks = []
        all_metadata = []
        
        # Iterate over batches
        batch_count = 0
        total_batches = (parquet_file.metadata.num_rows // batch_size) + 1
        
        # Variables for column identification
        emb_col = None
        text_col = None
        
        print(f"Processing in batches of {batch_size}...")
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_count += 1
            df_batch = batch.to_pandas()
            
            # --- 1. Identify Columns (only on first batch) ---
            if faiss_index is None:
                print(f"Columns found in file: {df_batch.columns.tolist()}")
                
                # A. Find embedding column
                if 'embedding' in df_batch.columns:
                    emb_col = 'embedding'
                elif 'embeddings' in df_batch.columns:
                    emb_col = 'embeddings'
                else:
                    # Heuristic: Find column with lists/arrays
                    for col in df_batch.columns:
                        val = df_batch.iloc[0][col]
                        if isinstance(val, (list, np.ndarray)) and len(val) > 100:
                            emb_col = col
                            break
                
                if not emb_col:
                    raise ValueError(f"Could not identify embedding column. Available: {df_batch.columns.tolist()}")
                
                # B. Find text column
                # List of potential names based on common datasets
                potential_text_cols = ['text', 'narrative', 'chunk', 'content', 'consumer_complaint_narrative', 'complaint_what_happened', 'body']
                
                for candidate in potential_text_cols:
                    if candidate in df_batch.columns:
                        text_col = candidate
                        break
                
                # Fallback: Find longest string column if name match fails
                if not text_col:
                    max_avg_len = 0
                    for col in df_batch.columns:
                        if col == emb_col: continue
                        # Check if string
                        if pd.api.types.is_string_dtype(df_batch[col]):
                            avg_len = df_batch[col].str.len().mean()
                            if avg_len > max_avg_len:
                                max_avg_len = avg_len
                                text_col = col
                
                if not text_col:
                    raise ValueError(f"Could not identify text/narrative column. Available: {df_batch.columns.tolist()}")
                
                print(f"  Using embedding column: '{emb_col}'")
                print(f"  Using text column: '{text_col}'")

            # --- 2. Process Embeddings for this batch ---
            # Stack embeddings into a matrix
            embeddings_list = df_batch[emb_col].tolist()
            batch_matrix = np.vstack(embeddings_list).astype('float32')
            
            # Normalize (L2) for Cosine Similarity
            faiss.normalize_L2(batch_matrix)
            
            # Initialize Index if needed
            if faiss_index is None:
                dimension = batch_matrix.shape[1]
                faiss_index = faiss.IndexFlatIP(dimension)
                print(f"  Initialized FAISS index with dimension: {dimension}")

            # Add to Index
            faiss_index.add(batch_matrix)
            
            # --- 3. Collect Text and Metadata ---
            batch_chunks = df_batch[text_col].tolist()
            all_chunks.extend(batch_chunks)
            
            # Metadata (exclude heavy columns)
            # We exclude embedding and text to keep metadata light
            metadata_cols = [c for c in df_batch.columns if c not in [emb_col, text_col]]
            batch_metadata = df_batch[metadata_cols].to_dict('records')
            all_metadata.extend(batch_metadata)
            
            print(f"  Processed batch {batch_count}/{total_batches} (Total indexed: {faiss_index.ntotal})")

        # --- 4. Save to Disk ---
        print(f"\nSaving to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(faiss_index, os.path.join(output_dir, "faiss_index.bin"))
        
        # Save text chunks
        with open(os.path.join(output_dir, "text_chunks.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(all_metadata, f)
            
        print("Success! The pre-built embeddings have been ingested.")
        print("You can now run 'python app.py' to use the full dataset.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ingest_prebuilt_embeddings()
