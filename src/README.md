# Source Code Modules

This directory contains reusable modules for the RAG Complaint Chatbot project.

## Modules

### `data_loader.py`
Handles loading of dataset files.
- `load_filtered_data(filepath)`: Loads the filtered CSV file.

### `vector_store_manager.py`
Handles saving and loading of the FAISS vector store.
- `save_vector_store(index, chunks, metadata, output_dir)`
- `load_vector_store(input_dir)`

### `rag_system.py`
Contains the core RAG logic.
- `RAGSystem`: Class that initializes the pipeline, retrieves documents, and generates answers.
  - `query(question)`: Returns answer and source documents.

## Usage

```python
from src.data_loader import load_filtered_data
from src.rag_system import RAGSystem

# Load data
df = load_filtered_data()

# Use RAG
rag = RAGSystem()
result = rag.query("What are the issues?")
print(result['answer'])
```
