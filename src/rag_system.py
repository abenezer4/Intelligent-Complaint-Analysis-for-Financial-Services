import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.vector_store_manager import load_vector_store

class RAGSystem:
    def __init__(self, vector_store_path: str = "vector_store", embedding_model_name: str = "all-MiniLM-L6-v2", llm_model_name: str = "google/flan-t5-base"):
        """
        Initialize the RAG system.
        """
        print("Initializing RAG System...")
        
        # Load Vector Store
        try:
            self.index, self.chunks, self.metadata = load_vector_store(vector_store_path)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
            
        # Load Embedding Model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load LLM
        print(f"Loading LLM: {llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise

        print("RAG System Initialized.")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a given query.
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })
        
        return results

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM given query and retrieved context.
        """
        # Prepare context
        context_text = "\n\n".join([f"Excerpt {i+1}: {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        # Prompt Template
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context_text}

Question: {query}

Answer:"""

        # Generate
        response = self.llm_pipeline(prompt)[0]['generated_text']
        return response

    def query(self, user_question: str, k: int = 5) -> Dict[str, Any]:
        """
        Full RAG pipeline: Query -> Retrieve -> Generate
        """
        retrieved_docs = self.retrieve(user_question, k)
        answer = self.generate_answer(user_question, retrieved_docs)
        
        return {
            "question": user_question,
            "answer": answer,
            "source_documents": retrieved_docs
        }

# Import faiss here to ensure it's available for the class methods that might use it implicitly via index object
import faiss
