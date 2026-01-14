import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RAGSystem

def test_rag():
    print("Testing RAG System...")
    try:
        rag = RAGSystem()
        
        question = "What are the common issues with Credit Cards?"
        print(f"\nQuestion: {question}")
        
        result = rag.query(question)
        
        print("\nAnswer:")
        print(result['answer'])
        
        print("\nSources:")
        for doc in result['source_documents'][:2]:
            print(f"- {doc['text'][:100]}...")
            
        print("\nTest passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag()
