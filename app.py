import gradio as gr
from src.rag_system import RAGSystem
import os

# Initialize RAG System
print("Initializing RAG System for UI...")
vector_store_path = "vector_store"

rag_system = None
try:
    if os.path.exists(vector_store_path):
        rag_system = RAGSystem(vector_store_path=vector_store_path)
    else:
        print(f"Warning: Vector store not found at {vector_store_path}. Please run Task 2 first.")
except Exception as e:
    print(f"Failed to initialize RAG system: {e}")

def chat_function(message, history):
    """
    Chat function for Gradio ChatInterface.
    Parameters:
        message (str): The current user message.
        history (list): List of dictionaries representing conversation history 
                       [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    Returns:
        str: The bot's response.
    """
    if rag_system is None:
        return "System not initialized correctly. Please check if the vector store exists and try again."
    
    try:
        # Generate response using RAG
        response = rag_system.query(message)
        answer = response["answer"]
        
        # Format sources
        sources_text = "\n\n**Sources:**\n"
        if response["source_documents"]:
            for i, doc in enumerate(response["source_documents"][:3]): # Show top 3
                text = doc.get('text', '')[:200]
                sources_text += f"{i+1}. {text}...\n"
        else:
            sources_text += "No relevant sources found."
            
        return answer + sources_text
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Create the ChatInterface
# In recent Gradio versions, ChatInterface uses the message format by default.
demo = gr.ChatInterface(
    fn=chat_function,
    title="CrediTrust Complaint Assistant",
    description="Ask questions about customer complaints and get insights based on real data.",
    examples=[
        "What are the common issues with Credit Cards?", 
        "How are mortgage payments handled?",
        "Why was my account closed?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
