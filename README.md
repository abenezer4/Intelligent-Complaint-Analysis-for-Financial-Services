# CrediTrust: Intelligent Complaint Analysis Chatbot (RAG)

## 📌 Project Overview
CrediTrust Financial is a digital finance company serving East African markets. This project implements a **Retrieval-Augmented Generation (RAG)** chatbot designed to help Product Managers and Support teams transform thousands of unstructured customer complaints into actionable insights.

Instead of manually reading thousands of narratives, users can ask questions in plain English and receive instant, evidence-backed answers derived from the **Consumer Financial Protection Bureau (CFPB)** dataset.

---

## 🚀 Key Features
- **Semantic Search:** Uses vector embeddings to find complaints based on meaning, not just keywords.
- **Evidence-Backed Answers:** The chatbot provides a summary answer and displays the actual source narratives used for transparency.
- **Large-Scale Data Handling:** Optimized to handle over **1.3 million text chunks** using pre-computed embeddings.
- **Modular Architecture:** Professional code structure with reusable modules for data loading, indexing, and RAG logic.

---

## 🛠️ Technical Implementation

### Task 1: EDA & Preprocessing
- Cleaned the CFPB dataset by filtering for specific financial products.
- Normalized text narratives (lowercasing, removing special characters) to improve embedding quality.

### Task 2: Chunking & Embeddings
- **Strategy:** Used `RecursiveCharacterTextSplitter` with a chunk size of 500 characters and 50-character overlap.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` for high-accuracy semantic mapping.
- **Indexing:** Built a fast similarity search index using **FAISS**.

### Task 3: RAG Pipeline & Evaluation
- **Retriever:** Searches the FAISS index to find the most relevant context for any query.
- **Generator:** Employs the `google/flan-t5-base` LLM to synthesize answers grounded strictly in the retrieved context.
- **Quality Analysis:** Achieved an average quality score of **4.6/5** on real-world test cases.

### Task 4: Interactive UI
- Built a web-based chat interface using **Gradio**.
- Designed for non-technical stakeholders to get answers without needing data analysts.

---

## 📁 Project Structure
```text
rag-complaint-chatbot/
├── app.py              # The Gradio web application (User Interface)
├── FINAL_REPORT.md     # Comprehensive project analysis (Medium post format)
├── src/                # Modular Python source code
│   ├── data_loader.py  # Data loading utilities
│   ├── vector_store_manager.py # FAISS index management
│   ├── rag_system.py   # Core RAG pipeline logic
│   └── ingest_prebuilt_embeddings.py # Fast-track ingestion for large datasets
├── notebooks/          # Interactive experimentation & evaluation
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding.ipynb
│   └── task3_rag_evaluation.ipynb
├── tests/              # Automated verification
│   └── test_rag_pipeline.py
├── vector_store/       # Persisted FAISS index and metadata
└── requirements.txt    # Project dependencies
```

---

## ⚙️ Setup and Installation

1. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Ingestion (Full Dataset):**
   - Download `complaint_embeddings.parquet` from the link provided in the assignment.
   - Run the ingestion script to build the full vector index:
     ```bash
     python src/ingest_prebuilt_embeddings.py
     ```

---

## 🖥️ How to Run

### 1. Launch the Chatbot
```bash
python app.py
```
*Open the local URL (usually http://127.0.0.1:7860) to interact with the AI.*

### 2. Run the Evaluation Notebook
Explore the qualitative analysis in `notebooks/task3_rag_evaluation.ipynb`.

### 3. Run Pipeline Tests
```bash
python tests/test_rag_pipeline.py
```

---

## 👥 Team & Acknowledgments
- **Facilitators:** Kerod, Mahbubah, Filimon, Smegnsh
- **Organization:** 10 Academy Artificial Intelligence Mastery (Week 7 Challenge)
- **Data Source:** Consumer Financial Protection Bureau (CFPB)
