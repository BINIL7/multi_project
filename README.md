# Multi-Modal RAG Prototype

This project is a simple multi-modal RAG system. 
It takes PDF files from the docs folder and extracts text, tables and images (OCR). 
All the extracted content is converted into embeddings and stored in a FAISS index. 
A small Streamlit UI is provided to test the retrieval.

---

## Files in the project

- ingestion.py       → Reads PDF, extracts text/table/image OCR, makes chunks and embeddings, builds FAISS index
- retriever.py       → Loads FAISS index and metadata, runs vector search for a given query
- app_streamlit.py   → Simple Streamlit UI to enter a question and view the retrieved results
- requirements.txt   → Python dependencies
- README.md          → This file

Folders:

indexes/
    faiss.index
    metadata.json

docs/
    (Place your PDF file here. Example: bini1.pdf)

---

## How to Install

1. Install the required packages:
   pip install -r requirements.txt

2. Place your PDF file inside the docs/ folder.
   Example:
   docs/bini1.pdf

---

## How to Run

### 1) Build the FAISS index
   python ingestion.py --input "./docs" --index_path "./indexes/faiss.index"

### 2) Start the Streamlit app
   streamlit run app_streamlit.py
