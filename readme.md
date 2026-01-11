# Personal Finance Advisor (AI RAG)

This project is a Retrieval-Augmented Generation (RAG) based Personal Finance Advisor
that answers finance-related questions using document-grounded knowledge.

## Features
- FAISS-based semantic search
- Groq LLM for fast inference
- Streamlit frontend
- Flask backend
- Hallucination-safe responses

## Tech Stack
- Python
- FAISS
- Sentence Transformers
- Groq API
- Flask
- Streamlit

## How it works
1. User enters a finance question
2. Question is embedded using MiniLM
3. Relevant document chunks are retrieved using FAISS
4. Context is passed to Groq LLM
5. Answer is generated with grounding

## How to Run Locally
```bash
pip install -r requirements.txt
python backend/app.py
streamlit run frontend/streamlit_app.py
