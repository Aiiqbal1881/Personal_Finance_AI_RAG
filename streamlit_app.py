import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(
    page_title="Personal Finance Advisor (AI RAG)",
    page_icon="💰",
    layout="centered"
)

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get AI-based answers from financial documents.")

# Load Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# Load FAISS index and documents
@st.cache_resource
def load_faiss():
    index = faiss.read_index("embeddings/finance_faiss/index.faiss")
    with open("embeddings/finance_faiss/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss()

# Input
question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. Difference between trading account and demat account"
)

if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is thinking..."):
            # Embed question
            q_embedding = embedder.encode([question])
            D, I = index.search(q_embedding, k=3)

            # Build context safely
            context = ""
            for idx in I[0]:
                if idx < len(documents):
                    context += documents[idx] + "\n\n"

            prompt = f"""
You are a personal finance advisor.
Answer the question ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

            # Groq LLM call
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful financial assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            answer = response.choices[0].message.content
            st.success("Answer:")
            st.write(answer)
