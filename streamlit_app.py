import os
import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ------------------ CONFIG ------------------

FAISS_INDEX_PATH = "embeddings/finance_faiss/index.faiss"
DOCUMENTS_PATH = "embeddings/finance_faiss/documents.pkl"
SIMILARITY_THRESHOLD = 1.2   # Lower = stricter RAG

# ------------------ STREAMLIT UI ------------------

st.set_page_config(
    page_title="Personal Finance Advisor (AI RAG)",
    page_icon="💰",
    layout="centered"
)

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get AI-powered answers.")

# ------------------ LOAD GROQ ------------------

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------ LOAD EMBEDDING MODEL ------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ------------------ LOAD FAISS ------------------

@st.cache_resource
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss()

# ------------------ USER INPUT ------------------

question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. Difference between trading account and demat account"
)

# ------------------ ASK AI ------------------

if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # 1️⃣ Embed question
                q_embedding = embedder.encode([question]).astype("float32")

                # 2️⃣ Search FAISS
                D, I = index.search(q_embedding, k=3)
                best_distance = D[0][0]

                # 3️⃣ Build context
                context = ""
                for idx in I[0]:
                    if idx < len(documents):
                        context += documents[idx] + "\n\n"

                # 4️⃣ Decide RAG strategy
                if best_distance <= SIMILARITY_THRESHOLD and context.strip():
                    strategy = "RAG"
                    prompt = f"""
You are a personal finance advisor.

Answer the question using the context below.
Explain clearly in simple language.

Context:
{context}

Question:
{question}
"""
                else:
                    strategy = "LLM_FALLBACK"
                    prompt = f"""
You are a personal finance advisor.

The knowledge base does not contain enough information.
Answer using your general financial knowledge clearly.

Question:
{question}
"""

                # 5️⃣ Call Groq
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a helpful financial expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

                answer = response.choices[0].message.content

                # 6️⃣ Display
                st.success("Answer:")
                st.write(answer)

                # (Optional debug info)
                with st.expander("Debug Info"):
                    st.write(f"Strategy used: {strategy}")
                    st.write(f"Best FAISS distance: {best_distance:.4f}")

            except Exception as e:
                st.error("Error while generating answer.")
                st.exception(e)
