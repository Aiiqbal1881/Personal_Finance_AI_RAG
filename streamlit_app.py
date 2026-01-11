import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(
    page_title="Personal Finance Advisor (AI RAG)",
    page_icon="💰",
    layout="centered"
)

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get AI-based answers from financial documents.")

# -------------------------------
# Load Groq API key (Streamlit Secrets)
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# -------------------------------
# Load FAISS index + documents
# -------------------------------
@st.cache_resource
def load_faiss():
    index = faiss.read_index("embeddings/finance_faiss/index.faiss")
    with open("embeddings/finance_faiss/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss()

# -------------------------------
# User input
# -------------------------------
question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. Difference between trading and demat account"
)

# -------------------------------
# Ask AI
# -------------------------------
if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                # Embed question
                q_embedding = embedder.encode([question])
                _, I = index.search(q_embedding, k=3)

                # Build context (LIMITED)
                context = ""
                for idx in I[0]:
                    if idx < len(documents):
                        context += documents[idx] + "\n\n"

                context = context[:1200]  # VERY IMPORTANT

                prompt = f"""
You are a personal finance advisor.
Answer the question ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

                # -------------------------------
                # Groq LLM call (SUPPORTED MODEL)
                # -------------------------------
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=256
                )

                answer = response.choices[0].message.content
                st.success("Answer:")
                st.write(answer)

            except Exception as e:
                st.error("❌ Error while generating answer")
                st.code(str(e))
