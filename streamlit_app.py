import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="Personal Finance Advisor (AI RAG)",
    page_icon="💰",
    layout="centered"
)

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get AI-based answers from financial documents.")

# -------------------------------
# Load Groq API Key
# -------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit secrets")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -------------------------------
# Load Embedding Model
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# -------------------------------
# Load FAISS
# -------------------------------
@st.cache_resource
def load_faiss():
    index = faiss.read_index("embeddings/finance_faiss/index.faiss")
    with open("embeddings/finance_faiss/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss()

# -------------------------------
# User Input
# -------------------------------
question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. What is a trading account?"
)

# -------------------------------
# Ask AI
# -------------------------------
if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                # Embed question
                q_embedding = embedder.encode([question])

                # Search FAISS
                _, indices = index.search(q_embedding, k=3)

                # Build context (SAFE)
                context = ""
                for idx in indices[0]:
                    if idx < len(documents):
                        context += documents[idx] + "\n"

                # HARD LIMIT context (Groq-safe)
                context = context[:2000]

                prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=400
                )

                answer = response.choices[0].message.content.strip()

                st.success("Answer:")
                st.write(answer)

            except Exception as e:
                st.error("❌ Error while generating answer. Please try again.")
