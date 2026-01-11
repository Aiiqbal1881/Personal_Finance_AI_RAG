import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Personal Finance Advisor (AI RAG)",
    page_icon="💰",
    layout="centered"
)

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get AI-based answers from financial documents.")

# --------------------------------------------------
# Load Groq API Key from Streamlit Secrets
# --------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# --------------------------------------------------
# Load FAISS Index & Documents
# --------------------------------------------------
@st.cache_resource
def load_faiss():
    index_path = "embeddings/finance_faiss/index.faiss"
    docs_path = "embeddings/finance_faiss/documents.pkl"

    if not os.path.exists(index_path):
        st.error("❌ FAISS index file not found.")
        st.stop()

    if not os.path.exists(docs_path):
        st.error("❌ documents.pkl file not found.")
        st.stop()

    index = faiss.read_index(index_path)

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    return index, documents

index, documents = load_faiss()

# --------------------------------------------------
# User Input
# --------------------------------------------------
question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. What is a demat account?"
)

# --------------------------------------------------
# Ask AI Button
# --------------------------------------------------
if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is thinking..."):
            # Embed the question
            query_embedding = embedder.encode([question])

            # Search FAISS
            distances, indices = index.search(query_embedding, k=3)

            # Build context safely
            context = ""
            for idx in indices[0]:
                if idx < len(documents):
                    context += documents[idx] + "\n\n"

            # Hard limit context to avoid Groq errors
            context = context[:3000]

            # Prompt
            prompt = f"""
You are a personal finance advisor.
Answer the question ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=512
                )

                answer = response.choices[0].message.content.strip()
                st.success("Answer:")
                st.write(answer)

            except Exception as e:
                st.error("❌ Error while generating answer. Please try again.")
