import os
import pickle
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
VECTOR_DB_DIR = "embeddings/manual_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

MAX_CONTEXT_CHARS = 3000   # 🔥 prevents token overflow
TOP_K = 2                 # 🔥 safe retrieval count

# --------------------------------------------------
# Load Groq client
# --------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --------------------------------------------------
# Load embedding model
# --------------------------------------------------
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------------------------------------------------
# Load FAISS index + documents
# --------------------------------------------------
index = faiss.read_index(os.path.join(VECTOR_DB_DIR, "index.faiss"))

with open(os.path.join(VECTOR_DB_DIR, "documents.pkl"), "rb") as f:
    documents = pickle.load(f)

# --------------------------------------------------
# Utility: truncate context safely
# --------------------------------------------------
def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# --------------------------------------------------
# MAIN RAG FUNCTION
# --------------------------------------------------
def ask_finance_question(question: str) -> str:
    """
    Takes a finance-related question and returns an answer
    using FAISS + Groq (RAG).
    """

    # 1️⃣ Embed query
    query_vector = embedder.encode([question]).astype("float32")

    # 2️⃣ Retrieve relevant chunks
    _, indices = index.search(query_vector, TOP_K)

    # 3️⃣ Build context
    context = ""
    for idx in indices[0]:
        if idx < len(documents):
            context += documents[idx] + "\n\n"

    # 4️⃣ Truncate context (CRITICAL)
    context = truncate_text(context)

    # 5️⃣ Build prompt
    prompt = f"""
You are a Personal Finance Advisor.

Use the context below as the primary source.
If the context is insufficient, provide general financial advice and clearly mention that it is general guidance.

Context:
{context}

Question:
{question}

Answer:
"""


    # 6️⃣ Call Groq LLM
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # 7️⃣ Return final answer
    return response.choices[0].message.content.strip()
