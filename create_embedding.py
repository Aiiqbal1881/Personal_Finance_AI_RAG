import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CHUNKS_FILE_DIR = "data/processed_text/"
VECTOR_DB_DIR = "embeddings/finance_faiss"

all_texts = []

for txt_file in os.listdir(CHUNKS_FILE_DIR):
    if txt_file.endswith(".txt"):
        with open(os.path.join(CHUNKS_FILE_DIR, txt_file), "r", encoding="utf-8") as f:
            text = f.read()
            all_texts.append(text)

print(f"📄 Loaded {len(all_texts)} documents")

# FREE embeddings (NO API KEY REQUIRED)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS index
vector_db = FAISS.from_texts(all_texts, embeddings)

# Save vector DB
vector_db.save_local(VECTOR_DB_DIR)

print("FAISS vector database created successfully")
