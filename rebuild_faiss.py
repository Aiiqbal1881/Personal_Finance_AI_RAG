import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

TEXT_DIR = "data/processed_text"
VECTOR_DB_DIR = "embeddings/manual_faiss"

os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = []

# Load all text files
for file in os.listdir(TEXT_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(TEXT_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)

print(f"Loaded {len(documents)} documents")

# Create embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and documents
faiss.write_index(index, f"{VECTOR_DB_DIR}/index.faiss")

with open(f"{VECTOR_DB_DIR}/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Manual FAISS index created successfully")
