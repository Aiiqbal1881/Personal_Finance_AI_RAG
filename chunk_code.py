import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

OUTPUT_TEXT_DIR = "data/processed_text/"

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

all_chunks = []

for txt_file in os.listdir(OUTPUT_TEXT_DIR):
    if txt_file.endswith(".txt"):
        with open(os.path.join(OUTPUT_TEXT_DIR, txt_file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

print(f"✅ Total text chunks created: {len(all_chunks)}")
