import os
from pypdf import PdfReader

RAW_PDF_DIR = "data/raw_pdfs/"
OUTPUT_TEXT_DIR = "data/processed_text/"

os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


for pdf_file in os.listdir(RAW_PDF_DIR):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
        print(f"Processing: {pdf_file}")

        extracted_text = extract_text_from_pdf(pdf_path)

        output_file = pdf_file.replace(".pdf", ".txt")
        output_path = os.path.join(OUTPUT_TEXT_DIR, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

print("All PDFs converted to text successfully Ariz.")
