import pdfplumber
import os

def dump_pdf(pdf_path, txt_path):
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found.")
        return
    
    print(f"Extracting text from {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"Saved extracted text to {txt_path} ({len(text)} characters).")

if __name__ == "__main__":
    pdf_files = []
    if os.path.exists("quotations"):
        pdf_files = [os.path.join("quotations", f) for f in os.listdir("quotations") if f.endswith(".pdf")]
        pdf_files.sort()
        
    if len(pdf_files) >= 2:
        dump_pdf(pdf_files[0], "extracted_text1.txt")
        dump_pdf(pdf_files[1], "extracted_text2.txt")
    else:
        dump_pdf("quote1.pdf", "extracted_text1.txt")
        dump_pdf("quote2.pdf", "extracted_text2.txt")
