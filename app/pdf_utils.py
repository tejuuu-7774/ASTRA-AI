import fitz  # PyMuPDF

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in pdf:
        text += page.get_text()

    print("\n========== PDF TEXT PREVIEW ==========\n")
    print(text[:1000])
    print("\n========== END PREVIEW ==========\n")

    return text