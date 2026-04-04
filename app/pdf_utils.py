from pypdf import PdfReader
from io import BytesIO

# OLD (optional - can keep or remove)
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# NEW (IMPORTANT - used now)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""

    pdf_stream = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text