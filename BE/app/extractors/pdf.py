import io
from pypdf import PdfReader

def extract_pdf(file_bytes: bytes) -> list[dict]:
    """
    Extracts text page-by-page from PDF files.
    Returns a list of dicts with 'text' and 'metadata' (page_number).
    """
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    pages_content = []

    for index, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_content.append({
                "text": page_text.strip(),
                "metadata": {
                    "page_number": index + 1,
                    "total_pages": len(reader.pages)
                }
            })

    return pages_content
