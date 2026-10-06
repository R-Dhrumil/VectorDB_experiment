import os
from app.extractors.pdf import extract_pdf
from app.extractors.docx import extract_docx
from app.extractors.excel import extract_excel
from app.extractors.txt import extract_txt

def extract_document_blocks(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Routes document bytes to the appropriate format extractor based on file extension.
    Sanitizes filename against path traversal.
    """
    safe_filename = os.path.basename(filename)
    ext = os.path.splitext(safe_filename)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_bytes)
    elif ext == ".docx":
        return extract_docx(file_bytes)
    elif ext in [".xlsx", ".xls", ".csv"]:
        return extract_excel(file_bytes)
    elif ext in [".txt", ".md"]:
        return extract_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format extension: '{ext}'")
