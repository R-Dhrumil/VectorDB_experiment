import io
from docx import Document as DocxDocument

def extract_docx(file_bytes: bytes) -> list[dict]:
    """
    Extracts paragraphs and heading structures from DOCX files.
    """
    docx_file = io.BytesIO(file_bytes)
    doc = DocxDocument(docx_file)

    extracted_blocks = []
    current_section = "General"

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect heading styles if available
        if para.style and para.style.name.startswith("Heading"):
            current_section = text

        extracted_blocks.append({
            "text": text,
            "metadata": {
                "section": current_section
            }
        })

    return extracted_blocks
