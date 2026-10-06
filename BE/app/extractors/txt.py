def extract_txt(file_bytes: bytes) -> list[dict]:
    """
    Extracts text from plain text or Markdown files using UTF-8 decoding (fallback to latin-1).
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="replace")

    return [{
        "text": text,
        "metadata": {}
    }]
