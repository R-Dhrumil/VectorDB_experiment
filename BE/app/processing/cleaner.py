import re

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text from document files:
    - Replaces null characters
    - Normalizes multiple spaces and indentation artifacts
    - Strips trailing whitespace on each line
    - Retains single blank line breaks between paragraphs
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip whitespace from each individual line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Normalize excessive blank lines to max 2 newlines (paragraph boundary)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
