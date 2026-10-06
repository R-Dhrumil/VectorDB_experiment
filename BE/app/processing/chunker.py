from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.documents import Document
from app.processing.cleaner import clean_text
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def chunk_document_blocks(
    blocks: list[dict],
    doc_id: str,
    file_name: str,
    strategy: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Document]:
    """
    Takes extracted blocks (with text and block metadata), cleans them, and creates
    chunked LangChain Document objects ready for embedding and vector database storage.
    """
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "sentence":
        splitter = CharacterTextSplitter(
            separator=". ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "paragraph":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Options: fixed, sentence, paragraph, recursive")

    all_chunks: list[Document] = []
    chunk_counter = 0

    for block in blocks:
        raw_text = block.get("text", "")
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        base_meta = block.get("metadata", {}).copy()
        base_meta["doc_id"] = doc_id
        base_meta["file_name"] = file_name

        sub_docs = splitter.create_documents([cleaned], metadatas=[base_meta])
        for doc in sub_docs:
            doc.metadata["chunk_index"] = chunk_counter
            all_chunks.append(doc)
            chunk_counter += 1

    return all_chunks
