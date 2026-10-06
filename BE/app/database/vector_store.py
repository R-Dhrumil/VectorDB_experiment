import os
from typing import Dict, Any, List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import PERSIST_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Global vector store singleton & background task state registry
_embeddings_instance = None
_vector_store_instance = None
_background_tasks_db: Dict[str, Dict[str, Any]] = {}

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance

def get_vector_store() -> Chroma:
    global _vector_store_instance
    if _vector_store_instance is None:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        _vector_store_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIRECTORY
        )
    return _vector_store_instance

def set_task_status(task_id: str, status: str, details: Optional[Dict[str, Any]] = None):
    _background_tasks_db[task_id] = {
        "status": status,
        "details": details or {},
    }

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _background_tasks_db.get(task_id)

def list_documents() -> List[Dict[str, Any]]:
    vs = get_vector_store()
    raw_data = vs.get()
    metadatas = raw_data.get("metadatas", [])

    docs_map: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "file_name": meta.get("file_name", "Unknown"),
                "chunk_count": 0
            }
        docs_map[doc_id]["chunk_count"] += 1

    return list(docs_map.values())

def delete_document_by_id(doc_id: str) -> int:
    vs = get_vector_store()
    raw_data = vs.get(where={"doc_id": doc_id})
    ids_to_delete = raw_data.get("ids", [])
    if ids_to_delete:
        vs.delete(ids=ids_to_delete)
    return len(ids_to_delete)
