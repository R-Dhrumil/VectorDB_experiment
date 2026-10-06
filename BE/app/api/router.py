import os
import uuid
import hashlib
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.config import MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from app.extractors.factory import extract_document_blocks
from app.processing.chunker import chunk_document_blocks
from app.database.vector_store import (
    get_vector_store,
    set_task_status,
    get_task_status,
    list_documents,
    delete_document_by_id
)
from app.processing.generator import generate_rag_answer

router = APIRouter(prefix="/documents", tags=["documents"])

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="User search query string")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top results to retrieve")
    file_name_filter: Optional[str] = Field(default=None, description="Optional filter by specific filename")

class AskRequest(BaseModel):
    query: str = Field(..., description="User question to be answered by the LLM")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    file_name_filter: Optional[str] = Field(default=None, description="Optional filter by specific filename")

class StoreTextRequest(BaseModel):
    text: str
    file_name: str = "raw_text.txt"
    strategy: str = "recursive"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

# --- Background Worker Function ---
def process_file_background(
    task_id: str,
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    try:
        set_task_status(task_id, "PROCESSING", {"doc_id": doc_id, "filename": filename})

        # 1. Extract text blocks based on extension
        blocks = extract_document_blocks(file_bytes, filename)

        # 2. Clean and chunk blocks into LangChain Documents
        chunks = chunk_document_blocks(
            blocks=blocks,
            doc_id=doc_id,
            file_name=filename,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            set_task_status(task_id, "FAILED", {"error": "No printable text could be extracted from the document."})
            return

        # 3. Add to ChromaDB vector store
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        set_task_status(task_id, "COMPLETED", {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "strategy": strategy
        })

    except Exception as e:
        set_task_status(task_id, "FAILED", {"error": str(e)})


# --- Endpoints ---

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    strategy: str = Form("recursive"),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP)
):
    """
    Accepts PDF, DOCX, XLSX, or TXT file uploads.
    Enqueues async text extraction, chunking, and embedding generation.
    """
    safe_filename = os.path.basename(file.filename or "uploaded_file.bin")
    ext = os.path.splitext(safe_filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    doc_id = f"doc_{file_hash[:12]}"
    task_id = str(uuid.uuid4())

    set_task_status(task_id, "QUEUED", {"doc_id": doc_id, "filename": safe_filename})

    background_tasks.add_task(
        process_file_background,
        task_id=task_id,
        doc_id=doc_id,
        file_bytes=file_bytes,
        filename=safe_filename,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return {
        "message": "File upload accepted. Processing in background.",
        "task_id": task_id,
        "doc_id": doc_id,
        "filename": safe_filename
    }


@router.get("/tasks/{task_id}")
def get_task_progress(task_id: str):
    """
    Returns the processing status of an async document ingestion job.
    """
    status_data = get_task_status(task_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return status_data


@router.post("/query")
def query_documents(request: QueryRequest):
    """
    Searches the vector database for chunks semantically similar to the query.
    """
    try:
        vs = get_vector_store()
        filter_dict = None
        if request.file_name_filter:
            filter_dict = {"file_name": request.file_name_filter}

        results_with_score = vs.similarity_search_with_score(
            request.query,
            k=request.top_k,
            filter=filter_dict
        )

        formatted_results = []
        for doc, score in results_with_score:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance_score": round(float(score), 4)
            })

        return {
            "query": request.query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Error: {str(e)}")


@router.post("/ask")
def ask_question(request: AskRequest):
    """
    Searches the vector database and uses Gemini LLM to generate an answer based on the retrieved context.
    """
    try:
        vs = get_vector_store()
        filter_dict = None
        if request.file_name_filter:
            filter_dict = {"file_name": request.file_name_filter}

        # 1. Retrieve relevant chunks
        results_with_score = vs.similarity_search_with_score(
            request.query,
            k=request.top_k,
            filter=filter_dict
        )

        formatted_results = []
        for doc, score in results_with_score:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance_score": round(float(score), 4)
            })
            
        # 2. Generate answer with Gemini
        answer = generate_rag_answer(request.query, formatted_results)

        return {
            "query": request.query,
            "answer": answer,
            "context_chunks_used": len(formatted_results),
            "sources": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search/Generation Error: {str(e)}")


@router.get("")
def get_all_documents():
    """
    Lists all documents currently stored in the vector database with chunk counts.
    """
    return {"documents": list_documents()}


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """
    Deletes a document and all of its associated vector chunks from the database.
    """
    deleted_count = delete_document_by_id(doc_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No document found with ID '{doc_id}'")
    return {"message": f"Successfully deleted document '{doc_id}' and {deleted_count} chunks."}
