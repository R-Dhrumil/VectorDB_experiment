import os
import uuid
import hashlib
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.config import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL
)
from app.extractors.factory import extract_document_blocks
from app.processing.chunker import chunk_document_blocks
from app.database.vector_store import (
    add_document_chunks,
    similarity_search,
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
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="Ollama embedding model to use")
    distance_metric: str = Field(default="cosine", description="Distance formula: cosine (<=>), l2 (<->), inner_product (<#>)")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text or paragraph to ingest")
    title: Optional[str] = Field(default=None, description="Optional label or title for the text snippet")
    strategy: str = Field(default="recursive", description="Chunking strategy: recursive, fixed, sentence, paragraph")
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=50, le=5000)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0, le=1000)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="Ollama embedding model to use")


class AskRequest(BaseModel):
    query: str = Field(..., description="User question to be answered by the LLM")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    file_name_filter: Optional[str] = Field(default=None, description="Optional filter by specific filename")
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="Ollama embedding model to use")
    distance_metric: str = Field(default="cosine", description="Distance formula: cosine (<=>), l2 (<->), inner_product (<#>)")




# --- Background Worker Function ---
def process_file_background(
    task_id: str,
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str
):
    try:
        set_task_status(task_id, "PROCESSING", {"doc_id": doc_id, "filename": filename})

        # 1. Extract text blocks based on file extension
        blocks = extract_document_blocks(file_bytes, filename)

        # 2. Clean and chunk blocks into Document objects
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

        # 3. Embed with Ollama and store in PostgreSQL pgvector
        inserted_count = add_document_chunks(
            chunks=chunks,
            doc_id=doc_id,
            file_name=filename,
            model_name=embedding_model
        )

        set_task_status(task_id, "COMPLETED", {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": inserted_count,
            "strategy": strategy,
            "embedding_model": embedding_model
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
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL)
):
    """
    Accepts PDF, DOCX, XLSX, or TXT file uploads.
    Enqueues async text extraction, chunking, and pgvector storage via Ollama embeddings.
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
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model
    )

    return {
        "message": "File upload accepted. Processing and embedding in background.",
        "task_id": task_id,
        "doc_id": doc_id,
        "filename": safe_filename,
        "embedding_model": embedding_model
    }


@router.post("/raw")
def ingest_raw_paragraph(request: IngestTextRequest):
    """
    Directly ingest and embed raw text or paragraph without requiring a file upload.
    """
    clean_snippet = request.text.strip()
    if not clean_snippet:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    text_hash = hashlib.sha256(clean_snippet.encode("utf-8")).hexdigest()
    doc_id = f"snippet_{text_hash[:10]}"
    display_title = (request.title.strip() if request.title else "") or f"paragraph_{text_hash[:6]}.txt"

    try:
        blocks = [{"text": clean_snippet, "metadata": {"source": "direct_input", "title": display_title}}]
        chunks = chunk_document_blocks(
            blocks=blocks,
            doc_id=doc_id,
            file_name=display_title,
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not generate any chunks from the provided text.")

        inserted_count = add_document_chunks(
            chunks=chunks,
            doc_id=doc_id,
            file_name=display_title,
            model_name=request.embedding_model
        )

        return {
            "message": "Paragraph successfully chunked, embedded, and stored in PostgreSQL (pgvector).",
            "doc_id": doc_id,
            "title": display_title,
            "chunks_created": inserted_count,
            "strategy": request.strategy,
            "embedding_model": request.embedding_model
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")



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
    Searches PostgreSQL pgvector for chunks semantically similar to the query.
    Supports distance_metric: 'cosine', 'l2', 'inner_product'.
    """
    try:
        results = similarity_search(
            query=request.query,
            k=request.top_k,
            file_name_filter=request.file_name_filter,
            model_name=request.embedding_model,
            distance_metric=request.distance_metric
        )

        return {
            "query": request.query,
            "embedding_model": request.embedding_model,
            "distance_metric": request.distance_metric,
            "results_count": len(results),
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")


@router.post("/ask")
def ask_question(request: AskRequest):
    """
    Searches pgvector using Ollama embeddings and uses Gemini LLM to generate an answer.
    Supports distance_metric: 'cosine', 'l2', 'inner_product'.
    """
    try:
        # 1. Retrieve relevant chunks from pgvector
        results = similarity_search(
            query=request.query,
            k=request.top_k,
            file_name_filter=request.file_name_filter,
            model_name=request.embedding_model,
            distance_metric=request.distance_metric
        )

        # 2. Generate answer with Gemini
        answer = generate_rag_answer(request.query, results)

        return {
            "query": request.query,
            "embedding_model": request.embedding_model,
            "distance_metric": request.distance_metric,
            "answer": answer,
            "context_chunks_used": len(results),
            "sources": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")


@router.get("")
def get_all_documents():
    """
    Lists all documents currently stored in PostgreSQL pgvector with chunk counts.
    """
    try:
        return {"documents": list_documents()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """
    Deletes a document and all of its associated vector chunks from PostgreSQL pgvector.
    """
    try:
        deleted_count = delete_document_by_id(doc_id)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"No document found with ID '{doc_id}'")
        return {"message": f"Successfully deleted document '{doc_id}' and {deleted_count} chunks from pgvector."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion Error: {str(e)}")
