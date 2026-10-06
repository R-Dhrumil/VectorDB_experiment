from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.api.router import router as documents_router
from app.database.vector_store import get_vector_store
from app.processing.chunker import chunk_document_blocks

app = FastAPI(
    title="VectorDB API - RAG Document Processor",
    description="Backend service for uploading, extracting, chunking, and querying multi-format documents (PDF, DOCX, XLSX, TXT) with ChromaDB / Vector Store.",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular router
app.include_router(documents_router)

# --- Legacy Compatibility Models and Endpoints ---
class LegacyStoreRequest(BaseModel):
    text: str
    strategy: str = "recursive"
    metadata: dict = {}

class LegacyQueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/store", tags=["legacy"])
def store_document_legacy(request: LegacyStoreRequest):
    """
    Legacy raw text ingestion endpoint.
    """
    try:
        blocks = [{"text": request.text, "metadata": request.metadata}]
        chunks = chunk_document_blocks(
            blocks=blocks,
            doc_id="legacy_raw_text",
            file_name="raw_input.txt",
            strategy=request.strategy
        )
        vs = get_vector_store()
        vs.add_documents(chunks)
        return {
            "message": "Successfully embedded and stored document",
            "chunks_created": len(chunks),
            "strategy_used": request.strategy
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query", tags=["legacy"])
def query_database_legacy(request: LegacyQueryRequest):
    """
    Legacy similarity search endpoint.
    """
    try:
        vs = get_vector_store()
        results = vs.similarity_search(request.query, k=request.top_k)
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
        return {
            "query": request.query,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
