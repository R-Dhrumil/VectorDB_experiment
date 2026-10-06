import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.api.router import router as documents_router
from app.database.vector_store import get_vector_store, init_db, add_document_chunks, similarity_search
from app.processing.chunker import chunk_document_blocks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vectordb-app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Attempt to initialize pgvector on startup
    try:
        init_db()
        logger.info("Successfully connected to PostgreSQL and verified pgvector extension.")
    except Exception as e:
        logger.warning(
            "Could not connect to PostgreSQL on startup. "
            "Ensure PostgreSQL is running and credentials in .env are correct. "
            f"Details: {str(e)}"
        )
    yield


app = FastAPI(
    title="VectorDB API - PostgreSQL (pgvector) + Ollama RAG",
    description="Backend service for uploading, chunking, and querying multi-format documents using PostgreSQL pgvector and Ollama nomic-embed-text.",
    version="2.1.0",
    lifespan=lifespan
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
        count = add_document_chunks(chunks=chunks, doc_id="legacy_raw_text", file_name="raw_input.txt")
        return {
            "message": "Successfully embedded and stored document in pgvector",
            "chunks_created": count,
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
        results = similarity_search(request.query, k=request.top_k)
        formatted_results = [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "distance_score": r["distance_score"]
            }
            for r in results
        ]
        return {
            "query": request.query,
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
