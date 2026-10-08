import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import Dict, Any, List, Optional, Callable
from app.core.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    PG_TABLE_NAME,
    EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL
)
from app.database.embeddings import get_embedding

# Background task state registry
_background_tasks_db: Dict[str, Dict[str, Any]] = {}


def get_db_connection():
    """
    Creates and returns a connection to the PostgreSQL database.
    """
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        connect_timeout=5
    )


def init_db():
    """
    Initializes PostgreSQL with pgvector:
    1. Installs the vector extension if not present.
    2. Creates the document chunks table with vector column.
    3. Creates HNSW index for fast approximate nearest neighbor search.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 2. Create chunks table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE_NAME} (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id VARCHAR(64) NOT NULL,
                    file_name VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Create HNSW vector indexes for supported distance metrics
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE_NAME}_embedding_hnsw_idx 
                ON {PG_TABLE_NAME} USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE_NAME}_embedding_hnsw_l2_idx 
                ON {PG_TABLE_NAME} USING hnsw (embedding vector_l2_ops);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE_NAME}_embedding_hnsw_ip_idx 
                ON {PG_TABLE_NAME} USING hnsw (embedding vector_ip_ops);
            """)

            # 4. Create standard indexes for filtering and deletions
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE_NAME}_doc_id_idx 
                ON {PG_TABLE_NAME} (doc_id);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE_NAME}_file_name_idx 
                ON {PG_TABLE_NAME} (file_name);
            """)

        conn.commit()
    finally:
        conn.close()


def add_document_chunks(
    chunks: List[Any], 
    doc_id: str, 
    file_name: str, 
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> int:
    """
    Embeds document chunks using Ollama in batches and stores them in PostgreSQL pgvector.
    
    :param chunks: List of LangChain Document objects.
    :param doc_id: Unique document identifier.
    :param file_name: Name of the original file.
    :param model_name: Ollama embedding model (e.g. nomic-embed-text).
    :param progress_callback: Optional callback receiving (processed_count, total_count).
    :return: Number of inserted chunks.
    """
    if not chunks:
        return 0

    init_db()

    total_chunks = len(chunks)
    batch_size = 15  # Process 15 chunks per batch to prevent Ollama timeout
    all_embeddings = []

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_contents = [doc.page_content for doc in batch]
        
        batch_embs = get_embedding(batch_contents, model=model_name, is_query=False)
        all_embeddings.extend(batch_embs)
        
        if progress_callback:
            progress_callback(min(i + len(batch), total_chunks), total_chunks)

    # Format vector into pgvector string format '[0.1,0.2,...]'
    records = []
    for idx, (doc, emb) in enumerate(zip(chunks, all_embeddings)):
        meta = doc.metadata.copy() if hasattr(doc, "metadata") and doc.metadata else {}
        meta_json = json.dumps(meta)
        emb_str = f"[{','.join(str(val) for val in emb)}]"
        records.append((doc_id, file_name, idx, doc.page_content, meta_json, emb_str))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            insert_query = f"""
                INSERT INTO {PG_TABLE_NAME} 
                (doc_id, file_name, chunk_index, content, metadata, embedding)
                VALUES %s;
            """
            execute_values(cur, insert_query, records, template="(%s, %s, %s, %s, %s, %s::vector)")
        conn.commit()
        return len(records)
    finally:
        conn.close()


SUPPORTED_METRICS = {
    "cosine": {"operator": "<=>", "name": "cosine"},
    "l2": {"operator": "<->", "name": "l2"},
    "euclidean": {"operator": "<->", "name": "l2"},
    "inner_product": {"operator": "<#>", "name": "inner_product"},
    "dot": {"operator": "<#>", "name": "inner_product"},
    "ip": {"operator": "<#>", "name": "inner_product"}
}


def similarity_search(
    query: str, 
    k: int = 5, 
    file_name_filter: Optional[str] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    distance_metric: str = "cosine"
) -> List[Dict[str, Any]]:
    """
    Computes query vector with Ollama and retrieves the top-K most similar chunks from pgvector.
    Supports selectable mathematical distance formulas:
      - 'cosine': Cosine distance operator '<=>'
      - 'l2' (or 'euclidean'): Euclidean distance operator '<->'
      - 'inner_product' (or 'dot'): Inner product distance operator '<#>'
    
    :param query: User query text.
    :param k: Top K chunks to retrieve.
    :param file_name_filter: Optional filter by specific file_name.
    :param model_name: Embedding model name.
    :param distance_metric: Distance metric formula ('cosine', 'l2', or 'inner_product').
    :return: List of dicts with content, metadata, distance_score, and distance_metric.
    """
    init_db()

    metric_key = (distance_metric or "cosine").lower().strip()
    if metric_key not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. "
            f"Allowed metrics: 'cosine', 'l2' (euclidean), 'inner_product' (dot)"
        )

    metric_info = SUPPORTED_METRICS[metric_key]
    op = metric_info["operator"]
    canonical_metric = metric_info["name"]

    # 1. Embed user query using search_query prefix
    query_vector = get_embedding(query, model=model_name, is_query=True)
    query_vector_str = f"[{','.join(str(val) for val in query_vector)}]"

    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if file_name_filter:
                sql = f"""
                    SELECT id, doc_id, file_name, chunk_index, content, metadata,
                           (embedding {op} %s::vector) AS distance_score
                    FROM {PG_TABLE_NAME}
                    WHERE file_name = %s
                    ORDER BY embedding {op} %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, (query_vector_str, file_name_filter, query_vector_str, k))
            else:
                sql = f"""
                    SELECT id, doc_id, file_name, chunk_index, content, metadata,
                           (embedding {op} %s::vector) AS distance_score
                    FROM {PG_TABLE_NAME}
                    ORDER BY embedding {op} %s::vector
                    LIMIT %s;
                """
                cur.execute(sql, (query_vector_str, query_vector_str, k))

            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "content": row["content"],
                "metadata": row["metadata"] or {},
                "distance_score": round(float(row["distance_score"]), 4),
                "distance_metric": canonical_metric
            })
        return results
    finally:
        conn.close()


def list_documents() -> List[Dict[str, Any]]:
    """
    Returns a distinct list of all ingested documents along with chunk counts.
    """
    init_db()
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT doc_id, file_name, COUNT(*) AS chunk_count, MIN(created_at) AS created_at
                FROM {PG_TABLE_NAME}
                GROUP BY doc_id, file_name
                ORDER BY created_at DESC;
            """)
            rows = cur.fetchall()
            return [
                {
                    "doc_id": r["doc_id"],
                    "file_name": r["file_name"],
                    "chunk_count": int(r["chunk_count"])
                }
                for r in rows
            ]
    finally:
        conn.close()


def delete_document_by_id(doc_id: str) -> int:
    """
    Deletes all chunks associated with doc_id from pgvector.
    """
    init_db()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {PG_TABLE_NAME} WHERE doc_id = %s;", (doc_id,))
            deleted_count = cur.rowcount
        conn.commit()
        return deleted_count
    finally:
        conn.close()


# Background task status helpers
def set_task_status(task_id: str, status: str, details: Optional[Dict[str, Any]] = None):
    _background_tasks_db[task_id] = {
        "status": status,
        "details": details or {},
    }


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    return _background_tasks_db.get(task_id)


# Legacy Compatibility Wrapper
class PgVectorStoreWrapper:
    def add_documents(self, documents: List[Any]):
        if not documents:
            return
        doc_id = documents[0].metadata.get("doc_id", "legacy_doc")
        file_name = documents[0].metadata.get("file_name", "legacy_file.txt")
        return add_document_chunks(documents, doc_id=doc_id, file_name=file_name)

    def similarity_search(self, query: str, k: int = 3):
        results = similarity_search(query, k=k)
        # Mock LangChain Document interface for backwards compatibility
        class SimpleDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        return [SimpleDoc(r["content"], r["metadata"]) for r in results]


def get_vector_store() -> PgVectorStoreWrapper:
    return PgVectorStoreWrapper()
