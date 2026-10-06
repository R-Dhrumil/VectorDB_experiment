import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

# Security and file processing bounds
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max file size limit
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"}

# Embedding & Vector DB Config
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
