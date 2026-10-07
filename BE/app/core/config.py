import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

# Security and file processing bounds
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max file size limit
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"}

# PostgreSQL & pgvector Config
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "document_chunks")

# Ollama & Embedding Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSION = 768

# LLM Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Chunking defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Backward compatibility aliases (for ChromaDB legacy imports)
COLLECTION_NAME = PG_TABLE_NAME
EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL

