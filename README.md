# VectorDB & RAG Document Processor (pgvector + Ollama)

A modular FastAPI backend and React frontend for extracting, chunking, embedding, and interacting with multi-format documents (`PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`, `MD`) using **PostgreSQL (`pgvector`)**, **Ollama (`nomic-embed-text`)**, and Google Gemini LLM.

---

## 📁 Project Architecture

```text
VectorDB_experiment/
├── BE/                           # Backend Application Root (FastAPI)
│   ├── app/                      # Main Modular Application
│   │   ├── api/                  # REST API Router Endpoints
│   │   ├── core/                 # App Settings & Configurations
│   │   ├── database/             # PostgreSQL pgvector & Ollama Embeddings
│   │   ├── extractors/           # Multi-Format Text Readers
│   │   └── processing/           # Text Sanitization, Chunking & LLM Gen
│   ├── .env.example              # Sample Environment Setup
│   ├── requirements.txt          # Python Dependencies
│   └── main.py                   # FastAPI Application Entrypoint
├── FE/                           # Frontend Application Root (React + Vite)
│   ├── src/                      # React Components and CSS
│   ├── package.json              # NPM Dependencies
│   └── vite.config.js            # Vite Configuration
└── README.md                     # Project Documentation
```

---

## 🛠️ Prerequisites

1. **PostgreSQL with `pgvector` Extension**
   - Ensure PostgreSQL is running.
   - Run in pgAdmin or psql:
     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     ```
2. **Ollama (for Local Embeddings)**
   - Start Ollama:
     ```bash
     ollama serve
     ```
   - Pull the `nomic-embed-text` embedding model:
     ```bash
     ollama pull nomic-embed-text
     ```

---

## 🚀 Quick Start (Backend)

### 1. Create and Activate Virtual Environment
```bash
cd BE
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Configure your database credentials and Gemini API Key in `BE/.env`:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=postgres
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   OLLAMA_BASE_URL=http://localhost:11434
   EMBEDDING_MODEL=nomic-embed-text
   ```

### 4. Start Backend Server
```bash
uvicorn main:app --reload --port 8000
```
Interactive API documentation will be available at: **`http://localhost:8000/docs`**

---

## 🎨 Quick Start (Frontend)

```bash
cd FE
npm install
npm run dev
```
The application will launch at **`http://localhost:5173/`**.

---

## 📡 Key API Endpoints (Backend)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload file for background chunking, Ollama embedding & pgvector storage |
| `POST` | `/documents/ask` | Ask a question about ingested documents (uses pgvector + Gemini) |
| `POST` | `/documents/query` | Vector similarity search query across stored document chunks |
| `GET` | `/documents/tasks/{task_id}` | Check async file ingestion job progress |
| `GET` | `/documents` | List all ingested documents and chunk counts |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its vector chunks |
