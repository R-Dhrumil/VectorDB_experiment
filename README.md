# VectorDB & RAG Document Processor

A modular FastAPI backend for extracting, chunking, embedding, and storing multi-format documents (`PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`, `MD`) in a local vector database (`ChromaDB` / `HuggingFace` embeddings).

---

## 📁 Project Architecture

```text
VectorDB_experiment/
├── BE/                           # Backend Application Root
│   ├── app/                      # Main Modular Application
│   │   ├── api/                  # REST API Router Endpoints
│   │   ├── core/                 # App Settings & Configurations
│   │   ├── database/             # Vector Store & Embedding Manager
│   │   ├── extractors/           # Multi-Format Text Readers (PDF, DOCX, XLSX, TXT)
│   │   ├── processing/           # Text Sanitization & Recursive Chunker
│   │   └── config.py             # Config Re-exporter
│   ├── tests/                    # Automated Test Suites
│   │   ├── test_backend.py
│   │   └── verify_full_be.py
│   ├── .env.example              # Sample Environment Setup
│   ├── main.py                   # FastAPI Application Entrypoint
│   └── requirements.txt          # Backend Dependencies
├── .gitignore                    # Git Ignore Configuration
├── pyrightconfig.json            # IDE Type Checker Settings
└── README.md                     # Project Documentation
```

---

## 🚀 Quick Start (Backend)

### 1. Create and Activate Virtual Environment
```powershell
cd BE
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Start Development Server
```powershell
python -m uvicorn main:app --reload --port 8000
```
Interactive API documentation will be available at: **`http://localhost:8000/docs`**

---

## 🧪 Running Tests

```powershell
cd BE
.\venv\Scripts\python.exe -m unittest discover -s tests
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload PDF, DOCX, XLSX, or TXT file for background chunking & embedding |
| `GET` | `/documents/tasks/{task_id}` | Check async file ingestion job progress |
| `POST` | `/documents/query` | Vector similarity search query across stored document chunks |
| `GET` | `/documents` | List all ingested documents and chunk counts |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its vector chunks |
