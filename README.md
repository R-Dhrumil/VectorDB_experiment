# VectorDB & RAG Document Processor

A modular FastAPI backend and React frontend for extracting, chunking, embedding, and interacting with multi-format documents (`PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`, `MD`) using a local vector database (`ChromaDB`) and Google Gemini LLM.

---

## 📁 Project Architecture

```text
VectorDB_experiment/
├── BE/                           # Backend Application Root (FastAPI)
│   ├── app/                      # Main Modular Application
│   │   ├── api/                  # REST API Router Endpoints
│   │   ├── core/                 # App Settings & Configurations
│   │   ├── database/             # Vector Store & Embedding Manager
│   │   ├── extractors/           # Multi-Format Text Readers
│   │   └── processing/           # Text Sanitization, Chunking & LLM Gen
│   ├── .env.example              # Sample Environment Setup
│   └── main.py                   # FastAPI Application Entrypoint
├── FE/                           # Frontend Application Root (React + Vite)
│   ├── src/                      # React Components and CSS
│   ├── package.json              # NPM Dependencies
│   └── vite.config.js            # Vite Configuration
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

### 3. Setup Environment Variables
Before running the backend, you need to provide your Gemini API key.
1. Copy the `.env.example` file and rename it to `.env` inside the `BE` folder.
2. Open `BE/.env` and paste your actual Gemini API Key:
```text
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Start Development Server (Backend)
To start the backend server, run the following command from inside the `BE` folder:
```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```
Interactive API documentation will be available at: **`http://localhost:8000/docs`**

---

## 🎨 Quick Start (Frontend)

To run the React application for uploading documents and asking questions:

```powershell
cd FE
npm install
npm run dev
```
The application will launch on your local network, usually at: **`http://localhost:5173/`**

---

## 📡 Key API Endpoints (Backend)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload PDF, DOCX, XLSX, or TXT file for background chunking & embedding |
| `POST` | `/documents/ask` | Ask a question about ingested documents (uses RAG + Gemini) |
| `POST` | `/documents/query` | Vector similarity search query across stored document chunks |
| `GET` | `/documents/tasks/{task_id}` | Check async file ingestion job progress |
| `GET` | `/documents` | List all ingested documents and chunk counts |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its vector chunks |
