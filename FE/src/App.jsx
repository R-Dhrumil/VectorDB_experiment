import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';
import './index.css';

const API_BASE = "http://localhost:8000/documents";

function App() {
  const [mode, setMode] = useState('upload'); // 'upload' or 'ask'
  const [chunking, setChunking] = useState('recursive');
  const [topK, setTopK] = useState(5);
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState({ html: '<div class="placeholder">Waiting for input...</div>', type: '' });
  const [dragActive, setDragActive] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setResult({ html: "Please select a file to upload first.", type: "error" });
      return;
    }

    setIsLoading(true);
    setResult({ html: '<div class="placeholder">Processing request...</div>', type: '' });
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('strategy', chunking);
      
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");
      
      setResult({ 
        html: `✅ Success!<br><br>Message: ${data.message}<br>Task ID: ${data.task_id}<br>Doc ID: ${data.doc_id}<br><br>The backend is now processing and chunking this file in the background. You can switch to "Ask Question" to test it!`, 
        type: 'success' 
      });
    } catch (error) {
      setResult({ html: `Error: ${error.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      setResult({ html: "Please type a question.", type: "error" });
      return;
    }

    setIsLoading(true);
    setResult({ html: '<div class="placeholder">Processing request...</div>', type: '' });
    
    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, top_k: topK })
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Query failed");
      
      const formattedAnswer = data.answer.replace(/\n/g, '<br>');
      
      setResult({ 
        html: `<strong>Answer:</strong><br><div class="markdown-content">${formattedAnswer}</div><hr style="border: 0; border-top: 1px solid var(--card-border); margin: 1rem 0;"><small style="color: var(--text-secondary)">Retrieved Context Chunks: ${data.context_chunks_used}</small>`, 
        type: '' 
      });
    } catch (error) {
      setResult({ html: `Error: ${error.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (mode === 'upload') handleUpload();
    else handleAsk();
  };

  return (
    <div className="app-container">
      <main className="glass-card">
        <header>
          <h1>RAG Vector DB</h1>
          <p>Upload a document to ingest, or type a question to ask the LLM.</p>
        </header>

        <section className="input-section">
          <div className="mode-toggle">
            <button 
              type="button" 
              className={`mode-btn ${mode === 'upload' ? 'active' : ''}`} 
              onClick={() => { setMode('upload'); setResult({ html: '<div class="placeholder">Waiting for input...</div>', type: '' }); }}
            >
              Upload File
            </button>
            <button 
              type="button" 
              className={`mode-btn ${mode === 'ask' ? 'active' : ''}`} 
              onClick={() => { setMode('ask'); setResult({ html: '<div class="placeholder">Waiting for input...</div>', type: '' }); }}
            >
              Ask Question
            </button>
          </div>
          
          <div className="input-zones">
            {mode === 'upload' ? (
              <div 
                className={`dropzone ${dragActive ? 'dragover' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadCloud className="drop-icon" size={40} />
                <p className="drop-text">Drag & drop a file here, or click to browse</p>
                <p className="supported-formats">Supports PDF, DOCX, XLSX, TXT, MD</p>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  className="hidden-input" 
                  accept=".pdf,.docx,.xlsx,.txt,.csv,.md"
                  onChange={handleFileChange}
                />
                {file && <div className="file-name-display">Selected: {file.name}</div>}
              </div>
            ) : (
              <div className="textzone">
                <textarea 
                  placeholder="e.g., Summarize the uploaded document..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                ></textarea>
              </div>
            )}
          </div>
        </section>

        {mode === 'upload' && (
          <section className="options-section">
            <label>Chunking types</label>
            <div className="pill-group">
              {['recursive', 'fixed', 'sentence', 'paragraph'].map(type => (
                <button 
                  key={type}
                  type="button" 
                  className={`pill ${chunking === type ? 'active' : ''}`}
                  onClick={() => setChunking(type)}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </section>
        )}

        <section className="options-section">
          <label>Vector DB Algorithm (HNSW) / Top K</label>
          <div className="pill-group">
            {[1, 3, 5, 10].map(k => (
              <button 
                key={k}
                type="button" 
                className={`pill ${topK === k ? 'active' : ''}`}
                onClick={() => setTopK(k)}
              >
                Top {k}
              </button>
            ))}
          </div>
        </section>

        <section className="submit-section">
          <button 
            type="button" 
            className="primary-btn" 
            onClick={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? <div className="spinner"></div> : <span className="btn-text">Submit</span>}
          </button>
        </section>

        <section className="result-section">
          <label>Result</label>
          <div 
            className={`result-box ${result.type}`} 
            dangerouslySetInnerHTML={{ __html: result.html }}
          ></div>
        </section>
      </main>
    </div>
  );
}

export default App;
