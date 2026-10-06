import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';
import './index.css';

const API_BASE = "http://localhost:8000/documents";

function App() {
  const [mode, setMode] = useState('upload'); // 'upload', 'paste', or 'ask'
  const [chunking, setChunking] = useState('recursive');
  const [embeddingModel, setEmbeddingModel] = useState('nomic-embed-text');
  const [distanceMetric, setDistanceMetric] = useState('cosine');
  const [topK, setTopK] = useState(5);
  const [file, setFile] = useState(null);
  const [paragraphText, setParagraphText] = useState('');
  const [paragraphTitle, setParagraphTitle] = useState('');
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
      formData.append('embedding_model', embeddingModel);

      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");

      setResult({
        html: `✅ <strong>Success!</strong><br><br><strong>Message:</strong> ${data.message}<br><strong>Task ID:</strong> ${data.task_id}<br><strong>Doc ID:</strong> ${data.doc_id}<br><strong>Model:</strong> <code>${data.embedding_model || embeddingModel}</code><br><br>The backend is extracting, chunking, and saving vectors to <strong>PostgreSQL (pgvector)</strong> in the background. Switch to "Ask Question" to test similarity retrieval!`,
        type: 'success'
      });
    } catch (error) {
      setResult({ html: `Error: ${error.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasteIngest = async () => {
    if (!paragraphText.trim()) {
      setResult({ html: "Please enter or paste a paragraph first.", type: "error" });
      return;
    }

    setIsLoading(true);
    setResult({ html: '<div class="placeholder">Chunking and embedding paragraph in PostgreSQL...</div>', type: '' });

    try {
      const response = await fetch(`${API_BASE}/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: paragraphText,
          title: paragraphTitle.trim() || undefined,
          strategy: chunking,
          embedding_model: embeddingModel
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ingestion failed");

      setResult({
        html: `✅ <strong>Success!</strong><br><br><strong>Message:</strong> ${data.message}<br><strong>Title/Snippet:</strong> <code>${data.title}</code><br><strong>Doc ID:</strong> <code>${data.doc_id}</code><br><strong>Chunks Created:</strong> ${data.chunks_created}<br><strong>Strategy:</strong> <code>${data.strategy}</code><br><strong>Model:</strong> <code>${data.embedding_model}</code><br><br>Paragraph is saved! Switch to <strong>"Ask Question"</strong> to run similarity queries against it.`,
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
        body: JSON.stringify({
          query: question,
          top_k: topK,
          embedding_model: embeddingModel,
          distance_metric: distanceMetric
        })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Query failed");

      const formattedAnswer = data.answer.replace(/\n/g, '<br>');

      setResult({
        html: `<strong>Answer:</strong><br><div class="markdown-content">${formattedAnswer}</div><hr style="border: 0; border-top: 1px solid var(--card-border); margin: 1rem 0;"><small style="color: var(--text-secondary)">Retrieved Chunks: ${data.context_chunks_used} &bull; Metric: <code>${data.distance_metric || distanceMetric}</code> &bull; Embedding: <code>${data.embedding_model || embeddingModel}</code></small>`,
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
    else if (mode === 'paste') handlePasteIngest();
    else handleAsk();
  };


  return (
    <div className="app-container">
      <main className="glass-card">
        <header>
          <h1>RAG Vector DB</h1>
          <p>Upload a document to ingest, or type a question to ask the LLM.</p>
          <div className="system-badge">
            <span className="badge-item">🗄️ PostgreSQL (pgvector)</span>
            <span className="badge-dot">•</span>
            <span className="badge-item">🧠 {embeddingModel}</span>
            <span className="badge-dot">•</span>
            <span className="badge-item">📐 {distanceMetric}</span>
          </div>
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
              className={`mode-btn ${mode === 'paste' ? 'active' : ''}`}
              onClick={() => { setMode('paste'); setResult({ html: '<div class="placeholder">Waiting for input...</div>', type: '' }); }}
            >
              Paste Paragraph
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
            {mode === 'upload' && (
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
            )}

            {mode === 'paste' && (
              <div className="textzone">
                <input
                  type="text"
                  className="title-input"
                  placeholder="Optional title / label (e.g. Return Policy, Article Excerpt)"
                  value={paragraphTitle}
                  onChange={(e) => setParagraphTitle(e.target.value)}
                />
                <textarea
                  placeholder="Paste any article, paragraph, or raw document text here directly..."
                  value={paragraphText}
                  onChange={(e) => setParagraphText(e.target.value)}
                ></textarea>
              </div>
            )}

            {mode === 'ask' && (
              <div className="textzone">
                <textarea
                  placeholder="e.g., What are the main points? Ask any conceptual question..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                ></textarea>
              </div>
            )}
          </div>
        </section>

        {(mode === 'upload' || mode === 'paste') && (
          <section className="options-section">
            <label>Chunking Strategy</label>
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
          <label>Embedding Model (Ollama Local)</label>
          <div className="pill-group">
            {[
              { id: 'nomic-embed-text', label: 'nomic-embed-text (768-d)' },

            ].map(m => (
              <button
                key={m.id}
                type="button"
                className={`pill ${embeddingModel === m.id ? 'active' : ''}`}
                onClick={() => setEmbeddingModel(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </section>

        {mode === 'ask' && (
          <>
            <section className="options-section">
              <label>Distance Formula (pgvector Metric)</label>
              <div className="pill-group">
                {[
                  { id: 'cosine', label: 'Cosine (<=>)' },
                  { id: 'l2', label: 'Euclidean / L2 (<->)' },
                  { id: 'inner_product', label: 'Inner Product (<#>)' }
                ].map(m => (
                  <button
                    key={m.id}
                    type="button"
                    className={`pill ${distanceMetric === m.id ? 'active' : ''}`}
                    onClick={() => setDistanceMetric(m.id)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </section>

            <section className="options-section">
              <label>Top K Results to Retrieve</label>
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
          </>
        )}

        <section className="submit-section">
          <button
            type="button"
            className="primary-btn"
            onClick={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? (
              <div className="spinner"></div>
            ) : (
              <span className="btn-text">
                {mode === 'upload' ? 'Upload & Ingest File' : mode === 'paste' ? 'Ingest Paragraph' : 'Ask Question'}
              </span>
            )}
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
