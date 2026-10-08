import React, { useState, useRef } from 'react';
import { UploadCloud } from 'lucide-react';
import './index.css';

const API_BASE = "http://localhost:8000/documents";

function App() {
  // --- Step 1: Ingestion States ---
  const [inputTab, setInputTab] = useState('upload'); // 'upload' or 'paste'
  const [file, setFile] = useState(null);
  const [paragraphText, setParagraphText] = useState('');
  const [paragraphTitle, setParagraphTitle] = useState('');
  const [chunking, setChunking] = useState('recursive');
  const [embeddingModel, setEmbeddingModel] = useState('nomic-embed-text');
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestFeedback, setIngestFeedback] = useState(null);
  const [isIngested, setIsIngested] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // --- Step 2: Query / Ask States ---
  const [question, setQuestion] = useState('');
  const [distanceMetric, setDistanceMetric] = useState('cosine');
  const [topK, setTopK] = useState(5);
  const [isAsking, setIsAsking] = useState(false);
  const [askResult, setAskResult] = useState(null);

  const fileInputRef = useRef(null);

  // --- Drag & Drop handlers ---
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
      setIngestFeedback(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setIngestFeedback(null);
    }
  };

  // --- Ingestion Handler (Upload or Paste) ---
  const handleIngest = async () => {
    if (inputTab === 'upload' && !file) {
      setIngestFeedback({ type: 'error', message: 'Please select a file to upload first.' });
      return;
    }
    if (inputTab === 'paste' && !paragraphText.trim()) {
      setIngestFeedback({ type: 'error', message: 'Please enter or paste a paragraph first.' });
      return;
    }

    setIsIngesting(true);
    setIngestFeedback(null);

    try {
      if (inputTab === 'upload') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('strategy', chunking);
        formData.append('embedding_model', embeddingModel);

        setIngestFeedback({
          type: 'info',
          message: `⏳ Uploading "${file.name}" to server...`
        });

        const response = await fetch(`${API_BASE}/upload`, {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Upload failed");

        const taskId = data.task_id;
        setIngestFeedback({
          type: 'info',
          message: `⏳ File accepted. Extracting text & preparing chunks...`
        });

        // Poll task status every 1 second
        let isDone = false;
        let attempts = 0;
        const maxAttempts = 180; // 3-minute safety limit

        while (!isDone && attempts < maxAttempts) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          attempts++;

          try {
            const pollRes = await fetch(`${API_BASE}/tasks/${taskId}`);
            if (!pollRes.ok) continue;
            const pollData = await pollRes.json();

            if (pollData.status === 'PROCESSING') {
              setIngestFeedback({
                type: 'info',
                message: `⏳ ${pollData.details?.message || "Processing document & embeddings in background..."}`
              });
            } else if (pollData.status === 'COMPLETED') {
              isDone = true;
              setIngestFeedback({
                type: 'success',
                message: `✅ File "${pollData.details?.filename || file.name}" processed successfully!`,
                details: `Created ${pollData.details?.chunks_created} chunks • Strategy: ${pollData.details?.strategy || chunking} • Stored in PostgreSQL (pgvector)`
              });
              setIsIngested(true);
            } else if (pollData.status === 'FAILED') {
              isDone = true;
              throw new Error(pollData.details?.error || "Document processing failed in background.");
            }
          } catch (pollErr) {
            if (isDone) throw pollErr;
            // Transient fetch error during poll, continue polling
            console.warn("Poll attempt error:", pollErr);
          }
        }

        if (!isDone) {
          throw new Error("Document processing timed out after 3 minutes. Please check backend terminal.");
        }
      } else {
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

        setIngestFeedback({
          type: 'success',
          message: `✅ Paragraph successfully chunked and stored in PostgreSQL!`,
          details: `Created ${data.chunks_created} chunks • Strategy: ${data.strategy} • Doc ID: ${data.doc_id}`
        });
        setIsIngested(true);
      }
    } catch (error) {
      setIngestFeedback({ type: 'error', message: `Error: ${error.message}` });
    } finally {
      setIsIngesting(false);
    }
  };

  // --- Ask Question Handler ---
  const handleAsk = async () => {
    if (!question.trim()) return;

    setIsAsking(true);
    setAskResult({ html: '<div class="placeholder">Searching pgvector & generating answer with Gemini...</div>', type: '' });

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

      setAskResult({
        html: `<strong>Answer:</strong><br><div class="markdown-content">${formattedAnswer}</div><hr style="border: 0; border-top: 1px solid var(--card-border); margin: 0.75rem 0;"><small style="color: var(--text-secondary)">⚡ LLM Engine: <code>${data.llm_provider || 'Groq Cloud'}</code> &bull; Chunks: ${data.context_chunks_used} &bull; Metric: <code>${data.distance_metric || distanceMetric}</code></small>`,
        type: ''
      });
    } catch (error) {
      setAskResult({ html: `Error: ${error.message}`, type: 'error' });
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="app-container">
      <main className="glass-card">
        <header>
          <h1>RAG Vector DB Experiment</h1>
          <p>Process multi-format documents, store vector embeddings in PostgreSQL, and evaluate retrieval formulas.</p>
          <div className="system-badge">
            <span className="badge-item">🗄️ PostgreSQL (pgvector)</span>
            <span className="badge-dot">•</span>
            <span className="badge-item">🧠 {embeddingModel}</span>
            <span className="badge-dot">•</span>
            <span className="badge-item">📐 {distanceMetric}</span>
          </div>
        </header>

        <div className="steps-grid">
          {/* ================= STEP 1: INGESTION & CHUNKING ================= */}
          <section className="step-card">
            <div className="step-header">
              <div className="step-title">
                <span className="step-num">1</span>
                <span>Document Ingestion & Chunking</span>
              </div>
              <span className={`step-status ${isIngested ? 'active' : ''}`}>
                {isIngested ? '✓ Ready' : 'Pending Ingestion'}
              </span>
            </div>

          <div className="mode-toggle">
            <button
              type="button"
              className={`mode-btn ${inputTab === 'upload' ? 'active' : ''}`}
              onClick={() => { setInputTab('upload'); setIngestFeedback(null); }}
            >
              Upload File
            </button>
            <button
              type="button"
              className={`mode-btn ${inputTab === 'paste' ? 'active' : ''}`}
              onClick={() => { setInputTab('paste'); setIngestFeedback(null); }}
            >
              Paste Paragraph
            </button>
          </div>

          <div className="input-zones">
            {inputTab === 'upload' ? (
              <>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept=".pdf,.docx,.xlsx,.txt,.csv,.md"
                  onChange={handleFileChange}
                />
                <div
                  className={`dropzone ${dragActive ? 'dragover' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={(e) => {
                    handleDrop(e);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  onClick={() => {
                    if (fileInputRef.current) {
                      fileInputRef.current.value = '';
                      fileInputRef.current.click();
                    }
                  }}
                >
                  {file ? (
                    <div className="selected-file-card" onClick={(e) => e.stopPropagation()}>
                      <div className="file-info">
                        <span className="file-icon">📄</span>
                        <div className="file-meta">
                          <span className="file-name">{file.name}</span>
                          <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="file-remove-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          setFile(null);
                          if (fileInputRef.current) fileInputRef.current.value = '';
                        }}
                      >
                        ✕ Change
                      </button>
                    </div>
                  ) : (
                    <>
                      <UploadCloud className="drop-icon" size={36} />
                      <p className="drop-text">Drag & drop your file here, or click to browse</p>
                      <p className="supported-formats">Supports PDF, DOCX, XLSX, TXT, MD</p>
                    </>
                  )}
                </div>
              </>
            ) : (
              <div className="textzone">
                <input
                  type="text"
                  className="title-input"
                  placeholder="Snippet Title / Label (e.g. Return Policy, Project Notes)"
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
          </div>

          {/* Chunking Strategy */}
          <div className="options-section" style={{ marginTop: '1rem' }}>
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
          </div>

          {/* Embedding Model */}
          <div className="options-section">
            <label>Embedding Model (Ollama Local)</label>
            <div className="pill-group">
              {[
                { id: 'nomic-embed-text', label: 'nomic-embed-text (768-d)' }
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
          </div>

          {/* Ingest Action Button */}
          <button
            type="button"
            className="primary-btn"
            onClick={handleIngest}
            disabled={isIngesting}
          >
            {isIngesting ? (
              <div className="spinner"></div>
            ) : (
              <span>⚡ Process & Store Embeddings in pgvector</span>
            )}
          </button>

          {/* Ingestion Feedback Banner */}
          {ingestFeedback && (
            <div className={`ingest-feedback ${ingestFeedback.type}`}>
              <div><strong>{ingestFeedback.message}</strong></div>
              {ingestFeedback.details && <small style={{ opacity: 0.85 }}>{ingestFeedback.details}</small>}
            </div>
          )}
        </section>

        {/* ================= STEP 2: ASK QUESTION & SEARCH ================= */}
        <section className="step-card">
          <div className="step-header">
            <div className="step-title">
              <span className="step-num">2</span>
              <span>Ask Question & Semantic Retrieval</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span className={`step-status ${isIngested ? 'active' : 'locked'}`}>
                {isIngested ? '🟢 Active & Ready' : '🔒 Locked (Ingest Step 1 First)'}
              </span>
              {!isIngested && (
                <button
                  type="button"
                  className="unlock-link"
                  onClick={() => setIsIngested(true)}
                  title="Click to unlock if you already have documents stored in database"
                >
                  (Or unlock now)
                </button>
              )}
            </div>
          </div>

          {/* Evaluation Settings (Formula & Top K) */}
          <div className="options-section">
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
          </div>

          <div className="options-section">
            <label>Top K Chunks to Retrieve</label>
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
          </div>

          {/* Unified Question Field with Integrated Ask Button */}
          <div className={`ask-box-wrapper ${!isIngested ? 'disabled' : ''}`}>
            <textarea
              className="ask-textarea"
              placeholder={isIngested ? "Ask any question or search query based on your stored documents..." : "Please ingest a document or paragraph above to unlock search..."}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (isIngested && !isAsking && question.trim()) {
                    handleAsk();
                  }
                }
              }}
              disabled={!isIngested}
            />
            <div className="ask-box-footer">
              <span className="ask-hint">
                {isIngested ? "Press Enter to search • Shift+Enter for new line" : "Ingest data in Step 1 first"}
              </span>
              <button
                type="button"
                className="ask-action-btn"
                onClick={handleAsk}
                disabled={!isIngested || isAsking || !question.trim()}
              >
                {isAsking ? (
                  <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                ) : (
                  <span>Ask Question →</span>
                )}
              </button>
            </div>
          </div>

          {/* Query Results Box */}
          {askResult && (
            <div className="result-section" style={{ marginTop: '1.25rem' }}>
              <label>Answer & Retrieved Context</label>
              <div
                className={`result-box ${askResult.type}`}
                dangerouslySetInnerHTML={{ __html: askResult.html }}
              ></div>
            </div>
          )}
        </section>
      </div>
    </main>
  </div>
  );
}

export default App;
