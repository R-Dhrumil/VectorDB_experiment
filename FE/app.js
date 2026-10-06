const API_BASE = "http://localhost:8000/documents";

// DOM Elements
const modeBtns = document.querySelectorAll('.mode-btn');
const inputZones = document.querySelectorAll('.input-zones > div');
const chunkingPills = document.querySelectorAll('#chunking-pills .pill');
const algoPills = document.querySelectorAll('#algo-pills .pill');
const submitBtn = document.getElementById('submit-btn');
const resultBox = document.getElementById('result-box');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name-display');
const dropzone = document.getElementById('upload-zone');
const questionInput = document.getElementById('question-input');
const chunkingSection = document.getElementById('chunking-section');

// State
let currentMode = 'upload'; // 'upload' or 'ask'
let selectedChunking = 'recursive';
let selectedTopK = 5;
let selectedFile = null;

// --- Event Listeners ---

// Mode Toggle
modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        currentMode = btn.dataset.mode;
        
        // Toggle input zones
        inputZones.forEach(zone => zone.classList.remove('active'));
        document.getElementById(`${currentMode}-zone`).classList.add('active');
        
        // Hide chunking section if in ask mode (since chunking happens on upload)
        if (currentMode === 'ask') {
            chunkingSection.style.display = 'none';
        } else {
            chunkingSection.style.display = 'block';
        }
        
        // Reset Result Box
        resultBox.innerHTML = '<div class="placeholder">Waiting for input...</div>';
        resultBox.className = 'result-box';
    });
});

// Pill Selection (Chunking)
chunkingPills.forEach(pill => {
    pill.addEventListener('click', () => {
        chunkingPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        selectedChunking = pill.dataset.value;
    });
});

// Pill Selection (Algo/TopK)
algoPills.forEach(pill => {
    pill.addEventListener('click', () => {
        algoPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        selectedTopK = parseInt(pill.dataset.value);
    });
});

// File Handling
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
    }
});

// Drag and drop visual cues
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        fileInput.files = e.dataTransfer.files; // sync with input
        fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
    }
});


// Submit Action
submitBtn.addEventListener('click', async () => {
    if (currentMode === 'upload') {
        await handleUpload();
    } else {
        await handleAsk();
    }
});


// --- Logic Functions ---

async function handleUpload() {
    if (!selectedFile) {
        showResult("Please select a file to upload first.", "error");
        return;
    }

    setLoading(true);
    resultBox.className = 'result-box';
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('strategy', selectedChunking);
        
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.detail || "Upload failed");
        
        showResult(`✅ Success!\n\nMessage: ${data.message}\nTask ID: ${data.task_id}\nDoc ID: ${data.doc_id}\n\nThe backend is now processing and chunking this file in the background. You can switch to "Ask Question" to test it!`, 'success');
        
    } catch (error) {
        showResult(`Error: ${error.message}`, "error");
    } finally {
        setLoading(false);
    }
}

async function handleAsk() {
    const question = questionInput.value.trim();
    if (!question) {
        showResult("Please type a question.", "error");
        return;
    }

    setLoading(true);
    resultBox.className = 'result-box';
    
    try {
        const payload = {
            query: question,
            top_k: selectedTopK
        };
        
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.detail || "Query failed");
        
        // Simple Markdown rendering for the result
        let formattedAnswer = data.answer.replace(/\n/g, '<br>');
        
        showResult(`
            <strong>Answer:</strong><br>
            <div class="markdown-content">${formattedAnswer}</div>
            <hr style="border: 0; border-top: 1px solid var(--card-border); margin: 1rem 0;">
            <small style="color: var(--text-secondary)">Retrieved Context Chunks: ${data.context_chunks_used}</small>
        `);
        
    } catch (error) {
        showResult(`Error: ${error.message}`, "error");
    } finally {
        setLoading(false);
    }
}

function setLoading(isLoading) {
    if (isLoading) {
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;
        resultBox.innerHTML = '<div class="placeholder">Processing request...</div>';
    } else {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
}

function showResult(htmlContent, type = "") {
    resultBox.innerHTML = htmlContent;
    if (type) {
        resultBox.classList.add(type);
    }
}
