// API Base URL - Change this if running from different host
// Use relative path when served from same origin, or absolute URL for cross-origin
const API_BASE = window.location.port === '5500' 
    ? 'http://localhost:8000/api/admin'  // Live Server -> Docker
    : '/api/admin';                       // Same origin (Docker)

// State
let documentsPage = 0;
let documentsLimit = 20;
let totalDocuments = 0;

// ============== Utility Functions ==============

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'success' ? 'toast-success' : 'toast-error'}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API Error');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast(error.message, 'error');
        throw error;
    }
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    try {
        return new Date(dateStr).toLocaleString();
    } catch {
        return dateStr;
    }
}

// ============== Navigation ==============

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
    
    // Show selected section
    const section = document.getElementById(`section-${sectionName}`);
    if (section) {
        section.classList.remove('hidden');
    }
    
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('bg-gray-800');
        if (link.dataset.section === sectionName) {
            link.classList.add('bg-gray-800');
        }
    });
    
    // Load section data
    switch (sectionName) {
        case 'dashboard':
            refreshStats();
            break;
        case 'documents':
            loadDocuments();
            break;
        case 'faiss':
            loadFaissStats();
            break;
    }
}

// ============== Dashboard ==============

async function refreshStats() {
    try {
        const stats = await apiCall('/stats');
        
        // Update stat cards
        document.getElementById('stat-documents').textContent = stats.neo4j.documents;
        document.getElementById('stat-chunks').textContent = stats.neo4j.chunks;
        document.getElementById('stat-questions').textContent = stats.neo4j.questions;
        document.getElementById('stat-vectors').textContent = stats.faiss.total_vectors;
        
        // Update detailed stats
        document.getElementById('neo4j-stats').innerHTML = `
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Documents</span>
                <span class="font-semibold">${stats.neo4j.documents}</span>
            </div>
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Sections</span>
                <span class="font-semibold">${stats.neo4j.sections}</span>
            </div>
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Chunks</span>
                <span class="font-semibold">${stats.neo4j.chunks}</span>
            </div>
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Questions</span>
                <span class="font-semibold">${stats.neo4j.questions}</span>
            </div>
            <div class="flex justify-between py-2">
                <span class="text-gray-600">Concepts</span>
                <span class="font-semibold">${stats.neo4j.concepts}</span>
            </div>
        `;
        
        document.getElementById('faiss-stats').innerHTML = `
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Total Vectors</span>
                <span class="font-semibold">${stats.faiss.total_vectors}</span>
            </div>
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Unique Chunks</span>
                <span class="font-semibold">${stats.faiss.unique_chunks}</span>
            </div>
            <div class="flex justify-between py-2 border-b">
                <span class="text-gray-600">Duplicates</span>
                <span class="font-semibold">${stats.faiss.duplicates}</span>
            </div>
            <div class="flex justify-between py-2">
                <span class="text-gray-600">Duplicate Rate</span>
                <span class="font-semibold">${stats.faiss.duplicate_rate}</span>
            </div>
        `;
        
        showToast('Stats refreshed');
    } catch (error) {
        console.error('Error refreshing stats:', error);
    }
}

// ============== Documents ==============

async function loadDocuments() {
    const language = document.getElementById('language-filter').value;
    const skip = documentsPage * documentsLimit;
    
    try {
        let url = `/documents?skip=${skip}&limit=${documentsLimit}`;
        if (language) {
            url += `&language=${language}`;
        }
        
        const data = await apiCall(url);
        totalDocuments = data.total;
        
        const tbody = document.getElementById('documents-table');
        
        if (data.documents.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="px-6 py-4 text-center text-gray-500">No documents found</td></tr>';
        } else {
            tbody.innerHTML = data.documents.map(doc => `
                <tr class="table-row">
                    <td class="px-6 py-4">
                        <div class="font-medium text-gray-900">${escapeHtml(doc.title)}</div>
                        <div class="text-sm text-gray-500">${doc.id.substring(0, 8)}...</div>
                    </td>
                    <td class="px-6 py-4">
                        <span class="px-2 py-1 text-xs rounded-full ${doc.language === 'vi' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}">
                            ${doc.language}
                        </span>
                    </td>
                    <td class="px-6 py-4 text-gray-600">${doc.sections_count}</td>
                    <td class="px-6 py-4 text-gray-600">${doc.chunks_count}</td>
                    <td class="px-6 py-4 text-gray-600">${doc.questions_count}</td>
                    <td class="px-6 py-4 text-gray-500 text-sm">${formatDate(doc.created_at)}</td>
                    <td class="px-6 py-4">
                        <div class="flex gap-2">
                            <button onclick="viewDocument('${doc.id}')" class="text-blue-600 hover:text-blue-800">View</button>
                            <button onclick="deleteDocument('${doc.id}')" class="text-red-600 hover:text-red-800">Delete</button>
                        </div>
                    </td>
                </tr>
            `).join('');
        }
        
        // Update pagination info
        const start = skip + 1;
        const end = Math.min(skip + documentsLimit, totalDocuments);
        document.getElementById('documents-info').textContent = `Showing ${start}-${end} of ${totalDocuments} documents`;
        
        // Update pagination buttons
        document.getElementById('btn-prev-docs').disabled = documentsPage === 0;
        document.getElementById('btn-next-docs').disabled = end >= totalDocuments;
        
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function prevDocumentsPage() {
    if (documentsPage > 0) {
        documentsPage--;
        loadDocuments();
    }
}

function nextDocumentsPage() {
    if ((documentsPage + 1) * documentsLimit < totalDocuments) {
        documentsPage++;
        loadDocuments();
    }
}

async function viewDocument(docId) {
    try {
        const data = await apiCall(`/documents/${docId}`);
        
        document.getElementById('modal-doc-title').textContent = data.document.title || 'Document Details';
        document.getElementById('modal-doc-content').innerHTML = `
            <div class="space-y-6">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <p class="text-sm text-gray-500">Document ID</p>
                        <p class="font-mono text-sm">${data.document.id}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Language</p>
                        <p>${data.document.language}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Source</p>
                        <p>${data.document.source || '-'}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Created</p>
                        <p>${formatDate(data.document.created_at)}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">Questions Generated</p>
                        <p class="font-semibold">${data.questions_count}</p>
                    </div>
                    <div>
                        <p class="text-sm text-gray-500">LlamaParse Used</p>
                        <p>${data.document.llamaparse_used ? 'Yes' : 'No'}</p>
                    </div>
                </div>
                
                <div>
                    <h4 class="font-semibold mb-3">Sections (${data.sections.length})</h4>
                    <div class="max-h-64 overflow-y-auto border rounded">
                        ${data.sections.map(s => `
                            <div class="flex justify-between items-center px-4 py-2 border-b hover:bg-gray-50">
                                <div>
                                    <span class="text-gray-400 mr-2">L${s.level}</span>
                                    <span>${escapeHtml(s.header || 'Root')}</span>
                                </div>
                                <span class="text-sm text-gray-500">${s.chunks_count} chunks</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="flex gap-4">
                    <button onclick="viewDocumentChunks('${docId}')" class="btn btn-secondary">
                        View Chunks
                    </button>
                    <button onclick="viewDocumentQuestions('${docId}')" class="btn btn-secondary">
                        View Questions
                    </button>
                    <button onclick="deleteDocument('${docId}'); closeDocumentModal();" class="btn btn-danger">
                        Delete Document
                    </button>
                </div>
            </div>
        `;
        
        document.getElementById('document-modal').classList.remove('hidden');
    } catch (error) {
        console.error('Error viewing document:', error);
    }
}

function closeDocumentModal() {
    document.getElementById('document-modal').classList.add('hidden');
}

async function viewDocumentChunks(docId) {
    try {
        const data = await apiCall(`/documents/${docId}/chunks?limit=100`);
        
        document.getElementById('modal-doc-title').textContent = 'Document Chunks';
        document.getElementById('modal-doc-content').innerHTML = `
            <div class="space-y-4">
                <p class="text-gray-600">Total: ${data.total} chunks</p>
                <div class="max-h-96 overflow-y-auto space-y-4">
                    ${data.chunks.map(chunk => `
                        <div class="border rounded p-4">
                            <div class="flex justify-between items-start mb-2">
                                <span class="text-xs bg-gray-100 px-2 py-1 rounded">${chunk.section_header}</span>
                                <span class="text-xs text-gray-500">${chunk.token_count} tokens</span>
                            </div>
                            <p class="text-sm text-gray-700">${escapeHtml(chunk.text)}</p>
                            <div class="mt-2 flex justify-between items-center">
                                <span class="text-xs text-gray-400 font-mono">${chunk.id.substring(0, 20)}...</span>
                                <button onclick="deleteChunk('${chunk.id}')" class="text-red-600 text-xs hover:text-red-800">Delete</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error viewing chunks:', error);
    }
}

async function viewDocumentQuestions(docId) {
    try {
        const data = await apiCall(`/documents/${docId}/questions?limit=100`);
        
        document.getElementById('modal-doc-title').textContent = 'Generated Questions';
        document.getElementById('modal-doc-content').innerHTML = `
            <div class="space-y-4">
                <p class="text-gray-600">Total: ${data.total} questions</p>
                <div class="max-h-96 overflow-y-auto space-y-4">
                    ${data.questions.map((q, idx) => `
                        <div class="border rounded p-4">
                            <div class="flex justify-between items-start mb-2">
                                <span class="font-semibold">Q${idx + 1}</span>
                                <div class="flex gap-2">
                                    <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">${q.question_type}</span>
                                    <span class="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">${q.difficulty}</span>
                                </div>
                            </div>
                            <p class="text-gray-800 mb-2">${escapeHtml(q.question)}</p>
                            <div class="space-y-1">
                                ${q.choices.map((c, i) => `
                                    <div class="text-sm ${String.fromCharCode(65 + i) === q.answer ? 'text-green-600 font-medium' : 'text-gray-600'}">
                                        ${String.fromCharCode(65 + i)}. ${escapeHtml(c)}
                                        ${String.fromCharCode(65 + i) === q.answer ? ' ✓' : ''}
                                    </div>
                                `).join('')}
                            </div>
                            <div class="mt-2 flex justify-end">
                                <button onclick="deleteQuestion('${q.id}')" class="text-red-600 text-xs hover:text-red-800">Delete</button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error viewing questions:', error);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document and all its data?')) {
        return;
    }
    
    try {
        const result = await apiCall(`/documents/${docId}`, { method: 'DELETE' });
        showToast(result.message);
        loadDocuments();
        refreshStats();
    } catch (error) {
        console.error('Error deleting document:', error);
    }
}

async function deleteChunk(chunkId) {
    if (!confirm('Delete this chunk and its questions?')) {
        return;
    }
    
    try {
        const result = await apiCall(`/chunks/${chunkId}`, { method: 'DELETE' });
        showToast(result.message);
    } catch (error) {
        console.error('Error deleting chunk:', error);
    }
}

async function deleteQuestion(questionId) {
    if (!confirm('Delete this question?')) {
        return;
    }
    
    try {
        const result = await apiCall(`/questions/${questionId}`, { method: 'DELETE' });
        showToast(result.message);
    } catch (error) {
        console.error('Error deleting question:', error);
    }
}

// ============== FAISS ==============

async function loadFaissStats() {
    try {
        const stats = await apiCall('/faiss/stats');
        
        document.getElementById('faiss-detailed-stats').innerHTML = `
            <div class="bg-blue-50 p-4 rounded-lg">
                <p class="text-blue-600 text-sm">Total Vectors</p>
                <p class="text-2xl font-bold text-blue-800">${stats.total_vectors}</p>
            </div>
            <div class="bg-green-50 p-4 rounded-lg">
                <p class="text-green-600 text-sm">Unique Chunks</p>
                <p class="text-2xl font-bold text-green-800">${stats.unique_chunks}</p>
            </div>
            <div class="bg-purple-50 p-4 rounded-lg">
                <p class="text-purple-600 text-sm">Dimension</p>
                <p class="text-2xl font-bold text-purple-800">${stats.dimension}</p>
            </div>
            <div class="bg-orange-50 p-4 rounded-lg">
                <p class="text-orange-600 text-sm">Documents</p>
                <p class="text-2xl font-bold text-orange-800">${stats.documents_count}</p>
            </div>
        `;
    } catch (error) {
        console.error('Error loading FAISS stats:', error);
    }
}

async function searchChunks() {
    const query = document.getElementById('search-query').value.trim();
    const docId = document.getElementById('search-doc-id').value.trim();
    
    if (!query) {
        showToast('Please enter a search query', 'error');
        return;
    }
    
    try {
        let url = `/search/chunks?query=${encodeURIComponent(query)}&limit=10`;
        if (docId) {
            url += `&document_id=${encodeURIComponent(docId)}`;
        }
        
        const data = await apiCall(url);
        
        const resultsDiv = document.getElementById('search-results');
        
        if (data.results.length === 0) {
            resultsDiv.innerHTML = '<p class="text-gray-500">No results found</p>';
        } else {
            resultsDiv.innerHTML = data.results.map((r, idx) => `
                <div class="border rounded p-4">
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-semibold">#${idx + 1}</span>
                        <span class="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            Score: ${r.score.toFixed(4)}
                        </span>
                    </div>
                    <p class="text-gray-700 text-sm">${escapeHtml(r.text)}</p>
                    <div class="mt-2 text-xs text-gray-400">
                        <span>Chunk: ${r.chunk_id.substring(0, 20)}...</span>
                        ${r.document_id ? `<span class="ml-4">Doc: ${r.document_id.substring(0, 8)}...</span>` : ''}
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Error searching:', error);
    }
}

async function rebuildFaissIndex() {
    if (!confirm('Rebuild FAISS index from Neo4j data? This may take a while.')) {
        return;
    }
    
    showToast('Rebuilding FAISS index...', 'success');
    
    try {
        const result = await apiCall('/faiss/rebuild', { method: 'POST' });
        showToast(`Index rebuilt: ${result.indexed} vectors`);
        loadFaissStats();
        refreshStats();
    } catch (error) {
        console.error('Error rebuilding index:', error);
    }
}

async function clearFaissIndex() {
    if (!confirm('Clear all FAISS vectors? This cannot be undone.')) {
        return;
    }
    
    try {
        const result = await apiCall('/faiss/clear', { method: 'DELETE' });
        showToast(result.message);
        loadFaissStats();
        refreshStats();
    } catch (error) {
        console.error('Error clearing FAISS:', error);
    }
}

// ============== Neo4j Query ==============

function setQuery(query) {
    document.getElementById('cypher-query').value = query;
}

async function executeCypherQuery() {
    const query = document.getElementById('cypher-query').value.trim();
    const readOnly = document.getElementById('read-only').checked;
    
    if (!query) {
        showToast('Please enter a query', 'error');
        return;
    }
    
    try {
        const result = await apiCall('/neo4j/query', {
            method: 'POST',
            body: JSON.stringify({ query, read_only: readOnly })
        });
        
        const resultsDiv = document.getElementById('query-results');
        
        if (result.results.length === 0) {
            resultsDiv.innerHTML = '<p class="text-gray-500">No results</p>';
        } else {
            // Create table from results
            const keys = Object.keys(result.results[0]);
            resultsDiv.innerHTML = `
                <p class="text-sm text-gray-500 mb-2">${result.count} results</p>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                ${keys.map(k => `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">${k}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
                            ${result.results.map(row => `
                                <tr class="hover:bg-gray-50">
                                    ${keys.map(k => `<td class="px-4 py-2 text-sm text-gray-700">${formatValue(row[k])}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }
    } catch (error) {
        document.getElementById('query-results').innerHTML = `
            <div class="text-red-600">
                <p class="font-semibold">Error</p>
                <p class="text-sm">${error.message}</p>
            </div>
        `;
    }
}

function formatValue(value) {
    if (value === null || value === undefined) return '<span class="text-gray-400">null</span>';
    if (typeof value === 'object') return `<pre class="text-xs">${JSON.stringify(value, null, 2)}</pre>`;
    return escapeHtml(String(value));
}

// ============== Clear All ==============

function showClearAllModal() {
    document.getElementById('confirm-clear-input').value = '';
    document.getElementById('clear-all-modal').classList.remove('hidden');
}

function closeClearAllModal() {
    document.getElementById('clear-all-modal').classList.add('hidden');
}

async function confirmClearAll() {
    const confirm = document.getElementById('confirm-clear-input').value;
    
    if (confirm !== 'CONFIRM') {
        showToast('Please type CONFIRM to proceed', 'error');
        return;
    }
    
    try {
        const result = await apiCall('/clear-all?confirm=CONFIRM', { method: 'DELETE' });
        showToast(result.message);
        closeClearAllModal();
        refreshStats();
    } catch (error) {
        console.error('Error clearing all:', error);
    }
}

// ============== Initialize ==============

document.addEventListener('DOMContentLoaded', () => {
    showSection('dashboard');
});

// Close modals on escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeDocumentModal();
        closeClearAllModal();
    }
});

// Close modals on outside click
document.getElementById('document-modal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('document-modal')) {
        closeDocumentModal();
    }
});

document.getElementById('clear-all-modal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('clear-all-modal')) {
        closeClearAllModal();
    }
});
