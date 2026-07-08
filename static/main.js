// ==============================================================================
// STATE MANAGEMENT & GLOBALS
// ==============================================================================
let systemConfig = null;
let dbStatus = null;
let currentRetrievedSources = {}; // Maps messageId -> array of sources
let pollingInterval = null;

// ==============================================================================
// MAIN INITIALIZATION
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
    initApp();
    setupEventListeners();
});

async function initApp() {
    await fetchConfigStatus();
    await fetchDbStatus();
    
    // Check if ingestion was already running on load
    if (dbStatus && dbStatus.ingestion_running) {
        startPollingIngestion();
    }
}

// ==============================================================================
// API CALLS
// ==============================================================================

async function fetchConfigStatus() {
    try {
        const response = await fetch("/api/config-status");
        systemConfig = await response.json();
        updateConfigUI();
    } catch (error) {
        console.error("Error fetching config status:", error);
    }
}

async function fetchDbStatus() {
    try {
        const response = await fetch("/api/status");
        dbStatus = await response.json();
        updateDbUI();
    } catch (error) {
        console.error("Error fetching database status:", error);
    }
}

async function triggerIngest() {
    try {
        const response = await fetch("/api/ingest", { method: "POST" });
        const result = await response.json();
        
        if (result.status === "started" || result.status === "already_running") {
            showNotification(result.message, "info");
            startPollingIngestion();
        }
    } catch (error) {
        console.error("Error triggering ingest:", error);
        showNotification("Failed to trigger ingestion.", "error");
    }
}

async function deleteDocument(docName) {
    if (!confirm(`Are you sure you want to delete all vector chunks for "${docName}"? This cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch("/api/documents", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ document_name: docName })
        });
        const result = await response.json();
        
        if (result.status === "deleted") {
            showNotification(result.message, "success");
            fetchDbStatus();
        }
    } catch (error) {
        console.error("Error deleting document:", error);
        showNotification("Failed to delete document.", "error");
    }
}

// ==============================================================================
// INGESTION POLLING
// ==============================================================================
function startPollingIngestion() {
    const progressPanel = document.getElementById("progress-panel");
    progressPanel.style.display = "block";
    
    document.getElementById("btn-ingest").disabled = true;
    document.getElementById("btn-ingest").classList.add("loading");
    
    // Set status indicator dot to busy
    const dot = document.getElementById("status-dot");
    dot.className = "status-indicator busy";

    if (pollingInterval) clearInterval(pollingInterval);
    
    pollingInterval = setInterval(async () => {
        await fetchDbStatus();
        
        if (dbStatus) {
            const progress = dbStatus.ingestion_progress;
            const processed = progress.processed || 0;
            const total = progress.total || 0;
            const percent = total > 0 ? Math.round((processed / total) * 100) : 0;
            
            document.getElementById("progress-percentage").innerText = `${percent}%`;
            document.getElementById("progress-fill").style.width = `${percent}%`;
            document.getElementById("progress-label").innerText = dbStatus.ingestion_running 
                ? `Ingesting (${processed}/${total})...`
                : "Finishing...";
                
            // Render logs
            const logsContainer = document.getElementById("progress-logs");
            logsContainer.innerHTML = progress.logs.map(log => `<div>> ${escapeHTML(log)}</div>`).join("");
            logsContainer.scrollTop = logsContainer.scrollHeight;
            
            if (!dbStatus.ingestion_running) {
                clearInterval(pollingInterval);
                pollingInterval = null;
                document.getElementById("btn-ingest").disabled = false;
                document.getElementById("btn-ingest").classList.remove("loading");
                progressPanel.style.display = "none";
                showNotification("PDF ingestion complete!", "success");
                fetchDbStatus();
            }
        }
    }, 1500);
}

// ==============================================================================
// UI UPDATES
// ==============================================================================
function updateConfigUI() {
    if (!systemConfig) return;
    
    const dot = document.getElementById("status-dot");
    const openaiVal = document.getElementById("status-openai");
    const qdrantVal = document.getElementById("status-qdrant");
    
    if (systemConfig.azure_openai_configured) {
        dot.className = "status-indicator online";
        openaiVal.innerText = "Configured";
        openaiVal.className = "val success";
    } else {
        dot.className = "status-indicator offline";
        openaiVal.innerText = "Missing Config";
        openaiVal.className = "val error";
    }
    
    qdrantVal.innerText = systemConfig.qdrant_storage;
}

function updateDbUI() {
    if (!dbStatus) return;
    
    document.getElementById("status-docs").innerText = dbStatus.total_documents;
    document.getElementById("status-chunks").innerText = dbStatus.total_chunks;
    document.getElementById("doc-count-badge").innerText = dbStatus.total_documents;
    
    // Render Document List
    const docList = document.getElementById("doc-list");
    const documents = dbStatus.documents || [];
    
    if (documents.length === 0) {
        docList.innerHTML = `
            <div class="doc-list-empty">
                <i class="fa-solid fa-folder-closed"></i>
                <p>No documents indexed yet.</p>
                <span>Put PDFs in the <code>input/</code> folder and click "Scan & Ingest".</span>
            </div>`;
        return;
    }
    
    const searchVal = document.getElementById("doc-search").value.toLowerCase();
    const filteredDocs = documents.filter(doc => doc.toLowerCase().includes(searchVal));
    
    if (filteredDocs.length === 0) {
        docList.innerHTML = `<div class="doc-list-empty"><p>No matching files found.</p></div>`;
        return;
    }
    
    docList.innerHTML = filteredDocs.map(doc => `
        <div class="doc-item">
            <div class="doc-info" title="${escapeHTML(doc)}">
                <span class="doc-icon"><i class="fa-solid fa-file-pdf"></i></span>
                <span class="doc-name">${escapeHTML(doc)}</span>
            </div>
            <div class="doc-actions">
                <button class="btn-delete-doc" onclick="deleteDocument('${escapeHTML(doc)}')">
                    <i class="fa-solid fa-trash"></i>
                </button>
            </div>
        </div>
    `).join("");
}

// ==============================================================================
// CHAT FUNCTIONALITY
// ==============================================================================
async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    
    const chatInput = document.getElementById("chat-input");
    const query = chatInput.value.trim();
    if (!query) return;
    
    chatInput.value = "";
    
    // Hide Welcome Screen
    const welcomeScreen = document.getElementById("welcome-screen");
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }
    
    // 1. Add User Message to Chat Window
    const messageId = "msg-" + Date.now();
    appendMessage(query, "user", messageId);
    
    // 2. Add Assistant Message with Typing Indicator
    const assistantMsgId = "msg-ai-" + Date.now();
    appendMessage("", "assistant", assistantMsgId, true);
    
    // Scroll chat to bottom
    scrollToBottom();
    
    try {
        // Send request to API
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: query, top_k: 5 })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server error occurred");
        }
        
        const result = await response.json();
        
        // Save retrieved sources linked to this message ID
        currentRetrievedSources[assistantMsgId] = result.sources || [];
        
        // 3. Replace Typing Indicator with actual content
        renderAIResponse(assistantMsgId, result.answer, result.sources);
        
    } catch (error) {
        console.error("Chat error:", error);
        removeTypingIndicator(assistantMsgId);
        renderErrorResponse(assistantMsgId, error.message);
    }
}

function appendMessage(text, sender, messageId, isLoader = false) {
    const chatWindow = document.getElementById("chat-window");
    
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    messageDiv.id = messageId;
    
    const avatarDiv = document.createElement("div");
    avatarDiv.className = "avatar";
    avatarDiv.innerHTML = sender === "user" ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "msg-bubble";
    
    if (isLoader) {
        bubbleDiv.innerHTML = `
            <div class="typing-loader">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>`;
    } else {
        bubbleDiv.innerHTML = sender === "user" ? escapeHTML(text) : formatResponseText(text, messageId);
    }
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
    chatWindow.appendChild(messageDiv);
    scrollToBottom();
}

function removeTypingIndicator(messageId) {
    const bubble = document.querySelector(`#${messageId} .msg-bubble`);
    if (bubble) bubble.innerHTML = "";
}

function renderAIResponse(messageId, answer, sources) {
    const bubble = document.querySelector(`#${messageId} .msg-bubble`);
    if (!bubble) return;
    
    // Clear typing loader
    bubble.innerHTML = "";
    
    // Process response text to HTML, converting Markdown and Injecting Clickable Citations
    const formattedHTML = formatResponseText(answer, messageId);
    
    const textContainer = document.createElement("div");
    textContainer.className = "message-text";
    textContainer.innerHTML = formattedHTML;
    bubble.appendChild(textContainer);
    
    // Attach source footnotes if any exist
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement("div");
        sourcesDiv.className = "message-sources";
        sourcesDiv.innerHTML = `<span class="sources-label">Retrieved References:</span>`;
        
        const listDiv = document.createElement("div");
        listDiv.className = "sources-list";
        
        // Group sources by document and pages for neat layout
        const groupedSources = {};
        sources.forEach((src, idx) => {
            const name = src.metadata.document_name;
            const pages = src.metadata.pages || [];
            const key = `${name}-P${pages.join(",")}`;
            
            if (!groupedSources[key]) {
                groupedSources[key] = {
                    idx: idx + 1,
                    name: name,
                    pages: pages
                };
            }
        });
        
        Object.values(groupedSources).forEach(src => {
            const btn = document.createElement("button");
            btn.className = "source-item-btn";
            btn.innerHTML = `<i class="fa-solid fa-file-pdf"></i> [${src.idx}] ${escapeHTML(src.name)} (Pg ${src.pages.join(", ")})`;
            btn.onclick = () => openSourceInDrawer(messageId, src.idx - 1);
            listDiv.appendChild(btn);
        });
        
        sourcesDiv.appendChild(listDiv);
        bubble.appendChild(sourcesDiv);
        
        // Add click events to inline citation numbers [1], [2] inside text
        const inlineTags = bubble.querySelectorAll(".citation-tag");
        inlineTags.forEach(tag => {
            const idx = parseInt(tag.getAttribute("data-idx")) - 1;
            tag.onclick = (e) => {
                e.stopPropagation();
                openSourceInDrawer(messageId, idx);
            };
        });
    }
    
    scrollToBottom();
}

function renderErrorResponse(messageId, errorText) {
    const bubble = document.querySelector(`#${messageId} .msg-bubble`);
    if (!bubble) return;
    
    bubble.innerHTML = `
        <div style="color: var(--error); display: flex; align-items: center; gap: 8px;">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span><strong>Error:</strong> ${escapeHTML(errorText)}</span>
        </div>`;
    scrollToBottom();
}

// ==============================================================================
// INSPECTOR DRAWER
// ==============================================================================
function openSourceInDrawer(messageId, sourceIdx) {
    const sources = currentRetrievedSources[messageId];
    if (!sources || !sources[sourceIdx]) return;
    
    const drawer = document.getElementById("context-drawer");
    const drawerContent = document.getElementById("drawer-content");
    
    // Render all sources for this message
    drawerContent.innerHTML = sources.map((src, idx) => {
        const isSelected = idx === sourceIdx;
        const scorePct = Math.round(src.score * 100);
        
        return `
            <div class="citation-card ${isSelected ? 'selected' : ''}" id="cit-card-${idx}">
                <div class="citation-card-header">
                    <span class="citation-card-title">
                        <i class="fa-solid fa-file-invoice"></i> [${idx + 1}] ${escapeHTML(src.metadata.document_name)}
                    </span>
                    <span class="citation-card-page">Page ${src.metadata.pages.join(", ")}</span>
                </div>
                <div class="citation-card-text">${escapeHTML(src.text)}</div>
                <div class="citation-card-score">Semantic Match: ${scorePct}%</div>
            </div>
        `;
    }).join("");
    
    // Open drawer
    drawer.classList.add("open");
    
    // Scroll selected card into view
    setTimeout(() => {
        const selectedCard = document.getElementById(`cit-card-${sourceIdx}`);
        if (selectedCard) {
            selectedCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }, 100);
}

function closeDrawer() {
    document.getElementById("context-drawer").classList.remove("open");
}

// ==============================================================================
// EVENT LISTENERS
// ==============================================================================
function setupEventListeners() {
    // Form Chat Submit
    document.getElementById("chat-form").addEventListener("submit", handleChatSubmit);
    
    // Ingest Button click
    document.getElementById("btn-ingest").addEventListener("click", triggerIngest);
    
    // Close Drawer click
    document.getElementById("btn-close-drawer").addEventListener("click", closeDrawer);
    
    // Search Bar input filter docs
    document.getElementById("doc-search").addEventListener("input", updateDbUI);
    
    // Clear Chat Button click
    document.getElementById("btn-clear-chat").addEventListener("click", () => {
        if (confirm("Are you sure you want to clear your current conversation history?")) {
            const chatWindow = document.getElementById("chat-window");
            chatWindow.innerHTML = `
                <div class="welcome-screen" id="welcome-screen">
                    <div class="welcome-logo">
                        <i class="fa-solid fa-brain-circuit"></i>
                    </div>
                    <h2>How can I help you today?</h2>
                    <p>Ask queries about the uploaded PDF files. The assistant will search the vector database, find corresponding passages, and answer with exact page citations.</p>
                    <div class="suggested-queries">
                        <h3>Suggested Questions:</h3>
                        <div class="suggestions-grid">
                            <button class="btn-suggestion">What are the primary procedures described in the documents?</button>
                            <button class="btn-suggestion">Summarize the main requirements mentioned in the texts.</button>
                            <button class="btn-suggestion">List the key category definitions or security rules.</button>
                            <button class="btn-suggestion">What is the context of Destination Details in Reporting?</button>
                        </div>
                    </div>
                </div>`;
            setupSuggestionClicks();
            closeDrawer();
        }
    });
    
    setupSuggestionClicks();
}

function setupSuggestionClicks() {
    const suggestions = document.querySelectorAll(".btn-suggestion");
    suggestions.forEach(btn => {
        btn.onclick = () => {
            document.getElementById("chat-input").value = btn.innerText;
            handleChatSubmit();
        };
    });
}

// ==============================================================================
// UTILITIES & HELPERS
// ==============================================================================

function scrollToBottom() {
    const chatWindow = document.getElementById("chat-window");
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHTML(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Super lightweight Markdown-to-HTML parser that preserves citations
 */
function formatResponseText(text, messageId) {
    if (!text) return "";
    
    let html = escapeHTML(text);
    
    // Convert bold: **text** -> <strong>text</strong>
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    
    // Convert italic: *text* -> <em>text</em>
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    
    // Convert backticks code: `code` -> <code>code</code>
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");
    
    // Handle inline lists
    // Group lines starting with "- " or "* " or "1. " and format them as lists
    const lines = html.split("\n");
    let inList = false;
    let listType = ""; // 'ul' or 'ol'
    let processedLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        
        if (line.startsWith("- ") || line.startsWith("* ")) {
            if (!inList || listType !== "ul") {
                if (inList) processedLines.push(`</${listType}>`);
                processedLines.push("<ul>");
                inList = true;
                listType = "ul";
            }
            processedLines.push(`<li>${line.substring(2)}</li>`);
        } else if (/^\d+\.\s/.test(line)) {
            const listStartTag = `<ol>`;
            if (!inList || listType !== "ol") {
                if (inList) processedLines.push(`</${listType}>`);
                processedLines.push(listStartTag);
                inList = true;
                listType = "ol";
            }
            const content = line.replace(/^\d+\.\s/, "");
            processedLines.push(`<li>${content}</li>`);
        } else {
            if (inList) {
                processedLines.push(`</${listType}>`);
                inList = false;
            }
            processedLines.push(line);
        }
    }
    
    if (inList) {
        processedLines.push(`</${listType}>`);
    }
    
    html = processedLines.join("\n");
    
    // Convert double linebreaks to paragraphs, single to breaks
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    
    // Wrap entire text in a paragraph if it doesn't contain block tags
    if (!html.startsWith("<p>") && !html.startsWith("<ul>") && !html.startsWith("<ol>")) {
        html = `<p>${html}</p>`;
    }
    
    // Convert citations: [1] or [2] etc. -> clickable tags
    // Matches patterns like [1], [2], [10] and converts them
    html = html.replace(/\[(\d+)\]/g, '<span class="citation-tag" data-idx="$1">[$1]</span>');
    
    return html;
}

function showNotification(message, type = "info") {
    // Create notification element
    const notif = document.createElement("div");
    notif.style.position = "fixed";
    notif.style.bottom = "20px";
    notif.style.right = "20px";
    notif.style.padding = "12px 20px";
    notif.style.borderRadius = "8px";
    notif.style.color = "white";
    notif.style.fontSize = "14px";
    notif.style.fontWeight = "600";
    notif.style.zIndex = "999";
    notif.style.boxShadow = "0 4px 15px rgba(0,0,0,0.3)";
    notif.style.display = "flex";
    notif.style.alignItems = "center";
    notif.style.gap = "8px";
    notif.style.animation = "slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)";
    
    let icon = "";
    if (type === "success") {
        notif.style.backgroundColor = "var(--success)";
        icon = '<i class="fa-solid fa-circle-check"></i>';
    } else if (type === "error") {
        notif.style.backgroundColor = "var(--error)";
        icon = '<i class="fa-solid fa-circle-xmark"></i>';
    } else {
        notif.style.backgroundColor = "var(--accent-purple)";
        icon = '<i class="fa-solid fa-circle-info"></i>';
    }
    
    notif.innerHTML = `${icon} <span>${escapeHTML(message)}</span>`;
    document.body.appendChild(notif);
    
    // Fade out and remove
    setTimeout(() => {
        notif.style.transition = "opacity 0.5s ease";
        notif.style.opacity = "0";
        setTimeout(() => {
            notif.remove();
        }, 500);
    }, 4000);
}
