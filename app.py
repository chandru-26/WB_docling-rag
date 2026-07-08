import os
import logging
from typing import Optional, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from ai_service import AIService
from vector_db import VectorDB
from ingest import IngestionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGBackend")

app = FastAPI(
    title="Antigravity PDF RAG System",
    description="FastAPI Backend for PDF Semantic Search & Chat System using Azure OpenAI and Qdrant."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Instances
ai_service = AIService()
vector_db = VectorDB()

# Ingestion state tracking
class IngestionState:
    def __init__(self):
        self.is_running = False
        self.logs = []
        self.processed_count = 0
        self.total_count = 0

state = IngestionState()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class DeleteRequest(BaseModel):
    document_name: str

def run_ingestion_background():
    """Background task to run the ingestion pipeline."""
    global state
    state.is_running = True
    state.logs = ["Starting ingestion pipeline..."]
    state.processed_count = 0
    state.total_count = 0
    
    try:
        pipeline = IngestionPipeline(ai_service=ai_service, vector_db=vector_db)
        
        # We manually inline parts of pipeline.run() to update state for the UI
        if not pipeline.ai_service.enabled:
            state.logs.append("Error: Azure OpenAI is not configured in .env.")
            state.is_running = False
            return

        from glob import glob
        pdf_files = glob(os.path.join(pipeline.input_dir, "*.pdf"))
        if not pdf_files:
            state.logs.append(f"No PDF files found in '{pipeline.input_dir}'.")
            state.is_running = False
            return

        state.logs.append(f"Found {len(pdf_files)} PDF(s) in input directory.")
        
        ingested_docs = pipeline.vector_db.get_ingested_documents()
        to_process = [p for p in pdf_files if os.path.basename(p) not in ingested_docs]
        
        state.total_count = len(to_process)
        state.logs.append(f"{len(pdf_files) - len(to_process)} PDFs are already indexed. {len(to_process)} need processing.")

        if not to_process:
            state.logs.append("All PDF documents are up-to-date.")
            state.is_running = False
            return

        for idx, pdf_path in enumerate(to_process):
            filename = os.path.basename(pdf_path)
            state.logs.append(f"[{idx+1}/{len(to_process)}] Processing: {filename}")
            
            success = pipeline.process_single_pdf(pdf_path)
            if success:
                state.processed_count += 1
                state.logs.append(f"Successfully processed {filename}.")
            else:
                state.logs.append(f"Failed to process {filename}.")

        state.logs.append("Ingestion completed successfully!")
    except Exception as e:
        logger.error(f"Error in background ingestion: {e}", exc_info=True)
        state.logs.append(f"Fatal error during ingestion: {str(e)}")
    finally:
        state.is_running = False

@app.get("/api/config-status")
def get_config_status():
    """Check if the backend has valid OpenAI credentials."""
    return {
        "azure_openai_configured": ai_service.enabled,
        "embedding_deployment": config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "chat_deployment": config.AZURE_OPENAI_CHAT_DEPLOYMENT,
        "qdrant_storage": "Remote Qdrant Server" if config.QDRANT_URL else "Local disk (qdrant_db)",
        "input_directory": os.path.abspath(config.INPUT_DIR),
    }

@app.get("/api/status")
def get_db_status():
    """Get database collection statistics."""
    try:
        stats = vector_db.get_db_stats()
        return {
            **stats,
            "ingestion_running": state.is_running,
            "ingestion_progress": {
                "processed": state.processed_count,
                "total": state.total_count,
                "logs": state.logs[-10:] if state.logs else []
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database status error: {str(e)}")

@app.post("/api/ingest")
def start_ingest(background_tasks: BackgroundTasks):
    """Triggers the PDF ingestion pipeline in the background."""
    if state.is_running:
        return {"status": "already_running", "message": "Ingestion pipeline is already running."}
    
    background_tasks.add_task(run_ingestion_background)
    return {"status": "started", "message": "Ingestion started in the background."}

@app.post("/api/chat", response_model=ChatResponse)
def chat_with_docs(request: ChatRequest):
    """Answers a user question based on matching text from the uploaded PDFs."""
    if not ai_service.enabled:
        raise HTTPException(status_code=500, detail="Azure OpenAI is not configured. Please edit the .env file.")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # 1. Generate query embedding
        query_vector = ai_service.get_embedding(request.message)
        
        # 2. Perform vector search in Qdrant
        top_k = request.top_k or 5
        hits = vector_db.search(query_vector, top_k=top_k)
        
        # 3. Generate answer using retrieved contexts
        answer = ai_service.generate_answer(request.message, hits)
        
        return ChatResponse(
            answer=answer,
            sources=hits
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.delete("/api/documents")
def delete_document(request: DeleteRequest):
    """Deletes a document from the Qdrant database."""
    try:
        vector_db.delete_document(request.document_name)
        return {"status": "deleted", "message": f"Document '{request.document_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

# Mount frontend files at root (if they exist)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {
            "message": "Welcome to PDF RAG API. Frontend static directory is not created yet.",
            "api_docs": "/docs"
        }

