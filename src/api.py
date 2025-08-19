"""FastAPI application for the RAG pipeline."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import get_settings
from .models import QuestionRequest, AnswerResponse
from .vector_store import VectorStore
from .retrieval_engine import RetrievalEngine
from .llm_client import LLMClient
from .data_ingestion import ingest_data

# Initialize settings
settings = get_settings()

# Initialize components
vector_store = VectorStore(settings.vector_db_path, settings.embedding_model)
retrieval_engine = RetrievalEngine(vector_store)
llm_client = LLMClient(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    max_tokens=settings.max_tokens,
    temperature=settings.temperature
)

# Initialize FastAPI app
app = FastAPI(
    title="BYD SEAL RAG Pipeline",
    description="RAG pipeline for BYD SEAL information with guardrails",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the vector database on startup."""
    try:
        # Check if collections exist and have data
        facts_count = vector_store.facts_collection.count()
        external_count = vector_store.external_collection.count()
        
        if facts_count == 0 or external_count == 0:
            print("Initializing vector database...")
            documents = ingest_data(settings.facts_file, settings.external_file)
            vector_store.add_documents(documents)
            print("Vector database initialized successfully")
        else:
            print(f"Vector database already initialized: {facts_count} facts, {external_count} external documents")
    
    except Exception as e:
        print(f"Error initializing vector database: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "BYD SEAL RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask questions about BYD SEAL",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        facts_count = vector_store.facts_collection.count()
        external_count = vector_store.external_collection.count()
        
        return {
            "status": "healthy",
            "vector_db": "connected",
            "facts_documents": facts_count,
            "external_documents": external_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the BYD SEAL.
    
    This endpoint implements strict guardrails:
    - Always prioritizes official facts over external sources
    - Refuses to answer pricing/warranty questions using external sources
    - Includes source citations for all answers
    """
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Retrieve relevant information with guardrails
        retrieval_result = retrieval_engine.retrieve(request.question)
        
        # Generate response using LLM
        response = llm_client.generate_response(request.question, retrieval_result)
        
        return response
    
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )


if __name__ == "__main__":
    run_server()
