"""Data models for the RAG pipeline."""

from typing import List, Literal, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """Citation information for answer sources."""
    source: Literal["facts", "external"]
    doc_id: str
    chunk_id: str


class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str


class AnswerResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    status: Literal["answered", "refused", "no_data"]
    citations: List[Citation]


class Document(BaseModel):
    """Document model for internal processing."""
    id: str
    content: str
    source: Literal["facts", "external"]
    metadata: dict
