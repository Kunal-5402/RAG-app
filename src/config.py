"""Configuration management for the RAG pipeline."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Vector Database
    vector_db_path: str = "./data/vector_db"
    
    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM Configuration
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.1
    
    # Data paths
    facts_file: str = "./data/byd_seal_facts.md"
    external_file: str = "./data/byd_seal_external.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields in .env file


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
