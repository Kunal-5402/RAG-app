"""Vector store management using ChromaDB."""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .models import Document


class VectorStore:
    """Vector store for document embeddings and retrieval."""
    
    def __init__(self, db_path: str, embedding_model: str):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create collections for facts and external data
        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.external_collection = self.client.get_or_create_collection(
            name="external",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        facts_docs = [doc for doc in documents if doc.source == "facts"]
        external_docs = [doc for doc in documents if doc.source == "external"]
        
        if facts_docs:
            self._add_to_collection(facts_docs, self.facts_collection)
        
        if external_docs:
            self._add_to_collection(external_docs, self.external_collection)
    
    def _add_to_collection(self, documents: List[Document], collection) -> None:
        """Add documents to a specific collection."""
        if not documents:
            return
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to {collection.name} collection")
    
    def search_facts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search in facts collection."""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        results = self.facts_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return self._format_results(results, "facts")
    
    def search_external(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search in external collection."""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        results = self.external_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return self._format_results(results, "external")
    
    def _format_results(self, results: Dict, source: str) -> List[Dict[str, Any]]:
        """Format ChromaDB results."""
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted_results
        
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'source': source,
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def clear_collections(self) -> None:
        """Clear all collections (useful for testing)."""
        try:
            self.client.delete_collection("facts")
            self.client.delete_collection("external")
        except Exception:
            pass  # Collections might not exist
        
        # Recreate collections
        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.external_collection = self.client.get_or_create_collection(
            name="external",
            metadata={"hnsw:space": "cosine"}
        )
