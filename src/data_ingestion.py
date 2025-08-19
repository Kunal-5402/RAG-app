"""Data ingestion module for processing facts and external data."""

import json
import hashlib
from typing import List, Dict, Any
import markdown
from pathlib import Path

from .models import Document


class DataProcessor:
    """Process and prepare documents for vector database ingestion."""
    
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def process_facts_file(self, file_path: str) -> List[Document]:
        """Process the facts markdown file into chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse markdown content
        lines = content.split('\n')
        documents = []
        current_section = ""
        current_content = []
        section_count = 0
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section if it has content
                if current_content:
                    section_text = '\n'.join(current_content).strip()
                    if section_text:
                        doc_id = f"F{section_count:03d}"
                        doc = Document(
                            id=doc_id,
                            content=f"{current_section}\n\n{section_text}",
                            source="facts",
                            metadata={
                                "section": current_section,
                                "file_path": file_path,
                                "section_id": section_count
                            }
                        )
                        documents.append(doc)
                        section_count += 1
                
                # Start new section
                current_section = line.strip()
                current_content = []
            else:
                if line.strip():
                    current_content.append(line)
        
        # Handle last section
        if current_content:
            section_text = '\n'.join(current_content).strip()
            if section_text:
                doc_id = f"F{section_count:03d}"
                doc = Document(
                    id=doc_id,
                    content=f"{current_section}\n\n{section_text}",
                    source="facts",
                    metadata={
                        "section": current_section,
                        "file_path": file_path,
                        "section_id": section_count
                    }
                )
                documents.append(doc)
        
        return self._chunk_documents(documents)
    
    def process_external_file(self, file_path: str) -> List[Document]:
        """Process the external JSON file into chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        for idx, item in enumerate(data):
            # Extract relevant information from YouTube video data
            title = item.get('title', '')
            description = item.get('description', '')
            transcript = item.get('transcriptText', {}).get('content', '')
            
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if description:
                content_parts.append(f"Description: {description}")
            if transcript:
                content_parts.append(f"Transcript: {transcript}")
            
            if content_parts:
                content = '\n\n'.join(content_parts)
                doc_id = f"E{idx:03d}"
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    source="external",
                    metadata={
                        "video_id": item.get('video_id', ''),
                        "channel": item.get('channel_title', ''),
                        "views": item.get('views', 0),
                        "published_at": item.get('publishedAt', ''),
                        "file_path": file_path
                    }
                )
                documents.append(doc)
        
        return self._chunk_documents(documents)
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks if needed."""
        chunked_docs = []
        
        for doc in documents:
            if len(doc.content) <= self.chunk_size:
                chunked_docs.append(doc)
            else:
                # Split into overlapping chunks
                words = doc.content.split()
                chunk_words = self.chunk_size // 5  # Rough estimate of words per chunk
                overlap_words = self.chunk_overlap // 5
                
                for i in range(0, len(words), chunk_words - overlap_words):
                    chunk_text = ' '.join(words[i:i + chunk_words])
                    chunk_id = f"{doc.id}:c{i // (chunk_words - overlap_words)}"
                    
                    chunk_doc = Document(
                        id=chunk_id,
                        content=chunk_text,
                        source=doc.source,
                        metadata={**doc.metadata, "chunk_start": i, "parent_id": doc.id}
                    )
                    chunked_docs.append(chunk_doc)
                    
                    if i + chunk_words >= len(words):
                        break
        
        return chunked_docs


def ingest_data(facts_path: str, external_path: str) -> List[Document]:
    """Ingest and process all data files."""
    processor = DataProcessor()
    
    facts_docs = processor.process_facts_file(facts_path)
    external_docs = processor.process_external_file(external_path)
    
    all_docs = facts_docs + external_docs
    print(f"Processed {len(facts_docs)} facts documents and {len(external_docs)} external documents")
    
    return all_docs
