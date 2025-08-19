"""Retrieval engine with guardrails for the RAG pipeline."""

import re
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .models import Citation


class GuardRailEngine:
    """Implements guardrails for safe information retrieval."""
    
    # Sensitive topics that should only come from facts
    SENSITIVE_KEYWORDS = [
        'price', 'pricing', 'cost', 'warranty', 'guarantee', 'available', 'availability',
        'in stock', 'purchase', 'buy', 'deal', 'discount', 'offer', 'aed', '$', 'usd',
        'delivery', 'shipping', 'insurance', 'financing', 'loan'
    ]
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def is_sensitive_query(self, query: str) -> bool:
        """Check if query contains sensitive keywords."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.SENSITIVE_KEYWORDS)
    
    def should_use_external(self, query: str, facts_results: List[Dict]) -> bool:
        """Determine if external sources should be used."""
        # Never use external for sensitive queries
        if self.is_sensitive_query(query):
            return False
        
        # Use external only if facts don't have sufficient information
        if not facts_results or len(facts_results) == 0:
            return True
        
        # Check if facts results have low confidence
        high_confidence_facts = [
            result for result in facts_results 
            if result.get('distance', 1.0) < (1.0 - self.confidence_threshold)
        ]
        
        return len(high_confidence_facts) == 0
    
    def filter_external_content(self, query: str, external_results: List[Dict]) -> List[Dict]:
        """Filter external content to remove potentially sensitive information."""
        filtered_results = []
        
        for result in external_results:
            content = result.get('content', '').lower()
            
            # Skip content that mentions sensitive topics
            if any(keyword in content for keyword in self.SENSITIVE_KEYWORDS):
                continue
            
            # Keep general information content
            filtered_results.append(result)
        
        return filtered_results


class RetrievalEngine:
    """Main retrieval engine that coordinates search and applies guardrails."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.guardrail_engine = GuardRailEngine()
        self.max_context_length = 2000
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant information with guardrails applied."""
        # Always search facts first
        facts_results = self.vector_store.search_facts(query, n_results=5)
        
        # Determine if external sources should be used
        use_external = self.guardrail_engine.should_use_external(query, facts_results)
        
        external_results = []
        if use_external:
            raw_external = self.vector_store.search_external(query, n_results=3)
            external_results = self.guardrail_engine.filter_external_content(query, raw_external)
        
        # Prepare context and citations
        context_parts = []
        citations = []
        
        # Add facts context first (higher priority)
        for result in facts_results[:3]:  # Limit to top 3 facts
            context_parts.append(f"[FACTS] {result['content']}")
            citations.append(Citation(
                source="facts",
                doc_id=result['id'].split(':')[0],
                chunk_id=result['id']
            ))
        
        # Add external context if allowed and available
        for result in external_results[:2]:  # Limit to top 2 external
            context_parts.append(f"[EXTERNAL] {result['content']}")
            citations.append(Citation(
                source="external", 
                doc_id=result['id'].split(':')[0],
                chunk_id=result['id']
            ))
        
        context = '\n\n'.join(context_parts)
        
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return {
            'context': context,
            'citations': citations,
            'facts_count': len(facts_results),
            'external_count': len(external_results),
            'is_sensitive': self.guardrail_engine.is_sensitive_query(query),
            'used_external': len(external_results) > 0
        }
