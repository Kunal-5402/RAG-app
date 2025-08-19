"""LLM client for generating responses."""

from typing import Dict, Any
import openai
from .models import AnswerResponse, Citation


class LLMClient:
    """Client for interfacing with LLM to generate responses."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int = 500, temperature: float = 0.1):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_response(self, query: str, retrieval_result: Dict[str, Any]) -> AnswerResponse:
        """Generate a response based on query and retrieved context."""
        
        context = retrieval_result.get('context', '')
        citations = retrieval_result.get('citations', [])
        is_sensitive = retrieval_result.get('is_sensitive', False)
        facts_count = retrieval_result.get('facts_count', 0)
        
        # Handle cases with no relevant information
        if not context.strip():
            if is_sensitive:
                return AnswerResponse(
                    answer="I cannot provide pricing, warranty, or availability information as it's not available in our official documentation.",
                    status="refused",
                    citations=[]
                )
            else:
                return AnswerResponse(
                    answer="I don't have sufficient information to answer your question about the BYD SEAL.",
                    status="no_data",
                    citations=[]
                )
        
        # For sensitive queries, refuse if no facts available
        if is_sensitive and facts_count == 0:
            return AnswerResponse(
                answer="I can only provide pricing, warranty, and availability information from our official documentation, which doesn't contain this information.",
                status="refused",
                citations=[]
            )
        
        # Generate system prompt with guardrails
        system_prompt = self._create_system_prompt(is_sensitive)
        
        # Generate user prompt with context
        user_prompt = self._create_user_prompt(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add citation markers to answer if not present
            answer = self._add_citation_markers(answer, citations)
            
            return AnswerResponse(
                answer=answer,
                status="answered",
                citations=citations
            )
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return AnswerResponse(
                answer="I'm sorry, I'm unable to generate a response at the moment. Please try again later.",
                status="refused",
                citations=[]
            )
    
    def _create_system_prompt(self, is_sensitive: bool) -> str:
        """Create system prompt with appropriate guardrails."""
        base_prompt = """You are a helpful assistant providing information about the BYD SEAL electric vehicle.

CRITICAL RULES:
1. ONLY use information provided in the context
2. NEVER make up or hallucinate information
3. Always cite sources using the format [source:doc_id:chunk_id]
4. If information is marked as [FACTS], it comes from official documentation
5. If information is marked as [EXTERNAL], it comes from external sources like videos/reviews

"""
        
        if is_sensitive:
            base_prompt += """6. For pricing, warranty, availability, or purchase information:
   - ONLY use [FACTS] sources
   - REFUSE to answer if only [EXTERNAL] sources are available
   - Be explicit about the source of financial/commercial information

"""
        
        base_prompt += """7. Keep responses concise and factual
8. If you cannot provide a complete answer, say so clearly"""
        
        return base_prompt
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with query and context."""
        return f"""Context:
{context}

Question: {query}

Please answer based ONLY on the provided context. Include appropriate citations."""
    
    def _add_citation_markers(self, answer: str, citations: list[Citation]) -> str:
        """Add citation markers to the answer if not already present."""
        # If answer already has citation markers, return as is
        if '[' in answer and ']' in answer:
            return answer
        
        # Add citations to the end if none exist
        if citations:
            citation_text = ' '.join([
                f"[{cite.source}:{cite.doc_id}:{cite.chunk_id}]" 
                for cite in citations[:3]  # Limit to first 3 citations
            ])
            return f"{answer} {citation_text}"
        
        return answer
