# RAG Pipeline Design Document

## Overview

This document explains the architecture and design decisions for the BYD SEAL RAG pipeline, focusing on guardrails implementation and factual accuracy.

## Core Requirements

Based on the evaluation criteria, the system must:
1. **Prioritize Facts**: Always use factual dataset before external sources
2. **Prevent Hallucinations**: Never generate ungrounded information
3. **Implement Guardrails**: Refuse sensitive information from unreliable sources
4. **Provide Citations**: Every answer must include traceable sources
5. **Handle Edge Cases**: Gracefully refuse when information is unavailable

## Architecture Components

### 1. Data Ingestion Layer (`src/data_ingestion.py`)

**Purpose**: Process and chunk documents for vector storage

**Key Design Decisions**:
- **Markdown Processing**: Parse sections as logical units for better retrieval
- **Smart Chunking**: Balance between context preservation and search granularity
- **Metadata Preservation**: Maintain source tracking for citation generation
- **Chunk Size**: 500 characters with 50-character overlap for context continuity

**Processing Strategy**:
```
Facts (MD) → Section-based chunks → Vector embeddings
External (JSON) → Video-based chunks → Vector embeddings
```

### 2. Vector Store (`src/vector_store.py`)

**Purpose**: Manage document embeddings and similarity search

**Technology Choice**: ChromaDB
- **Rationale**: Lightweight, no external dependencies, good for prototypes
- **Collections**: Separate facts and external for targeted retrieval
- **Embedding Model**: `all-MiniLM-L6-v2` for balance of speed and quality

**Key Features**:
- Persistent storage for production use
- Cosine similarity for semantic search
- Separate collections enable guardrail enforcement

### 3. Guardrail Engine (`src/retrieval_engine.py`)

**Purpose**: Enforce business rules and safety constraints

**Design Philosophy**: Fail-safe approach
- **Default to Safe**: When in doubt, refuse rather than risk incorrect information
- **Facts First**: Always search facts before considering external sources
- **Contextual Filtering**: Content-aware filtering based on query sensitivity

**Guardrail Rules**:
```python
def should_use_external(query, facts_results):
    if is_sensitive_query(query):
        return False  # Never use external for sensitive topics
    
    if has_sufficient_facts(facts_results):
        return False  # Don't need external if facts are good
    
    return True  # Only use external as last resort
```

**Sensitive Topic Detection**:
- **Keyword-based**: Price, warranty, availability, purchase terms
- **Context-aware**: "available colors" vs "available for purchase"
- **Configurable**: Easy to add new sensitive topics

### 4. LLM Client (`src/llm_client.py`)

**Purpose**: Generate natural language responses with citations

**Model Choice**: GPT-3.5-turbo
- **Rationale**: Good balance of capability and cost
- **Temperature**: 0.1 for factual consistency
- **Max Tokens**: 500 for concise responses

**Prompt Engineering**:
```
System Prompt:
- Use only provided context
- Cite all information with [source:doc:chunk] format
- Refuse if insufficient information
- Prioritize [FACTS] over [EXTERNAL] sources
```

### 5. API Layer (`src/api.py`)

**Purpose**: Provide HTTP interface with validation

**Framework**: FastAPI
- **Automatic validation** with Pydantic models
- **OpenAPI documentation** for easy integration
- **Async support** for better performance

**Error Handling**:
- Input validation for malformed requests
- Graceful degradation on LLM API failures
- Health checks for system monitoring

## Response Generation Flow

```
1. Question Input
   ↓
2. Facts Search (Always First)
   ↓
3. Guardrail Check
   ↓ (if needed)
4. External Search (Filtered)
   ↓
5. Context Assembly
   ↓
6. LLM Generation
   ↓
7. Citation Addition
   ↓
8. Response Validation
   ↓
9. JSON Response
```

## Guardrail Implementation Details

### Three-Tier Safety Model

**Tier 1: Query Classification**
```python
SENSITIVE_KEYWORDS = [
    'price', 'pricing', 'cost', 'warranty', 'guarantee',
    'available', 'availability', 'purchase', 'buy'
]

def is_sensitive_query(query):
    return any(keyword in query.lower() for keyword in SENSITIVE_KEYWORDS)
```

**Tier 2: Source Selection**
```python
def determine_sources(query, facts_results):
    if is_sensitive_query(query):
        return "facts_only"
    elif sufficient_facts_available(facts_results):
        return "facts_primary"  
    else:
        return "facts_and_external"
```

**Tier 3: Content Filtering**
```python
def filter_external_content(results):
    return [r for r in results if not contains_sensitive_info(r['content'])]
```

### Citation System

**Format**: `[source:doc_id:chunk_id]`
- **source**: "facts" or "external"
- **doc_id**: Document identifier (e.g., "F001", "E024")
- **chunk_id**: Specific chunk within document

**Implementation**:
- Citations generated during retrieval
- Embedded in LLM response
- Validated in final output

## Edge Case Handling

### No Information Available
```json
{
    "answer": "I don't have sufficient information about this aspect of the BYD SEAL.",
    "status": "no_data",
    "citations": []
}
```

### Sensitive Query Without Facts
```json
{
    "answer": "I can only provide pricing information from official documentation, which doesn't contain this information.",
    "status": "refused", 
    "citations": []
}
```

### Facts Available for Sensitive Query
```json
{
    "answer": "The BYD SEAL Design is priced at AED 149,900 [facts:F020:c0]",
    "status": "answered",
    "citations": [{"source": "facts", "doc_id": "F020", "chunk_id": "c0"}]
}
```

## Performance Considerations

### Embedding Generation
- **Caching**: Embeddings stored persistently 
- **Batch Processing**: Efficient initialization
- **Model Loading**: One-time startup cost

### Retrieval Optimization
- **Separate Collections**: Facts vs external for targeted search
- **Result Limiting**: Top-K retrieval to control latency
- **Context Truncation**: Prevent oversized LLM inputs

### API Performance
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Startup Optimization**: Lazy loading where possible

## Security and Safety

### Input Validation
- **Query Length Limits**: Prevent abuse
- **Content Filtering**: Block malicious inputs
- **Rate Limiting**: Prevent system overload

### Output Safety
- **Citation Validation**: Ensure all claims are sourced
- **Content Review**: Filter inappropriate responses
- **Fallback Mechanisms**: Safe defaults on errors

## Testing Strategy

### Unit Tests
- **Guardrail Logic**: Verify sensitive query detection
- **Content Filtering**: Test external content filtering
- **Response Format**: Validate API response structure

### Integration Tests
- **End-to-End Pipeline**: Full question-answer flow
- **Error Handling**: System behavior on failures
- **Performance Tests**: Response time validation

### Evaluation Criteria Alignment

**Grounding & Correctness (35 pts)**:
- All responses cite sources
- No information generated without context
- Facts prioritized over external sources

**Guardrails & Policy (25 pts)**:
- Sensitive topics handled appropriately
- Clear refusal messages
- External sources never override facts

**Retrieval Quality (20 pts)**:
- Semantic search with embeddings
- Chunking preserves context
- Robust to query variations

**Code Quality (10 pts)**:
- Modular architecture
- Comprehensive error handling
- Configuration management

**Documentation (10 pts)**:
- Clear setup instructions
- Architecture explanation
- Design rationale

## Future Improvements

### Scalability
- **Vector Database**: Switch to production-grade DB (Pinecone, Weaviate)
- **Caching Layer**: Redis for frequent queries
- **Load Balancing**: Multiple API instances

### Accuracy
- **Better Embeddings**: Domain-specific models
- **Query Understanding**: Intent classification
- **Response Validation**: Automated fact checking

### User Experience
- **Confidence Scores**: Response reliability indicators
- **Alternative Sources**: Suggest related information
- **Conversational Memory**: Multi-turn conversations

## Conclusion

This RAG pipeline design prioritizes safety and factual accuracy through multi-layered guardrails while maintaining good user experience. The architecture is modular, testable, and aligned with the evaluation criteria for preventing hallucinations and ensuring proper source attribution.
