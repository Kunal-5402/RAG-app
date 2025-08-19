# BYD SEAL RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about the BYD SEAL electric vehicle with strict guardrails to ensure factual accuracy and prevent hallucinations.

## Features

- **Facts-First Architecture**: Always prioritizes official documentation over external sources
- **Strict Guardrails**: Refuses to answer pricing/warranty questions using unreliable external sources
- **Source Citations**: All answers include traceable citations to source documents
- **No Hallucinations**: Built-in safeguards prevent generating ungrounded information
- **RESTful API**: Simple HTTP API for integration

## Architecture

The system implements a multi-layered approach:

1. **Data Ingestion**: Processes official facts (Markdown) and external data (JSON) into vector embeddings
2. **Retrieval Engine**: Searches facts first, only uses external sources for non-sensitive queries
3. **Guardrail System**: Filters content and enforces business rules
4. **LLM Generation**: Generates responses with proper citations
5. **API Layer**: FastAPI endpoint with validation and error handling

## Quick Start

### Prerequisites

- Conda package manager
- OpenAI API key

### Installation

1. Clone and navigate to the project:
```bash
cd /path/to/rag
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate rag
```

3. Set up environment variables:
```bash
touch .env
# Edit .env file with your OpenAI API key
```

4. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Usage

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the acceleration of BYD SEAL?"}'
```

**Response format:**
```json
{
  "answer": "The BYD SEAL AWD can accelerate 0-100 km/h in 3.8 seconds [facts:F002:c0]",
  "status": "answered",
  "citations": [
    {
      "source": "facts",
      "doc_id": "F002",
      "chunk_id": "c0"
    }
  ]
}
```

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `VECTOR_DB_PATH`: Path to store vector database (default: ./data/vector_db)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `LLM_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)
- `MAX_TOKENS`: Maximum response length (default: 500)
- `TEMPERATURE`: LLM temperature (default: 0.1)

## Data Sources

The system processes two types of data:

1. **Facts** (`data/byd_seal_facts.md`): Official BYD SEAL specifications, features, and pricing
2. **External** (`data/byd_seal_external.json`): YouTube videos, reviews, and community content

## Guardrails

The system implements strict guardrails:

### Sensitive Query Detection
Automatically identifies queries about:
- Pricing and costs
- Warranty information
- Availability and purchasing
- Financial terms and deals

### Response Rules
- **Facts First**: Always search official documentation first
- **No External for Sensitive**: Never use external sources for sensitive topics
- **Clear Refusals**: Explicitly refuse when information is not available in facts
- **Source Attribution**: All responses include citations

### Example Behaviors

**Pricing Query (Refused from External):**
```
Q: "What's the price of BYD SEAL?"
A: "I can only provide pricing information from our official documentation, which shows: BYD SEAL Design - AED 149,900, Premium - AED 154,900, Performance - AED 179,900 [facts:F020:c0]"
```

**General Query (Uses External if Needed):**
```
Q: "How's the ride quality?"
A: "Based on user reviews, the BYD SEAL offers smooth and comfortable ride quality [external:E005:c1]"
```

## Testing

Run the test suite:
```bash
# Run unit tests
python -m pytest tests/test_guardrails_simple.py -v

# Run integration tests
python test_pipeline.py
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check and database status
- `POST /ask` - Ask questions (see API usage above)

## Performance Considerations

- Initial startup includes vector database initialization (~30 seconds)
- Embedding generation is cached for efficiency
- Response times typically under 3 seconds
- Vector database persists between restarts

## Troubleshooting

**Common Issues:**

1. **"OpenAI API key not found"**
   - Ensure `.env` file contains valid `OPENAI_API_KEY`

2. **Vector database initialization fails**
   - Check that `data/` directory contains the required files
   - Verify file permissions

3. **Dependency conflicts**
   - Use the provided conda environment
   - Check NumPy version compatibility

**Logs and Debugging:**
- Application logs startup progress and errors
- Use `/health` endpoint to verify system status
- Check vector database document counts
