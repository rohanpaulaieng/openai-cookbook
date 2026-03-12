# My RAG Pipeline Experiments

Notes from building a production RAG system for enterprise Q&A.

## Setup
```python
from openai import OpenAI
client = OpenAI()

# Embed docs
def embed(text):
    return client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding
```

## Key Learnings
- Chunk size 512 tokens works best for technical docs
- Overlap of 50 tokens reduces context loss at boundaries
- Hybrid search (BM25 + dense) outperforms pure dense by ~15%
- Re-ranking with Cohere improves precision@5 significantly

## Results
| Config | Precision@5 | Latency |
|--------|------------|---------|
| Dense only | 0.71 | 120ms |
| Hybrid | 0.83 | 145ms |
| Hybrid + rerank | 0.89 | 210ms |