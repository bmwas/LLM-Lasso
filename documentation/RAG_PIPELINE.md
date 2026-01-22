# Retrieval-Augmented Generation (RAG) Pipeline

This document explains how the RAG pipeline works in LLM-Lasso to enhance feature penalty generation with domain-specific knowledge from scientific literature.

## Overview

RAG (Retrieval-Augmented Generation) combines the power of large language models with external knowledge retrieval. Instead of relying solely on the LLM's training data, RAG retrieves relevant documents at query time and includes them as context, enabling more accurate and domain-specific responses.

## RAG Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LLM-LASSO RAG PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

                              INDEXING PHASE (One-time Setup)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐           │
│   │              │     │                  │     │                  │           │
│   │  PDF Files   │────▶│  Text Extraction │────▶│    Chunking      │           │
│   │  (Papers)    │     │  (pymupdf4llm)   │     │  (1000 chars)    │           │
│   │              │     │                  │     │                  │           │
│   └──────────────┘     └──────────────────┘     └────────┬─────────┘           │
│                                                          │                      │
│                                                          ▼                      │
│                                              ┌──────────────────┐               │
│                                              │                  │               │
│                                              │    Embedding     │               │
│                                              │    (OpenAI)      │               │
│                                              │                  │               │
│                                              └────────┬─────────┘               │
│                                                       │                         │
│                                                       ▼                         │
│                                              ┌──────────────────┐               │
│                                              │                  │               │
│                                              │  Vector Store    │               │
│                                              │  (ChromaDB)      │               │
│                                              │                  │               │
│                                              └──────────────────┘               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


                              RETRIEVAL PHASE (Per Feature Query)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   ┌──────────────┐                                                              │
│   │              │                                                              │
│   │ Feature Name │──────────────────────┐                                       │
│   │ (e.g. "Age") │                      │                                       │
│   │              │                      │                                       │
│   └──────────────┘                      │                                       │
│                                         ▼                                       │
│                              ┌──────────────────┐                               │
│                              │                  │                               │
│                              │  Query Embedding │                               │
│                              │     (OpenAI)     │                               │
│                              │                  │                               │
│                              └────────┬─────────┘                               │
│                                       │                                         │
│                                       ▼                                         │
│                              ┌──────────────────┐                               │
│                              │                  │                               │
│                              │ Similarity Search│◀────┐                         │
│                              │   (ChromaDB)     │     │                         │
│                              │                  │     │                         │
│                              └────────┬─────────┘     │                         │
│                                       │               │                         │
│                                       │        ┌──────┴──────┐                  │
│                                       │        │ Vector Store│                  │
│                                       │        │  (indexed   │                  │
│                                       │        │   chunks)   │                  │
│                                       │        └─────────────┘                  │
│                                       ▼                                         │
│                              ┌──────────────────┐                               │
│                              │  Top-K Relevant  │                               │
│                              │  Document Chunks │                               │
│                              │  (default: 3)    │                               │
│                              └────────┬─────────┘                               │
│                                       │                                         │
└───────────────────────────────────────┼─────────────────────────────────────────┘
                                        │
                                        ▼
                              GENERATION PHASE
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────┐              │
│   │                      PROMPT CONSTRUCTION                      │              │
│   │  ┌─────────────────────────────────────────────────────────┐ │              │
│   │  │ System: You are an expert in [domain]...                │ │              │
│   │  ├─────────────────────────────────────────────────────────┤ │              │
│   │  │ Context from retrieved documents:                       │ │              │
│   │  │ [Chunk 1: "Study found age correlates with..."]         │ │              │
│   │  │ [Chunk 2: "Age is a significant factor in..."]          │ │              │
│   │  │ [Chunk 3: "Research indicates older patients..."]       │ │              │
│   │  ├─────────────────────────────────────────────────────────┤ │              │
│   │  │ Question: Rate the relevance of "Age" for predicting    │ │              │
│   │  │ [target condition] on a scale of 1-5...                 │ │              │
│   │  └─────────────────────────────────────────────────────────┘ │              │
│   └──────────────────────────────────────────────────────────────┘              │
│                                       │                                         │
│                                       ▼                                         │
│                              ┌──────────────────┐                               │
│                              │                  │                               │
│                              │   LLM (GPT-4o)   │                               │
│                              │                  │                               │
│                              └────────┬─────────┘                               │
│                                       │                                         │
│                                       ▼                                         │
│                              ┌──────────────────┐                               │
│                              │  Penalty Score   │                               │
│                              │  (1-5 rating)    │                               │
│                              │                  │                               │
│                              │  Lower = More    │                               │
│                              │  Relevant        │                               │
│                              └──────────────────┘                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Process Flow

### 1. Indexing Phase (One-time Setup)

The indexing phase prepares your PDF documents for efficient retrieval:

```
PDF Documents → Text Extraction → Chunking → Embedding → Vector Storage
```

**Step-by-step:**

1. **Text Extraction**: PDFs are processed using `pymupdf4llm`, which extracts text while preserving:
   - Document structure (headers, paragraphs)
   - Tables and figures
   - Page numbers and metadata

2. **Chunking**: Extracted text is split into smaller chunks:
   - Default chunk size: 1000 characters
   - Overlap: 200 characters (ensures context continuity)
   - Each chunk retains metadata (filename, page number, title)

3. **Embedding**: Each chunk is converted to a vector representation:
   - Uses OpenAI's embedding model (`text-embedding-ada-002`)
   - Creates 1536-dimensional vectors capturing semantic meaning

4. **Vector Storage**: Embeddings are stored in ChromaDB:
   - Persistent storage on disk
   - Efficient similarity search
   - Metadata preserved for each chunk

### 2. Retrieval Phase (Per Feature)

For each feature in your dataset, the system retrieves relevant context:

```
Feature Name → Query Embedding → Similarity Search → Top-K Chunks
```

**Step-by-step:**

1. **Query Formation**: The feature name (e.g., "Blood Pressure") becomes a search query

2. **Query Embedding**: The query is converted to the same vector space as documents

3. **Similarity Search**: ChromaDB finds the most similar document chunks:
   - Uses cosine similarity
   - Returns top-K results (default: 3 documents)
   - Includes relevance scores

4. **Context Assembly**: Retrieved chunks are formatted as context for the LLM

### 3. Generation Phase (LLM Scoring)

The LLM generates penalty scores using the retrieved context:

```
Prompt + Context + Feature → LLM → Penalty Score (1-5)
```

**Step-by-step:**

1. **Prompt Construction**: Combines:
   - System prompt (expert role, task description)
   - Retrieved document context
   - Feature name and target condition
   - Scoring instructions

2. **LLM Processing**: GPT-4o (or configured model) evaluates:
   - Relevance of feature to target condition
   - Evidence from retrieved documents
   - Domain knowledge from training

3. **Score Output**: Returns penalty score from 1-5:
   - **1-2**: Highly relevant (lower Lasso penalty)
   - **3**: Moderately relevant
   - **4-5**: Less relevant (higher Lasso penalty)

## Why RAG Improves LLM-Lasso

| Without RAG | With RAG |
|-------------|----------|
| LLM relies only on training data | LLM has access to domain-specific papers |
| May miss recent research | Can include latest publications |
| Generic knowledge only | Specific to your research domain |
| Potential hallucinations | Grounded in actual documents |

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Characters per text chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `pdf_rag_num_docs` | 3 | Documents retrieved per query |
| `persist_directory` | `pdf_vectorstore` | ChromaDB storage path |
| `collection_name` | `scientific_papers` | ChromaDB collection name |

## Example: How Context Flows

**Feature**: "Depression Score"  
**Target**: "Suicidal Ideation"

1. **Query**: "Depression Score suicidal ideation relevance"

2. **Retrieved Chunks**:
   ```
   Chunk 1 (Similarity: 0.89): "CDRS-R depression scores above 40 were 
   strongly associated with increased suicidal ideation in adolescents..."
   
   Chunk 2 (Similarity: 0.85): "Severity of depressive symptoms, as 
   measured by standardized instruments, is a key predictor..."
   
   Chunk 3 (Similarity: 0.82): "The relationship between depression 
   and suicidal behavior has been extensively documented..."
   ```

3. **LLM Evaluation**: Based on this evidence, assigns score of **2** (highly relevant)

4. **Lasso Impact**: Low penalty → feature more likely retained in final model

## File Structure

```
LLM-Lasso/
├── pdf_vectorstore/           # ChromaDB persistent storage
│   ├── chroma.sqlite3         # Vector database
│   └── ...
├── sample_pdfs/               # Your PDF documents
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
└── src/llm_lasso/llm_penalty/rag/
    ├── pdf_RAG_process.py     # PDF text extraction
    └── pdf_vectorstore.py     # ChromaDB management
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Set via environment: `export OPENAI_API_KEY='your-key'`
   - Or add to `.env` file in project root

2. **"Vectorstore not found"**
   - Run indexing first: `python playground/interactive_pdf_RAG.py`
   - Check `persist_directory` path

3. **"No relevant documents found"**
   - Verify PDFs were indexed correctly
   - Check if feature names match document content
   - Try increasing `pdf_rag_num_docs`

---

For implementation details, see the [main README](../README.md).
