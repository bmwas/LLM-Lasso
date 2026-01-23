"""
LLM Penalty module for LLM-Lasso.

This module provides functionality for:
- LLM-based penalty score generation
- Embedding backends (OpenAI, vLLM)
- Query processing and score collection
"""

from llm_lasso.llm_penalty.embeddings import (
    EmbeddingBackend,
    VLLMEmbeddings,
    get_embeddings
)
from llm_lasso.llm_penalty.llm import (
    LLMType,
    LLMQueryWrapperWithMemory
)
from llm_lasso.llm_penalty.penalty_collection import collect_penalties
from llm_lasso.llm_penalty.query_scores import query_scores_with_retries

__all__ = [
    # Embedding backends
    "EmbeddingBackend",
    "VLLMEmbeddings", 
    "get_embeddings",
    
    # LLM types and wrappers
    "LLMType",
    "LLMQueryWrapperWithMemory",
    
    # Penalty collection
    "collect_penalties",
    
    # Query scores
    "query_scores_with_retries",
]
