"""
RAG (Retrieval-Augmented Generation) module for LLM-Lasso.

This module provides functionality for retrieving context from various sources:
- OMIM database (online medical genetics database)
- PubMed (biomedical literature)
- PDF documents (local scientific papers)
"""

from llm_lasso.llm_penalty.rag.rag_context import get_rag_context
from llm_lasso.llm_penalty.rag.pdf_RAG_process import (
    extract_text_from_pdf,
    load_pdfs_from_directory,
    get_pdf_retrieval_context
)
from llm_lasso.llm_penalty.rag.pdf_vectorstore import (
    create_pdf_vectorstore,
    load_pdf_vectorstore,
    get_or_create_pdf_vectorstore,
    add_pdfs_to_vectorstore,
    delete_pdf_vectorstore
)

__all__ = [
    # Main RAG context function
    "get_rag_context",
    
    # PDF extraction functions
    "extract_text_from_pdf",
    "load_pdfs_from_directory",
    "get_pdf_retrieval_context",
    
    # PDF vectorstore functions
    "create_pdf_vectorstore",
    "load_pdf_vectorstore",
    "get_or_create_pdf_vectorstore",
    "add_pdfs_to_vectorstore",
    "delete_pdf_vectorstore",
]

