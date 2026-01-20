"""
This script provides functionality to create, load, and manage ChromaDB vectorstores
for PDF documents used in RAG (Retrieval-Augmented Generation).
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from llm_lasso.llm_penalty.rag.pdf_RAG_process import (
    load_pdfs_from_directory,
    extract_text_from_pdf
)
from llm_lasso.utils.chunking import chunk_pdf_documents


def create_pdf_vectorstore(
    pdf_directory: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_chunks: bool = True,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    collection_name: str = "pdf_documents"
) -> Chroma:
    """
    Create a ChromaDB vectorstore from PDF documents in a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        persist_directory: Path to directory where ChromaDB will be persisted.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        page_chunks: If True, extract text page by page before chunking.
        embedding_model: OpenAI embeddings model. If None, creates default.
        collection_name: Name for the ChromaDB collection.
    
    Returns:
        Chroma vectorstore populated with PDF document chunks.
    """
    # Initialize embeddings
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()
    
    # Load and extract text from PDFs
    print(f"Loading PDFs from {pdf_directory}...")
    raw_documents = load_pdfs_from_directory(
        pdf_directory,
        page_chunks=page_chunks,
        recursive=False
    )
    
    if not raw_documents:
        raise ValueError(f"No documents extracted from {pdf_directory}")
    
    # Chunk the documents
    print("Chunking documents...")
    chunked_documents = chunk_pdf_documents(
        raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunked_documents:
        raise ValueError("No chunks created from documents")
    
    # Convert to LangChain Document objects
    print("Converting to LangChain Documents...")
    documents_wrapped = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in chunked_documents
    ]
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create the vectorstore
    print(f"Creating ChromaDB vectorstore with {len(documents_wrapped)} documents...")
    vectorstore = Chroma.from_documents(
        documents=documents_wrapped,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Persist the database
    print(f"Persisting vectorstore to {persist_directory}...")
    vectorstore.persist()
    
    print(f"Successfully created vectorstore with {len(documents_wrapped)} document chunks")
    return vectorstore


def load_pdf_vectorstore(
    persist_directory: str,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    collection_name: str = "pdf_documents"
) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.
    
    Args:
        persist_directory: Path to directory where ChromaDB is persisted.
        embedding_model: OpenAI embeddings model. If None, creates default.
        collection_name: Name of the ChromaDB collection.
    
    Returns:
        Loaded Chroma vectorstore.
    
    Raises:
        FileNotFoundError: If persist_directory does not exist.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vectorstore not found at {persist_directory}. "
            "Use create_pdf_vectorstore() to create a new one."
        )
    
    # Initialize embeddings
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()
    
    print(f"Loading existing vectorstore from {persist_directory}...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    
    # Get collection stats
    collection = vectorstore._collection
    doc_count = collection.count()
    print(f"Loaded vectorstore with {doc_count} documents")
    
    return vectorstore


def get_or_create_pdf_vectorstore(
    pdf_directory: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_chunks: bool = True,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    collection_name: str = "pdf_documents",
    force_recreate: bool = False
) -> Chroma:
    """
    Get existing vectorstore or create a new one if it doesn't exist.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        persist_directory: Path to directory for ChromaDB persistence.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        page_chunks: If True, extract text page by page before chunking.
        embedding_model: OpenAI embeddings model. If None, creates default.
        collection_name: Name for the ChromaDB collection.
        force_recreate: If True, recreate vectorstore even if it exists.
    
    Returns:
        Chroma vectorstore.
    """
    # Initialize embeddings
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()
    
    # Check if vectorstore already exists
    if os.path.exists(persist_directory) and not force_recreate:
        print("Found existing vectorstore, loading...")
        return load_pdf_vectorstore(
            persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
    
    # Create new vectorstore
    print("Creating new vectorstore...")
    return create_pdf_vectorstore(
        pdf_directory=pdf_directory,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_chunks=page_chunks,
        embedding_model=embedding_model,
        collection_name=collection_name
    )


def add_pdfs_to_vectorstore(
    vectorstore: Chroma,
    pdf_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_chunks: bool = True
) -> Chroma:
    """
    Add new PDF documents to an existing vectorstore.
    
    Args:
        vectorstore: Existing Chroma vectorstore.
        pdf_paths: List of paths to PDF files to add.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        page_chunks: If True, extract text page by page before chunking.
    
    Returns:
        Updated Chroma vectorstore.
    """
    all_documents = []
    
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        try:
            raw_docs = extract_text_from_pdf(pdf_path, page_chunks=page_chunks)
            all_documents.extend(raw_docs)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    if not all_documents:
        print("No new documents to add")
        return vectorstore
    
    # Chunk the documents
    chunked_documents = chunk_pdf_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Convert to LangChain Document objects
    documents_wrapped = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in chunked_documents
    ]
    
    # Add to vectorstore
    print(f"Adding {len(documents_wrapped)} document chunks to vectorstore...")
    vectorstore.add_documents(documents_wrapped)
    
    # Persist changes
    vectorstore.persist()
    
    print(f"Successfully added {len(documents_wrapped)} chunks to vectorstore")
    return vectorstore


def delete_pdf_vectorstore(persist_directory: str) -> bool:
    """
    Delete a ChromaDB vectorstore from disk.
    
    Args:
        persist_directory: Path to the vectorstore directory.
    
    Returns:
        True if deletion was successful, False otherwise.
    """
    import shutil
    
    if not os.path.exists(persist_directory):
        print(f"Vectorstore not found at {persist_directory}")
        return False
    
    try:
        shutil.rmtree(persist_directory)
        print(f"Successfully deleted vectorstore at {persist_directory}")
        return True
    except Exception as e:
        print(f"Error deleting vectorstore: {e}")
        return False

