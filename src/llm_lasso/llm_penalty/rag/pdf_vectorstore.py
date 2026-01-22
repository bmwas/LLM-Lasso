"""
This script provides functionality to create, load, and manage ChromaDB vectorstores
for PDF documents used in RAG (Retrieval-Augmented Generation).

Extensive logging is available - set LOG_LEVEL=DEBUG for detailed output.
"""

import os
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from llm_lasso.llm_penalty.rag.pdf_RAG_process import (
    load_pdfs_from_directory,
    extract_text_from_pdf
)
from llm_lasso.utils.chunking import chunk_pdf_documents

# Set up module logger
logger = logging.getLogger("pdf_rag.vectorstore")

# If no handlers exist, add a basic one (for standalone usage)
if not logger.handlers and not logging.getLogger("pdf_rag").handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _log_directory_contents(directory: str, label: str = "Directory") -> Dict[str, Any]:
    """Log contents of a directory and return stats."""
    stats = {
        "path": directory,
        "exists": os.path.exists(directory),
        "files": [],
        "total_size": 0
    }
    
    if stats["exists"]:
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    stats["files"].append({"name": item, "size": size})
                    stats["total_size"] += size
                    logger.debug(f"  {label} file: {item} ({size:,} bytes)")
        except Exception as e:
            logger.warning(f"Could not list directory {directory}: {e}")
    
    logger.debug(f"{label} stats: {len(stats['files'])} files, {stats['total_size']:,} bytes total")
    return stats


def _log_collection_info(collection, label: str = "Collection") -> Dict[str, Any]:
    """Log detailed collection information."""
    info = {
        "name": None,
        "count": 0,
        "metadata": None
    }
    
    try:
        info["name"] = collection.name
        info["count"] = collection.count()
        info["metadata"] = collection.metadata
        
        logger.debug(f"{label} name: {info['name']}")
        logger.debug(f"{label} document count: {info['count']}")
        logger.debug(f"{label} metadata: {info['metadata']}")
        
        # Sample peek if documents exist
        if info["count"] > 0:
            sample = collection.peek(limit=3)
            logger.debug(f"{label} sample IDs: {sample.get('ids', [])}")
            if sample.get('metadatas'):
                unique_sources = set()
                for m in sample.get('metadatas', []):
                    if m and 'filename' in m:
                        unique_sources.add(m['filename'])
                logger.debug(f"{label} sample sources: {unique_sources}")
    except Exception as e:
        logger.warning(f"Could not get {label.lower()} info: {e}")
    
    return info


def create_pdf_vectorstore(
    pdf_directory: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_chunks: bool = True,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "pdf_documents",
    filter_references: bool = True
) -> Chroma:
    """
    Create a ChromaDB vectorstore from PDF documents in a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        persist_directory: Path to directory where ChromaDB will be persisted.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        page_chunks: If True, extract text page by page before chunking.
        embedding_model: LangChain Embeddings model (OpenAI, vLLM, etc.). 
                        If None, creates default OpenAI embeddings.
        collection_name: Name for the ChromaDB collection.
        filter_references: If True, filter out reference/bibliography sections from indexing.
    
    Returns:
        Chroma vectorstore populated with PDF document chunks.
    """
    operation_id = hashlib.md5(f"{pdf_directory}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Op:{operation_id}] ===== CREATE VECTORSTORE =====")
    logger.info(f"[Op:{operation_id}] Starting vectorstore creation")
    logger.debug(f"[Op:{operation_id}] Parameters:")
    logger.debug(f"  pdf_directory: {pdf_directory}")
    logger.debug(f"  persist_directory: {persist_directory}")
    logger.debug(f"  chunk_size: {chunk_size}")
    logger.debug(f"  chunk_overlap: {chunk_overlap}")
    logger.debug(f"  page_chunks: {page_chunks}")
    logger.debug(f"  collection_name: {collection_name}")
    logger.debug(f"  filter_references: {filter_references}")
    logger.debug(f"  embedding_model provided: {embedding_model is not None}")
    
    total_start = time.time()
    
    # Log PDF directory contents
    logger.info(f"[Op:{operation_id}] Scanning PDF directory...")
    pdf_stats = _log_directory_contents(pdf_directory, "PDF")
    pdf_files = [f for f in pdf_stats["files"] if f["name"].lower().endswith('.pdf')]
    logger.info(f"[Op:{operation_id}] Found {len(pdf_files)} PDF files")
    for pdf in pdf_files:
        logger.debug(f"  - {pdf['name']} ({pdf['size']:,} bytes)")
    
    # Initialize embeddings
    if embedding_model is None:
        logger.info(f"[Op:{operation_id}] Creating default OpenAI embeddings...")
        logger.warning(f"[Op:{operation_id}] No embedding_model provided, defaulting to OpenAI. "
                      "Consider using get_embeddings() for explicit backend selection.")
        embed_start = time.time()
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings()
        embed_time = time.time() - embed_start
        logger.debug(f"[Op:{operation_id}] Embeddings created in {embed_time:.2f}s")
    else:
        logger.debug(f"[Op:{operation_id}] Using provided embedding model: {type(embedding_model).__name__}")
    
    # Load and extract text from PDFs
    logger.info(f"[Op:{operation_id}] Loading PDFs from {pdf_directory}...")
    if filter_references:
        logger.info(f"[Op:{operation_id}] Reference filtering: ENABLED (page-level)")
    print(f"Loading PDFs from {pdf_directory}...")
    
    load_start = time.time()
    raw_documents, page_filter_stats = load_pdfs_from_directory(
        pdf_directory,
        page_chunks=page_chunks,
        recursive=False,
        filter_references=filter_references
    )
    load_time = time.time() - load_start
    
    logger.info(f"[Op:{operation_id}] PDF loading completed in {load_time:.2f}s")
    logger.info(f"[Op:{operation_id}] Extracted {len(raw_documents)} raw document segments")
    
    # Log page-level filtering statistics
    if filter_references and page_filter_stats.get('total_pages_filtered', 0) > 0:
        total_pages = page_filter_stats.get('total_pages_extracted', 0) + page_filter_stats.get('total_pages_filtered', 0)
        logger.info(
            f"[Op:{operation_id}] Page-level reference filtering: "
            f"{page_filter_stats['total_pages_filtered']}/{total_pages} pages filtered "
            f"({100*page_filter_stats['total_pages_filtered']/max(total_pages,1):.1f}%)"
        )
    
    if not raw_documents:
        logger.error(f"[Op:{operation_id}] No documents extracted from {pdf_directory}")
        raise ValueError(f"No documents extracted from {pdf_directory}")
    
    # Log raw document statistics
    total_raw_chars = sum(len(doc.get("content", "")) for doc in raw_documents)
    logger.debug(f"[Op:{operation_id}] Total raw content: {total_raw_chars:,} characters")
    
    sources = set()
    for doc in raw_documents:
        if doc.get("metadata", {}).get("filename"):
            sources.add(doc["metadata"]["filename"])
    logger.debug(f"[Op:{operation_id}] Unique source files: {sources}")
    
    # Chunk the documents
    logger.info(f"[Op:{operation_id}] Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    if filter_references:
        logger.info(f"[Op:{operation_id}] Reference filtering: ENABLED (chunk-level)")
    print("Chunking documents...")
    
    chunk_start = time.time()
    chunked_documents, chunk_filter_stats = chunk_pdf_documents(
        raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        filter_references=filter_references
    )
    chunk_time = time.time() - chunk_start
    
    logger.info(f"[Op:{operation_id}] Chunking completed in {chunk_time:.2f}s")
    logger.info(f"[Op:{operation_id}] Created {len(chunked_documents)} chunks from {len(raw_documents)} segments")
    
    # Log chunk-level filtering statistics
    if filter_references and chunk_filter_stats.get('chunks_filtered', 0) > 0:
        logger.info(
            f"[Op:{operation_id}] Chunk-level reference filtering: "
            f"{chunk_filter_stats['chunks_filtered']}/{chunk_filter_stats['total_chunks_created']} chunks filtered "
            f"({100*chunk_filter_stats['chunks_filtered']/max(chunk_filter_stats['total_chunks_created'],1):.1f}%)"
        )
    
    if not chunked_documents:
        logger.error(f"[Op:{operation_id}] No chunks created from documents")
        raise ValueError("No chunks created from documents")
    
    # Log chunk statistics
    chunk_lengths = [len(doc.get("content", "")) for doc in chunked_documents]
    avg_chunk_len = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    min_chunk_len = min(chunk_lengths) if chunk_lengths else 0
    max_chunk_len = max(chunk_lengths) if chunk_lengths else 0
    
    logger.debug(f"[Op:{operation_id}] Chunk statistics:")
    logger.debug(f"  Total chunks: {len(chunked_documents)}")
    logger.debug(f"  Avg chunk length: {avg_chunk_len:.1f} chars")
    logger.debug(f"  Min chunk length: {min_chunk_len} chars")
    logger.debug(f"  Max chunk length: {max_chunk_len} chars")
    logger.debug(f"  Total content: {sum(chunk_lengths):,} chars")
    
    # Log sample chunks
    for i, chunk in enumerate(chunked_documents[:3]):
        logger.debug(f"[Op:{operation_id}] Sample chunk {i+1}:")
        logger.debug(f"  Metadata: {chunk.get('metadata', {})}")
        logger.debug(f"  Content length: {len(chunk.get('content', ''))} chars")
        logger.debug(f"  Content preview: {chunk.get('content', '')[:100]}...")
    
    # Convert to LangChain Document objects
    logger.info(f"[Op:{operation_id}] Converting to LangChain Documents...")
    print("Converting to LangChain Documents...")
    
    convert_start = time.time()
    documents_wrapped = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in chunked_documents
    ]
    convert_time = time.time() - convert_start
    logger.debug(f"[Op:{operation_id}] Conversion completed in {convert_time:.3f}s")
    
    # Ensure persist directory exists
    logger.debug(f"[Op:{operation_id}] Ensuring persist directory exists: {persist_directory}")
    os.makedirs(persist_directory, exist_ok=True)
    logger.debug(f"[Op:{operation_id}] Persist directory ready")
    
    # Create the vectorstore
    logger.info(f"[Op:{operation_id}] Creating ChromaDB vectorstore...")
    logger.info(f"[Op:{operation_id}] Indexing {len(documents_wrapped)} documents...")
    print(f"Creating ChromaDB vectorstore with {len(documents_wrapped)} documents...")
    
    index_start = time.time()
    try:
        vectorstore = Chroma.from_documents(
            documents=documents_wrapped,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        index_time = time.time() - index_start
        logger.info(f"[Op:{operation_id}] ChromaDB indexing completed in {index_time:.2f}s")
        logger.debug(f"[Op:{operation_id}] Indexing rate: {len(documents_wrapped)/index_time:.1f} docs/sec")
    except Exception as e:
        logger.error(f"[Op:{operation_id}] ChromaDB creation failed: {e}")
        logger.exception("Full traceback:")
        raise
    
    # Persist the database
    logger.info(f"[Op:{operation_id}] Persisting vectorstore to disk...")
    print(f"Persisting vectorstore to {persist_directory}...")
    
    persist_start = time.time()
    try:
        vectorstore.persist()
        persist_time = time.time() - persist_start
        logger.info(f"[Op:{operation_id}] Persistence completed in {persist_time:.2f}s")
    except Exception as e:
        logger.error(f"[Op:{operation_id}] Persistence failed: {e}")
        logger.exception("Full traceback:")
        raise
    
    # Log final collection info
    _log_collection_info(vectorstore._collection, f"[Op:{operation_id}] Final collection")
    
    # Log persist directory after creation
    _log_directory_contents(persist_directory, f"[Op:{operation_id}] Persist")
    
    total_time = time.time() - total_start
    logger.info(f"[Op:{operation_id}] ===== VECTORSTORE CREATED =====")
    logger.info(f"[Op:{operation_id}] Total time: {total_time:.2f}s")
    logger.info(f"[Op:{operation_id}] Documents indexed: {len(documents_wrapped)}")
    logger.info(f"[Op:{operation_id}] Collection: {collection_name}")
    logger.info(f"[Op:{operation_id}] Persist path: {persist_directory}")
    
    print(f"Successfully created vectorstore with {len(documents_wrapped)} document chunks")
    return vectorstore


def load_pdf_vectorstore(
    persist_directory: str,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "pdf_documents"
) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.
    
    Args:
        persist_directory: Path to directory where ChromaDB is persisted.
        embedding_model: LangChain Embeddings model (OpenAI, vLLM, etc.).
                        If None, creates default OpenAI embeddings.
        collection_name: Name of the ChromaDB collection.
    
    Returns:
        Loaded Chroma vectorstore.
    
    Raises:
        FileNotFoundError: If persist_directory does not exist.
    """
    operation_id = hashlib.md5(f"{persist_directory}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Op:{operation_id}] ===== LOAD VECTORSTORE =====")
    logger.info(f"[Op:{operation_id}] Loading vectorstore from: {persist_directory}")
    logger.debug(f"[Op:{operation_id}] collection_name: {collection_name}")
    logger.debug(f"[Op:{operation_id}] embedding_model provided: {embedding_model is not None}")
    
    load_start = time.time()
    
    # Check if directory exists
    if not os.path.exists(persist_directory):
        logger.error(f"[Op:{operation_id}] Vectorstore not found at {persist_directory}")
        raise FileNotFoundError(
            f"Vectorstore not found at {persist_directory}. "
            "Use create_pdf_vectorstore() to create a new one."
        )
    
    # Log directory contents
    _log_directory_contents(persist_directory, f"[Op:{operation_id}] Persist")
    
    # Initialize embeddings
    if embedding_model is None:
        logger.info(f"[Op:{operation_id}] Creating default OpenAI embeddings...")
        logger.warning(f"[Op:{operation_id}] No embedding_model provided, defaulting to OpenAI. "
                      "Consider using get_embeddings() for explicit backend selection.")
        embed_start = time.time()
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings()
        embed_time = time.time() - embed_start
        logger.debug(f"[Op:{operation_id}] Embeddings created in {embed_time:.2f}s")
    
    logger.info(f"[Op:{operation_id}] Loading ChromaDB from disk...")
    print(f"Loading existing vectorstore from {persist_directory}...")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"[Op:{operation_id}] Failed to load vectorstore: {e}")
        logger.exception("Full traceback:")
        raise
    
    # Get collection stats
    collection = vectorstore._collection
    doc_count = collection.count()
    
    logger.info(f"[Op:{operation_id}] Vectorstore loaded successfully")
    _log_collection_info(collection, f"[Op:{operation_id}] Loaded collection")
    
    # Log unique sources in collection
    if doc_count > 0:
        try:
            all_meta = collection.get(include=["metadatas"])
            unique_sources = set()
            pages_per_source = {}
            for meta in all_meta.get('metadatas', []):
                if meta:
                    filename = meta.get('filename', 'Unknown')
                    page = meta.get('page', 0)
                    unique_sources.add(filename)
                    if filename not in pages_per_source:
                        pages_per_source[filename] = set()
                    pages_per_source[filename].add(page)
            
            logger.debug(f"[Op:{operation_id}] Sources in collection:")
            for source in sorted(unique_sources):
                pages = sorted(pages_per_source.get(source, []))
                page_range = f"{min(pages)}-{max(pages)}" if pages else "N/A"
                logger.debug(f"  - {source}: {len(pages)} pages ({page_range})")
        except Exception as e:
            logger.warning(f"[Op:{operation_id}] Could not enumerate sources: {e}")
    
    load_time = time.time() - load_start
    logger.info(f"[Op:{operation_id}] ===== VECTORSTORE LOADED =====")
    logger.info(f"[Op:{operation_id}] Load time: {load_time:.2f}s")
    logger.info(f"[Op:{operation_id}] Documents: {doc_count}")
    
    print(f"Loaded vectorstore with {doc_count} documents")
    return vectorstore


def get_or_create_pdf_vectorstore(
    pdf_directory: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    page_chunks: bool = True,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "pdf_documents",
    force_recreate: bool = False,
    filter_references: bool = True
) -> Chroma:
    """
    Get existing vectorstore or create a new one if it doesn't exist.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        persist_directory: Path to directory for ChromaDB persistence.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        page_chunks: If True, extract text page by page before chunking.
        embedding_model: LangChain Embeddings model (OpenAI, vLLM, etc.).
                        If None, creates default OpenAI embeddings.
        collection_name: Name for the ChromaDB collection.
        force_recreate: If True, recreate vectorstore even if it exists.
        filter_references: If True, filter out reference/bibliography sections from indexing.
    
    Returns:
        Chroma vectorstore.
    """
    operation_id = hashlib.md5(f"{persist_directory}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Op:{operation_id}] ===== GET OR CREATE VECTORSTORE =====")
    logger.debug(f"[Op:{operation_id}] pdf_directory: {pdf_directory}")
    logger.debug(f"[Op:{operation_id}] persist_directory: {persist_directory}")
    logger.debug(f"[Op:{operation_id}] force_recreate: {force_recreate}")
    logger.debug(f"[Op:{operation_id}] filter_references: {filter_references}")
    
    # Initialize embeddings
    if embedding_model is None:
        logger.info(f"[Op:{operation_id}] Creating default OpenAI embeddings...")
        logger.warning(f"[Op:{operation_id}] No embedding_model provided, defaulting to OpenAI. "
                      "Consider using get_embeddings() for explicit backend selection.")
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings()
    
    # Check if vectorstore already exists
    persist_exists = os.path.exists(persist_directory)
    logger.debug(f"[Op:{operation_id}] Persist directory exists: {persist_exists}")
    
    if persist_exists and not force_recreate:
        # Check if it has actual data
        try:
            persist_contents = os.listdir(persist_directory)
            logger.debug(f"[Op:{operation_id}] Persist directory contents: {persist_contents}")
            has_data = len(persist_contents) > 0
        except Exception as e:
            logger.warning(f"[Op:{operation_id}] Could not check persist contents: {e}")
            has_data = False
        
        if has_data:
            logger.info(f"[Op:{operation_id}] Found existing vectorstore, loading...")
            print("Found existing vectorstore, loading...")
            return load_pdf_vectorstore(
                persist_directory,
                embedding_model=embedding_model,
                collection_name=collection_name
            )
        else:
            logger.warning(f"[Op:{operation_id}] Persist directory exists but is empty")
    
    if force_recreate:
        logger.info(f"[Op:{operation_id}] Force recreate requested")
    
    # Create new vectorstore
    logger.info(f"[Op:{operation_id}] Creating new vectorstore...")
    print("Creating new vectorstore...")
    return create_pdf_vectorstore(
        pdf_directory=pdf_directory,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_chunks=page_chunks,
        embedding_model=embedding_model,
        collection_name=collection_name,
        filter_references=filter_references
    )


def add_pdfs_to_vectorstore(
    vectorstore: Chroma,
    pdf_paths: list,
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
    operation_id = hashlib.md5(f"{str(pdf_paths)}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Op:{operation_id}] ===== ADD PDFS TO VECTORSTORE =====")
    logger.info(f"[Op:{operation_id}] Adding {len(pdf_paths)} PDF files")
    
    for path in pdf_paths:
        logger.debug(f"[Op:{operation_id}] - {path}")
    
    # Log initial state
    initial_count = vectorstore._collection.count()
    logger.debug(f"[Op:{operation_id}] Initial document count: {initial_count}")
    
    add_start = time.time()
    all_documents = []
    
    for pdf_path in pdf_paths:
        logger.info(f"[Op:{operation_id}] Processing: {pdf_path}")
        print(f"Processing {pdf_path}...")
        try:
            file_start = time.time()
            raw_docs = extract_text_from_pdf(pdf_path, page_chunks=page_chunks)
            file_time = time.time() - file_start
            logger.debug(f"[Op:{operation_id}] Extracted {len(raw_docs)} segments in {file_time:.2f}s")
            all_documents.extend(raw_docs)
        except Exception as e:
            logger.error(f"[Op:{operation_id}] Error processing {pdf_path}: {e}")
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    if not all_documents:
        logger.warning(f"[Op:{operation_id}] No new documents to add")
        print("No new documents to add")
        return vectorstore
    
    logger.info(f"[Op:{operation_id}] Total raw segments: {len(all_documents)}")
    
    # Chunk the documents
    logger.info(f"[Op:{operation_id}] Chunking documents...")
    chunk_start = time.time()
    chunked_documents = chunk_pdf_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_time = time.time() - chunk_start
    logger.info(f"[Op:{operation_id}] Created {len(chunked_documents)} chunks in {chunk_time:.2f}s")
    
    # Convert to LangChain Document objects
    documents_wrapped = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in chunked_documents
    ]
    
    # Add to vectorstore
    logger.info(f"[Op:{operation_id}] Adding {len(documents_wrapped)} chunks to vectorstore...")
    print(f"Adding {len(documents_wrapped)} document chunks to vectorstore...")
    
    index_start = time.time()
    try:
        vectorstore.add_documents(documents_wrapped)
        index_time = time.time() - index_start
        logger.info(f"[Op:{operation_id}] Documents added in {index_time:.2f}s")
    except Exception as e:
        logger.error(f"[Op:{operation_id}] Failed to add documents: {e}")
        raise
    
    # Persist changes
    logger.info(f"[Op:{operation_id}] Persisting changes...")
    persist_start = time.time()
    vectorstore.persist()
    persist_time = time.time() - persist_start
    logger.debug(f"[Op:{operation_id}] Persistence completed in {persist_time:.2f}s")
    
    # Log final state
    final_count = vectorstore._collection.count()
    added_count = final_count - initial_count
    total_time = time.time() - add_start
    
    logger.info(f"[Op:{operation_id}] ===== PDFS ADDED =====")
    logger.info(f"[Op:{operation_id}] Documents added: {added_count}")
    logger.info(f"[Op:{operation_id}] Total documents: {final_count}")
    logger.info(f"[Op:{operation_id}] Total time: {total_time:.2f}s")
    
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
    
    operation_id = hashlib.md5(f"{persist_directory}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Op:{operation_id}] ===== DELETE VECTORSTORE =====")
    logger.info(f"[Op:{operation_id}] Deleting: {persist_directory}")
    
    if not os.path.exists(persist_directory):
        logger.warning(f"[Op:{operation_id}] Vectorstore not found at {persist_directory}")
        print(f"Vectorstore not found at {persist_directory}")
        return False
    
    # Log what will be deleted
    _log_directory_contents(persist_directory, f"[Op:{operation_id}] To delete")
    
    try:
        delete_start = time.time()
        shutil.rmtree(persist_directory)
        delete_time = time.time() - delete_start
        logger.info(f"[Op:{operation_id}] Vectorstore deleted in {delete_time:.2f}s")
        print(f"Successfully deleted vectorstore at {persist_directory}")
        return True
    except Exception as e:
        logger.error(f"[Op:{operation_id}] Deletion failed: {e}")
        logger.exception("Full traceback:")
        print(f"Error deleting vectorstore: {e}")
        return False


def get_vectorstore_info(vectorstore: Chroma) -> Dict[str, Any]:
    """
    Get detailed information about a vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore instance.
    
    Returns:
        Dictionary containing vectorstore information.
    """
    logger.info("Getting vectorstore info...")
    
    info = {
        "collection_name": None,
        "document_count": 0,
        "unique_sources": [],
        "source_stats": {},
        "metadata": None,
        "sample_documents": []
    }
    
    try:
        collection = vectorstore._collection
        info["collection_name"] = collection.name
        info["document_count"] = collection.count()
        info["metadata"] = collection.metadata
        
        logger.debug(f"Collection: {info['collection_name']}")
        logger.debug(f"Documents: {info['document_count']}")
        
        if info["document_count"] > 0:
            # Get all metadata
            all_meta = collection.get(include=["metadatas"])
            
            for meta in all_meta.get('metadatas', []):
                if meta:
                    filename = meta.get('filename', 'Unknown')
                    page = meta.get('page', 0)
                    
                    if filename not in info["source_stats"]:
                        info["source_stats"][filename] = {
                            "chunks": 0,
                            "pages": set()
                        }
                    info["source_stats"][filename]["chunks"] += 1
                    info["source_stats"][filename]["pages"].add(page)
            
            info["unique_sources"] = list(info["source_stats"].keys())
            
            # Convert sets to lists for JSON serialization
            for source in info["source_stats"]:
                info["source_stats"][source]["pages"] = sorted(
                    list(info["source_stats"][source]["pages"])
                )
            
            # Get sample documents
            sample = collection.peek(limit=5)
            for i, (doc_id, doc_text, meta) in enumerate(zip(
                sample.get('ids', []),
                sample.get('documents', []),
                sample.get('metadatas', [])
            )):
                info["sample_documents"].append({
                    "id": doc_id,
                    "metadata": meta,
                    "content_preview": doc_text[:200] if doc_text else None
                })
            
            logger.debug(f"Unique sources: {info['unique_sources']}")
            
    except Exception as e:
        logger.error(f"Failed to get vectorstore info: {e}")
        info["error"] = str(e)
    
    return info
