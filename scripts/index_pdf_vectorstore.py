#!/usr/bin/env python3
"""
PDF Vectorstore Indexing Script for LLM-Lasso

This script creates a ChromaDB vectorstore from PDF documents, supporting both
OpenAI and vLLM embedding backends. The vectorstore is used for RAG (Retrieval-
Augmented Generation) in the LLM-Lasso pipeline.

Usage:
    # Index with OpenAI embeddings (default)
    python scripts/index_pdf_vectorstore.py \
        --pdf-directory /path/to/pdfs \
        --persist-directory ./pdf_vectorstore \
        --embedding-backend openai

    # Index with vLLM embeddings (open-source)
    python scripts/index_pdf_vectorstore.py \
        --pdf-directory /path/to/pdfs \
        --persist-directory ./pdf_vectorstore \
        --embedding-backend vllm

Environment Variables:
    For OpenAI:
        - OPENAI_API_KEY: Your OpenAI API key
    
    For vLLM:
        - VLLM_EMBED_BASE_URL: Base URL for vLLM embedding service (default: http://localhost:8001/v1)
        - VLLM_API_KEY: API key for vLLM (if configured)
        - VLLM_EMBED_MODEL: Model name (default: qwen3-embed)

Notes:
    - Qwen3-Embedding-8B (used by default vLLM config) produces 4096-dimensional embeddings
    - OpenAI text-embedding-ada-002 produces 1536-dimensional embeddings
    - You MUST use the same embedding backend for indexing and querying
    - By default, existing indexes are cleaned before recreating (use --no-clean to append)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root and src to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Set up logging with console and optional file output."""
    logger = logging.getLogger("pdf_indexer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    console_handler.setFormatter(logging.Formatter(console_format, datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger


def get_embedding_model(backend: str, logger: logging.Logger):
    """
    Get the appropriate embedding model based on the backend.
    
    Args:
        backend: Either 'openai' or 'vllm'
        logger: Logger instance
        
    Returns:
        LangChain Embeddings instance
    """
    from llm_lasso.llm_penalty.embeddings import get_embeddings, VLLMEmbeddings
    
    if backend.lower() == "openai":
        logger.info("Using OpenAI embeddings backend")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI embeddings. "
                "Set it in your .env file or export it."
            )
        logger.debug(f"OpenAI API key found: {api_key[:8]}...{api_key[-4:]}")
        return get_embeddings(backend="openai")
    
    elif backend.lower() == "vllm":
        logger.info("Using vLLM embeddings backend (open-source)")
        
        # Check multiple environment variable names for flexibility
        # Priority: EMBED_BASE_URL > VLLM_EMBED_BASE_URL > default
        base_url = os.environ.get(
            "EMBED_BASE_URL", 
            os.environ.get("VLLM_EMBED_BASE_URL", "http://localhost:8001/v1")
        )
        api_key = os.environ.get("VLLM_API_KEY", "")
        model = os.environ.get("VLLM_EMBED_MODEL", "qwen3-embed")
        
        # Log which env var was used
        if "EMBED_BASE_URL" in os.environ:
            logger.info(f"  Base URL: {base_url} (from EMBED_BASE_URL)")
        elif "VLLM_EMBED_BASE_URL" in os.environ:
            logger.info(f"  Base URL: {base_url} (from VLLM_EMBED_BASE_URL)")
        else:
            logger.info(f"  Base URL: {base_url} (default)")
        
        logger.info(f"  Model: {model}")
        logger.info(f"  API Key: {'configured' if api_key else 'not set'}")
        
        # Test connection before proceeding
        logger.info("Testing vLLM embedding endpoint...")
        embeddings = VLLMEmbeddings(base_url=base_url, api_key=api_key, model=model)
        
        try:
            test_embedding = embeddings.embed_query("test")
            logger.info(f"  Connection successful! Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to vLLM embedding service at {base_url}: {e}\n"
                "Make sure the vLLM embedding service is running:\n"
                "  docker compose --env-file .env -f opensource_llms/docker-compose.yml up\n"
                "Or set EMBED_BASE_URL in your .env file to point to your vLLM server."
            )
        
        return embeddings
    
    else:
        raise ValueError(f"Unknown embedding backend: {backend}. Use 'openai' or 'vllm'.")


def count_pdf_files(directory: str) -> tuple:
    """Count PDF files in directory and return stats."""
    pdf_files = []
    total_size = 0
    
    for f in os.listdir(directory):
        if f.lower().endswith('.pdf'):
            path = os.path.join(directory, f)
            size = os.path.getsize(path)
            pdf_files.append((f, size))
            total_size += size
    
    return pdf_files, total_size


def main():
    parser = argparse.ArgumentParser(
        description="Create PDF vectorstore for LLM-Lasso RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index with OpenAI embeddings
    python scripts/index_pdf_vectorstore.py \\
        --pdf-directory ./sample_pdfs \\
        --embedding-backend openai

    # Index with vLLM embeddings (requires running vLLM service)
    python scripts/index_pdf_vectorstore.py \\
        --pdf-directory ./sample_pdfs \\
        --embedding-backend vllm

    # Force re-index with specific chunk settings (skip chunks shorter than 50 chars)
    python scripts/index_pdf_vectorstore.py \\
        --pdf-directory ./sample_pdfs \\
        --persist-directory ./my_vectorstore \\
        --chunk-size 1500 \\
        --chunk-overlap 300 \\
        --min-chunk-length 50 \\
        --embedding-backend vllm

    # Index with duplicate detection (appends to existing vectorstore)
    python scripts/index_pdf_vectorstore.py \\
        --pdf-directory ./sample_pdfs \\
        --persist-directory ./my_vectorstore \\
        --no-clean \\
        --check-duplicates \\
        --duplicate-threshold 0.9 \\
        --embedding-backend vllm
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--pdf-directory",
        type=str,
        required=True,
        help="Path to directory containing PDF files to index"
    )
    
    # Optional arguments
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="pdf_vectorstore",
        help="Path to directory where ChromaDB will be persisted (default: pdf_vectorstore)"
    )
    
    parser.add_argument(
        "--embedding-backend",
        type=str,
        choices=["openai", "vllm"],
        default="openai",
        help="Embedding backend to use: 'openai' or 'vllm' (default: openai)"
    )
    
    parser.add_argument(
        "--collection-name",
        type=str,
        default="pdf_documents",
        help="Name for the ChromaDB collection (default: pdf_documents)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of each text chunk in characters (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks in characters (default: 200)"
    )
    
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=50,
        help="Minimum character length for a chunk to be indexed. Chunks shorter than this "
             "(e.g. section headers like 'SUICIDAL BEHAVIOR') are skipped to reduce retrieval noise. "
             "Use 0 to disable (default: 50)"
    )
    
    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=None,
        metavar="N",
        help="Minimum word count for a chunk. Chunks with fewer words are skipped. "
             "Use 0 or omit to disable (default: disabled)"
    )
    
    parser.add_argument(
        "--no-normalize-newlines",
        action="store_true",
        help="Disable newline normalization. By default, short lines (e.g. section headers) "
             "are merged with the next paragraph so they are not standalone chunks; "
             "this preserves subtitles while reducing noise."
    )
    
    parser.add_argument(
        "--max-short-line-chars",
        type=int,
        default=80,
        help="Lines shorter than this (chars) are merged with the next line when "
             "normalize-newlines is enabled (default: 80)"
    )
    
    parser.add_argument(
        "--no-page-chunks",
        action="store_true",
        help="Disable page-by-page extraction (extract full document at once)"
    )
    
    parser.add_argument(
        "--no-filter-references",
        action="store_true",
        help="Disable filtering of reference/bibliography sections"
    )
    
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean existing vectorstore (append to existing instead)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )

    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Enable duplicate document detection using similarity search"
    )

    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold (0.0-1.0) for duplicate detection. "
             "Higher values are more conservative (default: 0.95)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("=" * 70)
    logger.info("PDF Vectorstore Indexing - LLM-Lasso")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Validate PDF directory
    pdf_directory = os.path.abspath(args.pdf_directory)
    if not os.path.exists(pdf_directory):
        logger.error(f"PDF directory not found: {pdf_directory}")
        sys.exit(1)
    
    if not os.path.isdir(pdf_directory):
        logger.error(f"PDF path is not a directory: {pdf_directory}")
        sys.exit(1)
    
    # Count PDF files
    pdf_files, total_size = count_pdf_files(pdf_directory)
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_directory}")
        sys.exit(1)
    
    logger.info(f"PDF Directory: {pdf_directory}")
    logger.info(f"  Found {len(pdf_files)} PDF files ({total_size / 1024 / 1024:.2f} MB total)")
    for filename, size in pdf_files[:10]:  # Show first 10
        logger.debug(f"    - {filename} ({size / 1024:.1f} KB)")
    if len(pdf_files) > 10:
        logger.debug(f"    ... and {len(pdf_files) - 10} more files")
    
    # Set up persist directory
    persist_directory = os.path.abspath(args.persist_directory)
    logger.info(f"Persist Directory: {persist_directory}")
    
    if os.path.exists(persist_directory):
        if args.no_clean:
            logger.warning("Existing vectorstore found - will append (--no-clean specified)")
        else:
            logger.info("Existing vectorstore will be cleaned and recreated")
    
    # Log configuration
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Embedding Backend: {args.embedding_backend}")
    logger.info(f"  Collection Name: {args.collection_name}")
    logger.info(f"  Chunk Size: {args.chunk_size}")
    logger.info(f"  Chunk Overlap: {args.chunk_overlap}")
    logger.info(f"  Min Chunk Length: {args.min_chunk_length} (chars)")
    logger.info(f"  Min Chunk Words: {args.min_chunk_words or 'disabled'}")
    logger.info(f"  Normalize Newlines: {not args.no_normalize_newlines}")
    logger.info(f"  Max Short Line Chars: {args.max_short_line_chars}")
    logger.info(f"  Page Chunks: {not args.no_page_chunks}")
    logger.info(f"  Filter References: {not args.no_filter_references}")
    logger.info(f"  Clean Existing: {not args.no_clean}")
    logger.info(f"  Check Duplicates: {args.check_duplicates}")
    if args.check_duplicates:
        logger.info(f"  Duplicate Threshold: {args.duplicate_threshold}")
    logger.info("")
    
    # Get embedding model
    try:
        logger.info("Initializing embedding model...")
        embeddings = get_embedding_model(args.embedding_backend, logger)
        logger.info("Embedding model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        sys.exit(1)
    
    # Create vectorstore
    from llm_lasso.llm_penalty.rag.pdf_vectorstore import create_pdf_vectorstore
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("Starting vectorstore creation...")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    try:
        vectorstore = create_pdf_vectorstore(
            pdf_directory=pdf_directory,
            persist_directory=persist_directory,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            page_chunks=not args.no_page_chunks,
            embedding_model=embeddings,
            collection_name=args.collection_name,
            filter_references=not args.no_filter_references,
            clean_existing=not args.no_clean,
            check_duplicates=args.check_duplicates,
            duplicate_threshold=args.duplicate_threshold,
            min_chunk_length=args.min_chunk_length,
            min_chunk_words=args.min_chunk_words,
            normalize_newlines=not args.no_normalize_newlines,
            max_short_line_chars=args.max_short_line_chars
        )
        
        elapsed_time = time.time() - start_time
        
        # Get stats
        collection = vectorstore._collection
        doc_count = collection.count()
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"  Documents indexed: {doc_count}")
        logger.info(f"  Collection name: {args.collection_name}")
        logger.info(f"  Persist directory: {persist_directory}")
        logger.info(f"  Embedding backend: {args.embedding_backend}")
        
        # Get unique sources
        if doc_count > 0:
            all_meta = collection.get(include=["metadatas"])
            unique_sources = set()
            for meta in all_meta.get('metadatas', []):
                if meta and 'filename' in meta:
                    unique_sources.add(meta['filename'])
            logger.info(f"  Unique source files: {len(unique_sources)}")
        
        logger.info("")
        logger.info("You can now use this vectorstore with the LLM-Lasso pipeline:")
        logger.info(f"  python scripts/run_pbd_llm_lasso.py \\")
        logger.info(f"      --pdf_rag \\")
        logger.info(f"      --pdf_persist_directory {persist_directory} \\")
        logger.info(f"      --pdf_collection_name {args.collection_name} \\")
        logger.info(f"      --llm-backend {args.embedding_backend} \\")
        logger.info(f"      ...")
        
        # IMPORTANT: Remind about matching backends
        logger.info("")
        logger.info("IMPORTANT: When running the pipeline, use the SAME embedding backend:")
        logger.info(f"           --llm-backend {args.embedding_backend}")
        
    except Exception as e:
        logger.error(f"Vectorstore creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("")
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
