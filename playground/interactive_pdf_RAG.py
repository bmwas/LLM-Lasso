"""
Populate a Chroma-based vector store with PDF documents from scientific papers.
Then, use a hybrid chain to answer user queries by retrieving relevant documents 
from the vector store and combining them with the LLM's general knowledge.

This script is analogous to interactive_omim_RAG.py but for local PDF documents.

Extensive logging is available for debugging:
  - Set LOG_LEVEL environment variable (DEBUG, INFO, WARNING, ERROR)
  - Use 'debug on/off' command in interactive mode
  - Use 'logs' command to view recent log entries
  - Use 'inspect' command to view detailed collection info
"""

import os
import sys
import warnings
import logging
import time
import traceback
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque

# ==================== Logging Setup ====================

# Custom log handler that stores recent logs in memory for inspection
class MemoryLogHandler(logging.Handler):
    """Custom handler that stores recent log records in memory."""
    def __init__(self, capacity: int = 500):
        super().__init__()
        self.log_buffer = deque(maxlen=capacity)
    
    def emit(self, record):
        self.log_buffer.append(self.format(record))
    
    def get_logs(self, n: int = 50) -> List[str]:
        """Get the last n log entries."""
        return list(self.log_buffer)[-n:]
    
    def clear(self):
        """Clear the log buffer."""
        self.log_buffer.clear()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def format(self, record):
        # Add color to level name
        color = self.COLORS.get(record.levelname, '')
        record.colored_levelname = f"{color}{record.levelname:8}{self.RESET}"
        
        # Format timestamp
        record.short_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Format location
        record.location = f"{record.filename}:{record.lineno}"
        
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging with console and optional file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("pdf_rag")
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    logger.handlers = []  # Clear existing handlers
    
    # Console formatter (colored)
    console_format = "%(short_time)s │ %(colored_levelname)s │ %(location)-25s │ %(message)s"
    console_formatter = ColoredFormatter(console_format)
    
    # File formatter (no colors, more detail)
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Memory formatter (simpler)
    memory_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    memory_formatter = logging.Formatter(memory_format, datefmt='%H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Memory handler (always capture everything)
    global memory_handler
    memory_handler = MemoryLogHandler(capacity=1000)
    memory_handler.setLevel(logging.DEBUG)
    memory_handler.setFormatter(memory_formatter)
    logger.addHandler(memory_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.debug(f"File logging enabled: {log_file}")
    
    return logger


def set_log_level(level: str):
    """Dynamically change the console log level."""
    logger = logging.getLogger("pdf_rag")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, MemoryLogHandler):
            handler.setLevel(getattr(logging, level.upper()))
            logger.info(f"Console log level changed to: {level.upper()}")


def mask_api_key(key: str) -> str:
    """Mask API key for safe logging."""
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


# Initialize logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("LOG_FILE", None)
logger = setup_logging(LOG_LEVEL, LOG_FILE)
memory_handler = None  # Will be set by setup_logging

# ==================== Script Initialization ====================

logger.info("=" * 70)
logger.info("PDF RAG Interactive System - Starting")
logger.info("=" * 70)

# Log system information
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Python executable: {sys.executable}")
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Script file: {__file__}")
logger.debug(f"Process ID: {os.getpid()}")
logger.debug(f"Log level: {LOG_LEVEL}")

# Add parent directory to path for imports (use absolute path based on script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
logger.debug(f"SCRIPT_DIR: {SCRIPT_DIR}")
logger.debug(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.debug(f"sys.path updated with PROJECT_ROOT")

# Load environment variables from .env file
logger.info("Loading environment variables...")
from dotenv import load_dotenv
env_file = os.path.join(PROJECT_ROOT, '.env')
env_loaded = load_dotenv(env_file)
logger.debug(f".env file path: {env_file}")
logger.debug(f".env file exists: {os.path.exists(env_file)}")
logger.debug(f".env loaded successfully: {env_loaded}")

# Log which env vars are available (masked)
env_vars_of_interest = ['OPENAI_API_KEY', 'OPENAI_API', 'OPENAI_ORG_ID', 'LOG_LEVEL', 'LOG_FILE']
for var in env_vars_of_interest:
    value = os.environ.get(var)
    if value:
        if 'KEY' in var or 'API' in var:
            logger.debug(f"Environment variable {var}: {mask_api_key(value)}")
        else:
            logger.debug(f"Environment variable {var}: {value}")

# Import dependencies with timing
logger.info("Importing dependencies...")
import_start = time.time()

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    logger.debug("Imported langchain_openai (ChatOpenAI, OpenAIEmbeddings)")
except ImportError as e:
    logger.error(f"Failed to import langchain_openai: {e}")
    raise

try:
    from langchain_community.vectorstores import Chroma
    logger.debug("Imported langchain_community.vectorstores.Chroma")
except ImportError as e:
    logger.error(f"Failed to import Chroma: {e}")
    raise

try:
    from langchain.schema import HumanMessage, SystemMessage
    logger.debug("Imported langchain.schema (HumanMessage, SystemMessage)")
except ImportError as e:
    logger.error(f"Failed to import langchain.schema: {e}")
    raise

# Import PDF RAG components
try:
    from llm_lasso.llm_penalty.rag.pdf_RAG_process import load_pdfs_from_directory
    logger.debug("Imported load_pdfs_from_directory from pdf_RAG_process")
except ImportError as e:
    logger.error(f"Failed to import pdf_RAG_process: {e}")
    logger.error("Make sure llm_lasso is installed: pip install -e .")
    raise

try:
    from llm_lasso.llm_penalty.rag.pdf_vectorstore import (
        get_or_create_pdf_vectorstore,
        create_pdf_vectorstore,
        load_pdf_vectorstore
    )
    logger.debug("Imported vectorstore functions from pdf_vectorstore")
except ImportError as e:
    logger.error(f"Failed to import pdf_vectorstore: {e}")
    raise

try:
    from llm_lasso.utils.chunking import chunk_pdf_documents
    logger.debug("Imported chunk_pdf_documents from chunking")
except ImportError as e:
    logger.error(f"Failed to import chunking: {e}")
    raise

import_time = time.time() - import_start
logger.info(f"Dependencies imported successfully in {import_time:.2f}s")

warnings.filterwarnings("ignore")  # Suppress warnings
logger.debug("Warnings filter set to ignore")

# Try to import constants, fall back to environment variables
logger.info("Loading API configuration...")
try:
    import constants
    OPENAI_API_KEY = getattr(constants, 'OPENAI_API', None)
    logger.debug(f"Loaded constants module")
    logger.debug(f"OPENAI_API from constants: {'Found' if OPENAI_API_KEY else 'Not found'}")
    if OPENAI_API_KEY:
        logger.debug(f"API key from constants: {mask_api_key(OPENAI_API_KEY)}")
except (ImportError, ModuleNotFoundError) as e:
    constants = None
    OPENAI_API_KEY = None
    logger.debug(f"constants module not found: {e}")

# Set OpenAI API key from constants or environment variable (already loaded from .env)
# Check both OPENAI_API_KEY (standard) and OPENAI_API (used in this project's constants)
api_key_source = None
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    api_key_source = "constants module"
elif "OPENAI_API_KEY" in os.environ:
    api_key_source = "OPENAI_API_KEY environment variable"
elif "OPENAI_API" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]
    api_key_source = "OPENAI_API environment variable"
else:
    logger.critical("OpenAI API key not found!")
    logger.error("Checked: constants.OPENAI_API, env OPENAI_API_KEY, env OPENAI_API")
    print("\n" + "="*60)
    print("ERROR: OpenAI API key not found!")
    print("="*60)
    print("\nPlease either:")
    print("1. Create a '.env' file in the project root with:")
    print("   OPENAI_API_KEY=your-openai-api-key")
    print("   (or OPENAI_API=your-openai-api-key)")
    print("\n2. Or create '_my_constants.py' with:")
    print("   OPENAI_API = 'your-openai-api-key'")
    print("\n3. Or set the environment variable:")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print("="*60)
    sys.exit(1)

logger.info(f"API key loaded from: {api_key_source}")
logger.debug(f"API key (masked): {mask_api_key(os.environ.get('OPENAI_API_KEY', ''))}")

# ==================== Configuration ====================

logger.info("Loading configuration...")

# Enable persistence to save the database to disk
PERSIST = True
logger.debug(f"PERSIST: {PERSIST}")

# File paths - use constants if available, otherwise use defaults (relative to project root)
_default_pdf_data = os.path.join(PROJECT_ROOT, 'sample_pdfs')
_default_pdf_persist = os.path.join(PROJECT_ROOT, 'pdf_vectorstore')

if constants is not None:
    PDF_DATA_DIRECTORY = getattr(constants, 'PDF_DATA_DIRECTORY', _default_pdf_data)
    PDF_PERSIST_DIRECTORY = getattr(constants, 'PDF_PERSIST_DIRECTORY', _default_pdf_persist)
    logger.debug("Using paths from constants module")
else:
    PDF_DATA_DIRECTORY = _default_pdf_data
    PDF_PERSIST_DIRECTORY = _default_pdf_persist
    logger.debug("Using default paths")

# Convert to absolute paths if they're relative
if not os.path.isabs(PDF_DATA_DIRECTORY):
    PDF_DATA_DIRECTORY = os.path.join(PROJECT_ROOT, PDF_DATA_DIRECTORY)
if not os.path.isabs(PDF_PERSIST_DIRECTORY):
    PDF_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, PDF_PERSIST_DIRECTORY)

logger.info(f"PDF_DATA_DIRECTORY: {PDF_DATA_DIRECTORY}")
logger.info(f"PDF_PERSIST_DIRECTORY: {PDF_PERSIST_DIRECTORY}")

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
logger.debug(f"CHUNK_SIZE: {CHUNK_SIZE}")
logger.debug(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")

# ==================== Statistics Tracking ====================

class SessionStats:
    """Track session statistics for logging and debugging."""
    def __init__(self):
        self.start_time = datetime.now()
        self.queries_count = 0
        self.total_docs_retrieved = 0
        self.total_api_calls = 0
        self.total_tokens_used = 0  # If available
        self.total_retrieval_time = 0.0
        self.total_llm_time = 0.0
        self.errors_count = 0
        self.last_query_time = None
        self.last_retrieval_count = 0
    
    def log_query(self, query: str, docs_retrieved: int, retrieval_time: float, llm_time: float):
        self.queries_count += 1
        self.total_docs_retrieved += docs_retrieved
        self.total_api_calls += 1
        self.total_retrieval_time += retrieval_time
        self.total_llm_time += llm_time
        self.last_query_time = datetime.now()
        self.last_retrieval_count = docs_retrieved
    
    def log_error(self):
        self.errors_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "session_start": self.start_time.isoformat(),
            "uptime_seconds": uptime,
            "queries_count": self.queries_count,
            "total_docs_retrieved": self.total_docs_retrieved,
            "avg_docs_per_query": self.total_docs_retrieved / max(1, self.queries_count),
            "total_api_calls": self.total_api_calls,
            "total_retrieval_time": self.total_retrieval_time,
            "total_llm_time": self.total_llm_time,
            "avg_retrieval_time": self.total_retrieval_time / max(1, self.queries_count),
            "avg_llm_time": self.total_llm_time / max(1, self.queries_count),
            "errors_count": self.errors_count,
        }

stats = SessionStats()

# ==================== Step 1: Create or Load Vector Store ====================

logger.info("=" * 50)
logger.info("Step 1: Initializing Vector Store")
logger.info("=" * 50)

print("=" * 60)
print("PDF RAG Interactive System")
print("=" * 60)

# Check if PDF directory exists
logger.debug(f"Checking PDF directory: {PDF_DATA_DIRECTORY}")
if not os.path.exists(PDF_DATA_DIRECTORY):
    logger.warning(f"PDF directory not found at {PDF_DATA_DIRECTORY}")
    print(f"\nWarning: PDF directory not found at {PDF_DATA_DIRECTORY}")
    print("Please ensure sample_pdfs/ directory exists with PDF files.")
    alt_path = os.path.join(PROJECT_ROOT, 'sample_pdfs')
    if os.path.exists(alt_path):
        PDF_DATA_DIRECTORY = alt_path
        logger.info(f"Found alternate path: {PDF_DATA_DIRECTORY}")
        print(f"Found alternate path: {PDF_DATA_DIRECTORY}")
    else:
        logger.critical("No PDF directory found. Exiting.")
        print("No PDF directory found. Exiting.")
        sys.exit(1)

# List PDF files in directory
logger.debug("Scanning PDF directory for files...")
pdf_files = [f for f in os.listdir(PDF_DATA_DIRECTORY) if f.lower().endswith('.pdf')]
logger.info(f"Found {len(pdf_files)} PDF files in directory")
for pdf_file in pdf_files:
    file_path = os.path.join(PDF_DATA_DIRECTORY, pdf_file)
    file_size = os.path.getsize(file_path)
    logger.debug(f"  - {pdf_file} ({file_size:,} bytes)")

# Check persist directory
logger.debug(f"Checking persist directory: {PDF_PERSIST_DIRECTORY}")
persist_exists = os.path.exists(PDF_PERSIST_DIRECTORY)
logger.debug(f"Persist directory exists: {persist_exists}")
if persist_exists:
    persist_contents = os.listdir(PDF_PERSIST_DIRECTORY)
    logger.debug(f"Persist directory contents: {persist_contents}")

# Initialize embeddings
logger.info("Initializing OpenAI embeddings...")
print("\nInitializing OpenAI embeddings...")
embeddings_start = time.time()
try:
    embeddings = OpenAIEmbeddings()
    embeddings_time = time.time() - embeddings_start
    logger.info(f"OpenAI embeddings initialized in {embeddings_time:.2f}s")
    logger.debug(f"Embeddings model: {embeddings.model}")
    logger.debug(f"Embeddings chunk size: {embeddings.chunk_size}")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    logger.error(traceback.format_exc())
    raise

# Create or load the PDF-based vector store
print(f"\nPDF Directory: {PDF_DATA_DIRECTORY}")
print(f"Persist Directory: {PDF_PERSIST_DIRECTORY}")

logger.info("Creating/loading ChromaDB vectorstore...")
vectorstore_start = time.time()

try:
    logger.debug("Calling get_or_create_pdf_vectorstore...")
    logger.debug(f"  pdf_directory: {PDF_DATA_DIRECTORY}")
    logger.debug(f"  persist_directory: {PDF_PERSIST_DIRECTORY}")
    logger.debug(f"  chunk_size: {CHUNK_SIZE}")
    logger.debug(f"  chunk_overlap: {CHUNK_OVERLAP}")
    logger.debug(f"  page_chunks: True")
    logger.debug(f"  collection_name: scientific_papers")
    logger.debug(f"  force_recreate: False")
    
    vectorstore = get_or_create_pdf_vectorstore(
        pdf_directory=PDF_DATA_DIRECTORY,
        persist_directory=PDF_PERSIST_DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        page_chunks=True,
        embedding_model=embeddings,
        collection_name="scientific_papers",
        force_recreate=False  # Set to True to rebuild the database
    )
    
    vectorstore_time = time.time() - vectorstore_start
    logger.info(f"Vector store ready in {vectorstore_time:.2f}s")
    print("\nVector store ready!")
    
    # Log detailed vectorstore info
    try:
        collection = vectorstore._collection
        doc_count = collection.count()
        collection_name = collection.name
        logger.info(f"ChromaDB Collection: '{collection_name}'")
        logger.info(f"Total document chunks indexed: {doc_count}")
        logger.debug(f"Collection metadata: {collection.metadata}")
        
        # Get sample of document metadata
        if doc_count > 0:
            sample = collection.peek(limit=5)
            logger.debug(f"Sample document IDs: {sample.get('ids', [])[:5]}")
            if sample.get('metadatas'):
                logger.debug(f"Sample metadata: {json.dumps(sample['metadatas'][:2], indent=2)}")
    except Exception as e:
        logger.warning(f"Could not get detailed collection info: {e}")

except Exception as e:
    logger.error(f"Error creating/loading vector store: {e}")
    logger.error(traceback.format_exc())
    print(f"\nError creating/loading vector store: {e}")
    print("\nAttempting to create new vector store...")
    
    logger.info("Attempting to force create new vectorstore...")
    
    # Force create a new one
    vectorstore = create_pdf_vectorstore(
        pdf_directory=PDF_DATA_DIRECTORY,
        persist_directory=PDF_PERSIST_DIRECTORY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        page_chunks=True,
        embedding_model=embeddings,
        collection_name="scientific_papers"
    )
    logger.info("New vectorstore created successfully")

# ==================== Step 2: Initialize Retriever and LLM ====================

logger.info("=" * 50)
logger.info("Step 2: Initializing Retriever and LLM")
logger.info("=" * 50)

# Initialize retriever with configurable k
K_DOCUMENTS = 5  # Number of documents to retrieve
logger.info(f"Initializing retriever with k={K_DOCUMENTS}")
retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCUMENTS})
logger.debug(f"Retriever search type: {retriever.search_type}")
logger.debug(f"Retriever search kwargs: {retriever.search_kwargs}")

# Initialize LLM
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0
logger.info(f"Initializing LLM: model={LLM_MODEL}, temperature={LLM_TEMPERATURE}")
print(f"\nInitializing LLM ({LLM_MODEL})...")

llm_start = time.time()
try:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    llm_init_time = time.time() - llm_start
    logger.info(f"LLM initialized in {llm_init_time:.2f}s")
    logger.debug(f"LLM model name: {llm.model_name}")
    logger.debug(f"LLM temperature: {llm.temperature}")
    logger.debug(f"LLM max tokens: {llm.max_tokens}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    logger.error(traceback.format_exc())
    raise

# ==================== Step 3: Define Hybrid Chain ====================

logger.info("=" * 50)
logger.info("Step 3: Setting up Hybrid Chain")
logger.info("=" * 50)

def hybrid_chain(query: str, retriever, llm, chat_history: list, max_length: int = 4000) -> str:
    """
    Hybrid chain combining RAG with fallback to pretrained knowledge.

    Parameters:
    - query: User query.
    - retriever: Retriever object for vector database.
    - llm: GPT model.
    - chat_history: List of previous interactions.
    - max_length: Maximum character length for retrieved context.

    Returns:
    - Answer string (text content only).
    """
    query_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:8]
    logger.info(f"[Query:{query_id}] Processing query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
    logger.debug(f"[Query:{query_id}] Full query: {query}")
    logger.debug(f"[Query:{query_id}] Query length: {len(query)} chars")
    logger.debug(f"[Query:{query_id}] Chat history length: {len(chat_history)} entries")
    logger.debug(f"[Query:{query_id}] Max context length: {max_length}")
    
    # Step 1: Retrieve relevant documents
    logger.info(f"[Query:{query_id}] Retrieving documents from vectorstore...")
    retrieval_start = time.time()
    
    try:
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"[Query:{query_id}] Retrieved {len(retrieved_docs)} documents in {retrieval_time:.3f}s")
    except Exception as e:
        logger.error(f"[Query:{query_id}] Retrieval failed: {e}")
        logger.error(traceback.format_exc())
        stats.log_error()
        raise
    
    # Log detailed document info
    for i, doc in enumerate(retrieved_docs):
        doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        logger.debug(f"[Query:{query_id}] Doc {i+1}/{len(retrieved_docs)}:")
        logger.debug(f"  - Hash: {doc_hash}")
        logger.debug(f"  - Source: {doc.metadata.get('filename', 'Unknown')}")
        logger.debug(f"  - Page: {doc.metadata.get('page', 'N/A')}")
        logger.debug(f"  - Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
        logger.debug(f"  - Content length: {len(doc.page_content)} chars")
        logger.debug(f"  - Content preview: {doc.page_content[:200]}...")
        logger.debug(f"  - Full metadata: {doc.metadata}")

    llm_start = time.time()
    
    if retrieved_docs:
        logger.info(f"[Query:{query_id}] Building context from {len(retrieved_docs)} documents")
        
        # Combine retrieved documents into context with source information
        context_parts = []
        for doc in retrieved_docs:
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        original_context_length = len(context)
        context = context[:max_length]  # Ensure the context is within LLM limits
        
        logger.debug(f"[Query:{query_id}] Context original length: {original_context_length} chars")
        logger.debug(f"[Query:{query_id}] Context truncated length: {len(context)} chars")
        logger.debug(f"[Query:{query_id}] Context truncated: {original_context_length > max_length}")

        # Create a prompt with retrieved context
        system_msg = "You are an expert assistant with knowledge of scientific literature and research papers."
        user_msg = (
            f"Using the following context from scientific papers, provide the most accurate and relevant answer to the question. "
            "Prioritize the provided context, but if the context does not contain enough information to fully address the question, "
            "use your best general knowledge to complete the answer. Always cite the source document when using information from the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        logger.debug(f"[Query:{query_id}] System message length: {len(system_msg)} chars")
        logger.debug(f"[Query:{query_id}] User message length: {len(user_msg)} chars")
        logger.debug(f"[Query:{query_id}] Total prompt length: {len(system_msg) + len(user_msg)} chars")
        
        logger.info(f"[Query:{query_id}] Calling LLM API (document-grounded)...")
        try:
            response = llm(messages)
            llm_time = time.time() - llm_start
            logger.info(f"[Query:{query_id}] LLM response received in {llm_time:.3f}s")
            logger.debug(f"[Query:{query_id}] Response length: {len(response.content)} chars")
            logger.debug(f"[Query:{query_id}] Response preview: {response.content[:200]}...")
            
            # Log token usage if available
            if hasattr(response, 'response_metadata'):
                logger.debug(f"[Query:{query_id}] Response metadata: {response.response_metadata}")
            if hasattr(response, 'usage_metadata'):
                logger.debug(f"[Query:{query_id}] Usage metadata: {response.usage_metadata}")
                
        except Exception as e:
            logger.error(f"[Query:{query_id}] LLM API call failed: {e}")
            logger.error(traceback.format_exc())
            stats.log_error()
            raise
            
        final_response = f"📚 Document-Grounded Answer:\n{response.content}"
        
        # Show sources
        sources = list(set([doc.metadata.get('filename', 'Unknown') for doc in retrieved_docs]))
        final_response += f"\n\n📄 Sources consulted: {', '.join(sources)}"
        logger.debug(f"[Query:{query_id}] Sources used: {sources}")
        
    else:
        logger.warning(f"[Query:{query_id}] No documents retrieved, falling back to general knowledge")
        
        # Fallback to GPT's general knowledge
        system_msg = "You are an expert assistant with knowledge of scientific literature and research papers."
        user_msg = f"Answer the following question based on your general knowledge:\n\nQuestion: {query}"
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        logger.info(f"[Query:{query_id}] Calling LLM API (general knowledge fallback)...")
        try:
            response = llm(messages)
            llm_time = time.time() - llm_start
            logger.info(f"[Query:{query_id}] LLM response received in {llm_time:.3f}s")
            logger.debug(f"[Query:{query_id}] Response length: {len(response.content)} chars")
        except Exception as e:
            logger.error(f"[Query:{query_id}] LLM API call failed: {e}")
            stats.log_error()
            raise
            
        final_response = f"🧠 General Knowledge Answer (no relevant documents found):\n{response.content}"

    # Update stats
    stats.log_query(query, len(retrieved_docs), retrieval_time, llm_time)
    logger.info(f"[Query:{query_id}] Query completed successfully")
    
    return final_response


def show_collection_stats():
    """Display statistics about the vector store collection."""
    logger.info("Fetching collection statistics...")
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"\n📊 Collection Statistics:")
        print(f"   - Collection name: {collection.name}")
        print(f"   - Total document chunks: {count}")
        print(f"   - Retrieval k value: {K_DOCUMENTS}")
        print(f"   - Persist directory: {PDF_PERSIST_DIRECTORY}")
        
        logger.info(f"Collection '{collection.name}' has {count} documents")
        logger.debug(f"Collection metadata: {collection.metadata}")
        
        # Get unique sources
        if count > 0:
            try:
                all_docs = collection.get(include=["metadatas"])
                unique_sources = set()
                for metadata in all_docs.get('metadatas', []):
                    if metadata and 'filename' in metadata:
                        unique_sources.add(metadata['filename'])
                print(f"   - Unique source files: {len(unique_sources)}")
                for source in sorted(unique_sources):
                    print(f"      • {source}")
                logger.debug(f"Unique sources: {unique_sources}")
            except Exception as e:
                logger.warning(f"Could not get source files: {e}")
                
    except Exception as e:
        logger.error(f"Could not get collection stats: {e}")
        print(f"   Could not get collection stats: {e}")


def show_detailed_inspection():
    """Show detailed inspection of the vectorstore contents."""
    logger.info("Running detailed vectorstore inspection...")
    print("\n" + "=" * 60)
    print("🔍 Detailed Vectorstore Inspection")
    print("=" * 60)
    
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"\n📁 Collection Information:")
        print(f"   Name: {collection.name}")
        print(f"   Total chunks: {count}")
        print(f"   Metadata: {collection.metadata}")
        
        logger.debug(f"Collection name: {collection.name}")
        logger.debug(f"Collection count: {count}")
        
        if count > 0:
            # Get all documents
            all_docs = collection.get(include=["metadatas", "documents"])
            
            # Analyze by source file
            source_stats = {}
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata:
                    filename = metadata.get('filename', 'Unknown')
                    page = metadata.get('page', 0)
                    if filename not in source_stats:
                        source_stats[filename] = {'chunks': 0, 'pages': set()}
                    source_stats[filename]['chunks'] += 1
                    source_stats[filename]['pages'].add(page)
            
            print(f"\n📄 Source Files ({len(source_stats)} files):")
            for filename, stats_data in sorted(source_stats.items()):
                pages = sorted(stats_data['pages'])
                page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0] if pages else "N/A")
                print(f"   • {filename}")
                print(f"     Chunks: {stats_data['chunks']}, Pages: {page_range}")
                logger.debug(f"File {filename}: {stats_data['chunks']} chunks, pages {page_range}")
            
            # Sample documents
            print(f"\n📝 Sample Documents (first 3):")
            sample = collection.peek(limit=3)
            for i, (doc_id, doc_text, metadata) in enumerate(zip(
                sample.get('ids', []),
                sample.get('documents', []),
                sample.get('metadatas', [])
            )):
                print(f"\n   [{i+1}] ID: {doc_id}")
                print(f"       Source: {metadata.get('filename', 'Unknown')}, Page: {metadata.get('page', 'N/A')}")
                print(f"       Preview: {doc_text[:150]}..." if doc_text else "       (no content)")
                logger.debug(f"Sample doc {i+1}: {doc_id}, metadata: {metadata}")
        
        # Session statistics
        session_summary = stats.get_summary()
        print(f"\n📈 Session Statistics:")
        print(f"   Uptime: {session_summary['uptime_seconds']:.1f}s")
        print(f"   Queries: {session_summary['queries_count']}")
        print(f"   Total docs retrieved: {session_summary['total_docs_retrieved']}")
        print(f"   Avg docs/query: {session_summary['avg_docs_per_query']:.1f}")
        print(f"   Avg retrieval time: {session_summary['avg_retrieval_time']:.3f}s")
        print(f"   Avg LLM time: {session_summary['avg_llm_time']:.3f}s")
        print(f"   Errors: {session_summary['errors_count']}")
        
        logger.info(f"Session stats: {json.dumps(session_summary, indent=2)}")
        
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        logger.error(traceback.format_exc())
        print(f"   Inspection failed: {e}")
    
    print("=" * 60)


def show_recent_logs(n: int = 30):
    """Display recent log entries."""
    global memory_handler
    if memory_handler:
        logs = memory_handler.get_logs(n)
        print(f"\n📋 Recent Logs (last {len(logs)} entries):")
        print("-" * 60)
        for log in logs:
            print(log)
        print("-" * 60)
    else:
        print("Log buffer not available")


# ==================== Step 4: Interactive Chat Loop ====================

logger.info("=" * 50)
logger.info("Step 4: Starting Interactive Chat Loop")
logger.info("=" * 50)

print("\n" + "=" * 60)
print("Interactive PDF RAG Chat")
print("=" * 60)
show_collection_stats()
print("\nCommands:")
print("  - Type your question to query the PDF documents")
print("  - 'stats' - Show collection statistics")
print("  - 'inspect' - Detailed vectorstore inspection")
print("  - 'logs [n]' - Show recent n log entries (default 30)")
print("  - 'debug on/off' - Toggle debug logging")
print("  - 'rebuild' - Rebuild the vector store from PDFs")
print("  - 'session' - Show session statistics")
print("  - 'quit', 'q', or 'exit' - End the chat")
print("=" * 60)

chat_history = []
logger.info("Chat loop started, waiting for user input...")

while True:
    try:
        query = input("\n🔍 Prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        logger.info("Chat terminated by user (EOF/Interrupt)")
        print("\n\nGoodbye!")
        break
    
    if not query:
        continue
    
    logger.debug(f"User input received: '{query}'")
        
    if query.lower() in ['quit', 'q', 'exit']:
        logger.info("Chat terminated by user (quit command)")
        logger.info(f"Final session stats: {json.dumps(stats.get_summary(), indent=2)}")
        print("Goodbye!")
        break
    
    if query.lower() == 'stats':
        show_collection_stats()
        continue
    
    if query.lower() == 'inspect':
        show_detailed_inspection()
        continue
    
    if query.lower().startswith('logs'):
        parts = query.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
        show_recent_logs(n)
        continue
    
    if query.lower() == 'debug on':
        set_log_level('DEBUG')
        print("🔧 Debug logging enabled")
        continue
    
    if query.lower() == 'debug off':
        set_log_level('INFO')
        print("🔧 Debug logging disabled")
        continue
    
    if query.lower() == 'session':
        summary = stats.get_summary()
        print("\n📈 Session Statistics:")
        print(json.dumps(summary, indent=2))
        continue
    
    if query.lower() == 'rebuild':
        logger.info("Rebuild command received")
        print("\nRebuilding vector store...")
        try:
            rebuild_start = time.time()
            vectorstore = create_pdf_vectorstore(
                pdf_directory=PDF_DATA_DIRECTORY,
                persist_directory=PDF_PERSIST_DIRECTORY,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                page_chunks=True,
                embedding_model=embeddings,
                collection_name="scientific_papers"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCUMENTS})
            rebuild_time = time.time() - rebuild_start
            logger.info(f"Vectorstore rebuilt in {rebuild_time:.2f}s")
            print(f"Vector store rebuilt successfully in {rebuild_time:.2f}s!")
            show_collection_stats()
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            logger.error(traceback.format_exc())
            stats.log_error()
            print(f"Error rebuilding: {e}")
        continue

    # Get the hybrid response
    logger.info("Processing user query...")
    print("\n⏳ Searching documents and generating response...")
    
    try:
        query_start = time.time()
        answer = hybrid_chain(query, retriever, llm, chat_history)
        total_time = time.time() - query_start
        print(f"\n{answer}")
        print(f"\n⏱️ Total time: {total_time:.2f}s")
        logger.info(f"Query completed in {total_time:.2f}s total")
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        logger.error(traceback.format_exc())
        stats.log_error()
        print(f"\n❌ Error processing query: {e}")
        print("Check 'logs' command for details")
        continue

    # Update chat history
    chat_history.append((query, answer))
    logger.debug(f"Chat history updated, now {len(chat_history)} entries")

logger.info("=" * 50)
logger.info("PDF RAG Interactive System - Shutdown")
logger.info(f"Final stats: {json.dumps(stats.get_summary())}")
logger.info("=" * 50)
