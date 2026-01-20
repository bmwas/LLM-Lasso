"""
Populate a Chroma-based vector store with PDF documents from scientific papers.
Then, use a hybrid chain to answer user queries by retrieving relevant documents 
from the vector store and combining them with the LLM's general knowledge.

This script is analogous to interactive_omim_RAG.py but for local PDF documents.
"""

import os
import sys
import warnings

# Add parent directory to path for imports (use absolute path based on script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))  # Load from project root .env

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage

# Import PDF RAG components
from llm_lasso.llm_penalty.rag.pdf_RAG_process import load_pdfs_from_directory
from llm_lasso.llm_penalty.rag.pdf_vectorstore import (
    get_or_create_pdf_vectorstore,
    create_pdf_vectorstore,
    load_pdf_vectorstore
)
from llm_lasso.utils.chunking import chunk_pdf_documents

warnings.filterwarnings("ignore")  # Suppress warnings

# Try to import constants, fall back to environment variables
try:
    import constants
    OPENAI_API_KEY = getattr(constants, 'OPENAI_API', None)
except (ImportError, ModuleNotFoundError):
    constants = None
    OPENAI_API_KEY = None

# Set OpenAI API key from constants or environment variable (already loaded from .env)
# Check both OPENAI_API_KEY (standard) and OPENAI_API (used in this project's constants)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
elif "OPENAI_API_KEY" in os.environ:
    pass  # Already set
elif "OPENAI_API" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]
else:
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

# ==================== Configuration ====================

# Enable persistence to save the database to disk
PERSIST = True

# File paths - use constants if available, otherwise use defaults (relative to project root)
_default_pdf_data = os.path.join(PROJECT_ROOT, 'sample_pdfs')
_default_pdf_persist = os.path.join(PROJECT_ROOT, 'pdf_vectorstore')

if constants is not None:
    PDF_DATA_DIRECTORY = getattr(constants, 'PDF_DATA_DIRECTORY', _default_pdf_data)
    PDF_PERSIST_DIRECTORY = getattr(constants, 'PDF_PERSIST_DIRECTORY', _default_pdf_persist)
else:
    PDF_DATA_DIRECTORY = _default_pdf_data
    PDF_PERSIST_DIRECTORY = _default_pdf_persist

# Convert to absolute paths if they're relative
if not os.path.isabs(PDF_DATA_DIRECTORY):
    PDF_DATA_DIRECTORY = os.path.join(PROJECT_ROOT, PDF_DATA_DIRECTORY)
if not os.path.isabs(PDF_PERSIST_DIRECTORY):
    PDF_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, PDF_PERSIST_DIRECTORY)

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==================== Step 1: Create or Load Vector Store ====================

print("=" * 60)
print("PDF RAG Interactive System")
print("=" * 60)

# Check if PDF directory exists
if not os.path.exists(PDF_DATA_DIRECTORY):
    print(f"\nWarning: PDF directory not found at {PDF_DATA_DIRECTORY}")
    print("Please ensure sample_pdfs/ directory exists with PDF files.")
    alt_path = os.path.join(PROJECT_ROOT, 'sample_pdfs')
    if os.path.exists(alt_path):
        PDF_DATA_DIRECTORY = alt_path
        print(f"Found alternate path: {PDF_DATA_DIRECTORY}")
    else:
        print("No PDF directory found. Exiting.")
        sys.exit(1)

# Initialize embeddings
print("\nInitializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings()

# Create or load the PDF-based vector store
print(f"\nPDF Directory: {PDF_DATA_DIRECTORY}")
print(f"Persist Directory: {PDF_PERSIST_DIRECTORY}")

try:
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
    print("\nVector store ready!")
except Exception as e:
    print(f"\nError creating/loading vector store: {e}")
    print("\nAttempting to create new vector store...")
    
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

# ==================== Step 2: Initialize Retriever and LLM ====================

# Initialize retriever with configurable k
K_DOCUMENTS = 5  # Number of documents to retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCUMENTS})

# Initialize LLM
print("\nInitializing LLM (gpt-4o)...")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ==================== Step 3: Define Hybrid Chain ====================

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
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)

    if retrieved_docs:
        # Combine retrieved documents into context with source information
        context_parts = []
        for doc in retrieved_docs:
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context_parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        context = context[:max_length]  # Ensure the context is within LLM limits

        # Create a prompt with retrieved context
        messages = [
            SystemMessage(content="You are an expert assistant with knowledge of scientific literature and research papers."),
            HumanMessage(content=f"Using the following context from scientific papers, provide the most accurate and relevant answer to the question. "
                "Prioritize the provided context, but if the context does not contain enough information to fully address the question, "
                "use your best general knowledge to complete the answer. Always cite the source document when using information from the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}")
        ]
        response = llm(messages)
        final_response = f"📚 Document-Grounded Answer:\n{response.content}"
        
        # Show sources
        sources = list(set([doc.metadata.get('filename', 'Unknown') for doc in retrieved_docs]))
        final_response += f"\n\n📄 Sources consulted: {', '.join(sources)}"
    else:
        # Fallback to GPT's general knowledge
        messages = [
            SystemMessage(content="You are an expert assistant with knowledge of scientific literature and research papers."),
            HumanMessage(content=f"Answer the following question based on your general knowledge:\n\nQuestion: {query}")
        ]
        response = llm(messages)
        final_response = f"🧠 General Knowledge Answer (no relevant documents found):\n{response.content}"

    return final_response


def show_collection_stats():
    """Display statistics about the vector store collection."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"\n📊 Collection Statistics:")
        print(f"   - Total document chunks: {count}")
        print(f"   - Retrieval k value: {K_DOCUMENTS}")
    except Exception as e:
        print(f"   Could not get collection stats: {e}")


# ==================== Step 4: Interactive Chat Loop ====================

print("\n" + "=" * 60)
print("Interactive PDF RAG Chat")
print("=" * 60)
show_collection_stats()
print("\nCommands:")
print("  - Type your question to query the PDF documents")
print("  - 'stats' - Show collection statistics")
print("  - 'rebuild' - Rebuild the vector store from PDFs")
print("  - 'quit', 'q', or 'exit' - End the chat")
print("=" * 60)

chat_history = []

while True:
    try:
        query = input("\n🔍 Prompt: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye!")
        break
    
    if not query:
        continue
        
    if query.lower() in ['quit', 'q', 'exit']:
        print("Goodbye!")
        break
    
    if query.lower() == 'stats':
        show_collection_stats()
        continue
    
    if query.lower() == 'rebuild':
        print("\nRebuilding vector store...")
        try:
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
            print("Vector store rebuilt successfully!")
            show_collection_stats()
        except Exception as e:
            print(f"Error rebuilding: {e}")
        continue

    # Get the hybrid response
    print("\n⏳ Searching documents and generating response...")
    answer = hybrid_chain(query, retriever, llm, chat_history)
    print(f"\n{answer}")

    # Update chat history
    chat_history.append((query, answer))

