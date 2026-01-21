"""
View ALL documents in a PDF vectorstore with their full content.

This script retrieves and displays every document chunk stored in the vectorstore,
organized by source file and page.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_lasso.llm_penalty.rag import load_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings
from collections import defaultdict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from .env")
    else:
        print(f"⚠️  .env file not found at {env_file}")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Try to import constants, but handle gracefully if _my_constants.py doesn't exist
try:
    import constants
    HAS_CONSTANTS = True
except (ImportError, ModuleNotFoundError):
    HAS_CONSTANTS = False
    print("⚠️  Note: _my_constants.py not found. Using environment variables and defaults.")

def view_all_documents(
    persist_directory=None,
    collection_name="scientific_papers",
    show_content=True,
    max_content_length=500,
    save_to_file=None
):
    """
    View all documents in the PDF vectorstore.
    
    Args:
        persist_directory: Path to vectorstore directory (default: from constants or 'pdf_vectorstore')
        collection_name: Name of the collection (default: "scientific_papers")
        show_content: Whether to show document content (default: True)
        max_content_length: Max characters to show per document (default: 500)
        save_to_file: Optional path to save output to file
    """
    # Set up API key - try constants first, then environment variable (already loaded from .env)
    if HAS_CONSTANTS:
        api_key = getattr(constants, 'OPENAI_API', None)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # Check for OPENAI_API_KEY or OPENAI_API in environment (may have been loaded from .env)
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        # Try OPENAI_API as alternative name
        if "OPENAI_API" in os.environ and os.environ["OPENAI_API"]:
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]
        else:
            print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
            print("   Options:")
            print("   1. Add OPENAI_API_KEY=your-key to .env file")
            print("   2. Or create _my_constants.py with OPENAI_API = 'your-key'")
            print("   3. Or set: export OPENAI_API_KEY='your-key'")
    
    # Use default directory if not provided
    if persist_directory is None:
        if HAS_CONSTANTS:
            persist_directory = getattr(constants, 'PDF_PERSIST_DIRECTORY', 'pdf_vectorstore')
        else:
            persist_directory = 'pdf_vectorstore'
    
    print("=" * 80)
    print("📚 Viewing ALL Documents in PDF Vectorstore")
    print("=" * 80)
    print(f"Persist Directory: {persist_directory}")
    print(f"Collection Name: {collection_name}")
    print("=" * 80)
    
    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        print("\nMake sure OPENAI_API_KEY is set:")
        print("  1. Create _my_constants.py with OPENAI_API = 'your-key'")
        print("  2. Or set: export OPENAI_API_KEY='your-key'")
        return
    
    # Load vectorstore
    try:
        vectorstore = load_pdf_vectorstore(
            persist_directory=persist_directory,
            embedding_model=embeddings,
            collection_name=collection_name
        )
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        print(f"\nTried to load from: {persist_directory}")
        print(f"Collection name: {collection_name}")
        print("\nMake sure the vectorstore exists. Create it with:")
        print("  python playground/interactive_pdf_RAG.py")
        print("\nOr specify a different path with --path")
        return
    
    # Get collection
    collection = vectorstore._collection
    total_count = collection.count()
    
    print(f"\n📊 Total document chunks: {total_count}")
    
    if total_count == 0:
        print("⚠️  Vectorstore is empty!")
        return
    
    # Get ALL documents
    print("\n📥 Retrieving all documents...")
    all_docs = collection.get(include=["metadatas", "documents"])
    # IDs are always returned, no need to include them
    
    # Organize by source file
    documents_by_file = defaultdict(list)
    
    for i, (doc_id, doc_text, metadata) in enumerate(zip(
        all_docs.get('ids', []),
        all_docs.get('documents', []),
        all_docs.get('metadatas', [])
    )):
        filename = metadata.get('filename', 'Unknown') if metadata else 'Unknown'
        page = metadata.get('page', 'N/A') if metadata else 'N/A'
        title = metadata.get('title', filename) if metadata else filename
        
        documents_by_file[filename].append({
            'id': doc_id,
            'content': doc_text,
            'page': page,
            'title': title,
            'metadata': metadata or {}
        })
    
    # Display organized by file
    output_lines = []
    
    output_lines.append(f"\n📁 Found {len(documents_by_file)} unique source file(s):\n")
    
    for filename in sorted(documents_by_file.keys()):
        file_docs = documents_by_file[filename]
        pages = sorted(set([doc['page'] for doc in file_docs]), 
                      key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0)
        
        output_lines.append("=" * 80)
        output_lines.append(f"📄 File: {filename}")
        output_lines.append(f"   Title: {file_docs[0]['title']}")
        output_lines.append(f"   Total chunks: {len(file_docs)}")
        output_lines.append(f"   Pages: {min(pages) if pages else 'N/A'} - {max(pages) if pages else 'N/A'} ({len(set(pages))} unique pages)")
        output_lines.append("-" * 80)
        
        # Show each document chunk
        for idx, doc in enumerate(file_docs, 1):
            output_lines.append(f"\n  Chunk #{idx} (ID: {doc['id'][:20]}...)")
            output_lines.append(f"  Page: {doc['page']}")
            
            if show_content:
                content_preview = doc['content']
                if len(content_preview) > max_content_length:
                    content_preview = content_preview[:max_content_length] + "... [truncated]"
                output_lines.append(f"  Content:\n  {content_preview}")
            
            # Show metadata if available
            if doc['metadata']:
                metadata_str = ", ".join([f"{k}={v}" for k, v in doc['metadata'].items() 
                                         if k not in ['filename', 'page', 'title']])
                if metadata_str:
                    output_lines.append(f"  Metadata: {metadata_str}")
            
            output_lines.append("")
        
        output_lines.append("")
    
    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file if requested
    if save_to_file:
        with open(save_to_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALL DOCUMENTS IN PDF VECTORSTORE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Persist Directory: {persist_directory}\n")
            f.write(f"Collection Name: {collection_name}\n")
            f.write(f"Total chunks: {total_count}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output_text)
        print(f"\n💾 Output saved to: {save_to_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    print(f"Total chunks: {total_count}")
    print(f"Unique files: {len(documents_by_file)}")
    for filename in sorted(documents_by_file.keys()):
        chunks = len(documents_by_file[filename])
        pages = len(set([doc['page'] for doc in documents_by_file[filename]]))
        print(f"  • {filename}: {chunks} chunks across {pages} pages")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="View all documents in PDF vectorstore"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to vectorstore directory (default: from constants)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="scientific_papers",
        help="Collection name (default: scientific_papers)"
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Don't show document content, only metadata"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=500,
        help="Maximum characters to show per document (default: 500)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save output to file"
    )
    
    args = parser.parse_args()
    
    view_all_documents(
        persist_directory=args.path,
        collection_name=args.collection,
        show_content=not args.no_content,
        max_content_length=args.max_length,
        save_to_file=args.save
    )
