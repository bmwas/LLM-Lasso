"""
Visualize ChromaDB vectorstore collections using chromaviz.

This script helps you inspect what papers/documents are stored in your ChromaDB vectorstores.
"""

import chromadb
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, "..")

try:
    from chromaviz import visualize_collection
except ImportError:
    print("Error: chromaviz is not installed.")
    print("Install it with: pip install chromaviz")
    sys.exit(1)

# Try to import constants to get default paths
try:
    import constants
    PDF_PERSIST_DIRECTORY = getattr(constants, 'PDF_PERSIST_DIRECTORY', 'pdf_vectorstore')
    OMIM_PERSIST_DIRECTORY = getattr(constants, 'OMIM_PERSIST_DIRECTORY', None)
except ImportError:
    PDF_PERSIST_DIRECTORY = 'pdf_vectorstore'
    OMIM_PERSIST_DIRECTORY = None

def visualize_pdf_vectorstore(persist_directory=None, collection_name="scientific_papers"):
    """
    Visualize the PDF vectorstore collection.
    
    Args:
        persist_directory: Path to PDF vectorstore directory. If None, uses default.
        collection_name: Name of the collection. Default is "scientific_papers" 
                        (used by interactive_pdf_RAG.py) or "pdf_documents" 
                        (default in pdf_vectorstore.py)
    """
    if persist_directory is None:
        persist_directory = PDF_PERSIST_DIRECTORY
    
    # Convert to absolute path if relative
    if not os.path.isabs(persist_directory):
        persist_directory = os.path.abspath(persist_directory)
    
    if not os.path.exists(persist_directory):
        print(f"Error: PDF vectorstore directory not found at: {persist_directory}")
        print(f"\nAvailable options:")
        print(f"1. Create the vectorstore first using:")
        print(f"   python playground/interactive_pdf_RAG.py")
        print(f"2. Or specify a different path:")
        print(f"   visualize_pdf_vectorstore(persist_directory='/path/to/your/pdf_vectorstore')")
        return
    
    print(f"Loading PDF vectorstore from: {persist_directory}")
    print(f"Collection name: {collection_name}")
    print("-" * 60)
    
    # Point to your existing Chroma DB directory
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Try to get the collection (may fail if collection name is wrong)
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✓ Found collection: {collection_name}")
        print(f"  Total documents: {collection.count()}")
    except Exception as e:
        print(f"✗ Collection '{collection_name}' not found.")
        print(f"  Error: {e}")
        print(f"\nAvailable collections:")
        # List available collections
        try:
            collections = client.list_collections()
            if collections:
                for col in collections:
                    print(f"  - {col.name}")
                print(f"\nTry using one of the collection names above.")
            else:
                print("  (No collections found)")
        except Exception as list_error:
            print(f"  Could not list collections: {list_error}")
        return
    
    # Visualize the collection
    print("\nVisualizing collection...")
    visualize_collection(collection)


def visualize_omim_vectorstore(persist_directory=None):
    """
    Visualize the OMIM vectorstore collection.
    
    Args:
        persist_directory: Path to OMIM vectorstore directory. If None, tries to get from constants.
    """
    if persist_directory is None:
        if OMIM_PERSIST_DIRECTORY:
            persist_directory = OMIM_PERSIST_DIRECTORY
        else:
            print("Error: OMIM_PERSIST_DIRECTORY not set in constants.")
            print("Please specify the path manually:")
            print("  visualize_omim_vectorstore(persist_directory='/path/to/omim_vectorstore')")
            return
    
    # Convert to absolute path if relative
    if not os.path.isabs(persist_directory):
        persist_directory = os.path.abspath(persist_directory)
    
    if not os.path.exists(persist_directory):
        print(f"Error: OMIM vectorstore directory not found at: {persist_directory}")
        return
    
    print(f"Loading OMIM vectorstore from: {persist_directory}")
    print("-" * 60)
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    # List available collections
    try:
        collections = client.list_collections()
        if not collections:
            print("No collections found in OMIM vectorstore.")
            return
        
        print(f"Found {len(collections)} collection(s):")
        for col in collections:
            print(f"  - {col.name} ({col.count()} documents)")
        
        # Use the first collection (or default ChromaDB collection name)
        collection_name = collections[0].name if collections else "langchain"
        collection = client.get_collection(name=collection_name)
        
        print(f"\nVisualizing collection: {collection_name}")
        visualize_collection(collection)
    except Exception as e:
        print(f"Error accessing collections: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ChromaDB vectorstore collections")
    parser.add_argument(
        "--type",
        choices=["pdf", "omim"],
        default="pdf",
        help="Type of vectorstore to visualize (default: pdf)"
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
        default=None,
        help="Collection name (default: 'scientific_papers' for PDF, auto-detect for OMIM)"
    )
    
    args = parser.parse_args()
    
    if args.type == "pdf":
        collection_name = args.collection or "scientific_papers"
        visualize_pdf_vectorstore(persist_directory=args.path, collection_name=collection_name)
    elif args.type == "omim":
        visualize_omim_vectorstore(persist_directory=args.path)

