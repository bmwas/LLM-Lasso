"""
This script provides functionality to process PDF documents for RAG (Retrieval-Augmented Generation).
It extracts text from scientific papers using pymupdf4llm and prepares them for vectorstore ingestion.
"""

import os
from pathlib import Path
from typing import Optional
import pymupdf4llm
import pymupdf
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str, page_chunks: bool = True) -> list[dict]:
    """
    Extract text from a PDF file using pymupdf4llm.
    
    Args:
        pdf_path: Path to the PDF file.
        page_chunks: If True, return text chunked by page. If False, return full document text.
    
    Returns:
        List of dictionaries containing extracted text and metadata.
        Each dict has keys: 'content', 'metadata' (with 'source', 'page', 'total_pages')
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Open the PDF to get page count and title
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    
    # Try to extract title from metadata or first page
    title = doc.metadata.get("title", "") if doc.metadata else ""
    if not title:
        title = pdf_path.stem  # Use filename without extension as fallback
    
    doc.close()
    
    results = []
    
    if page_chunks:
        # Extract text page by page for better granularity
        for page_num in range(total_pages):
            try:
                # Extract markdown text for this specific page
                md_text = pymupdf4llm.to_markdown(
                    str(pdf_path),
                    pages=[page_num],
                    show_progress=False
                )
                
                if md_text.strip():
                    results.append({
                        "content": md_text.strip(),
                        "metadata": {
                            "source": str(pdf_path),
                            "filename": pdf_path.name,
                            "title": title,
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "section": f"page_{page_num + 1}"
                        }
                    })
            except Exception as e:
                print(f"Warning: Error extracting page {page_num + 1} from {pdf_path}: {e}")
                continue
    else:
        # Extract entire document at once
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path), show_progress=False)
            if md_text.strip():
                results.append({
                    "content": md_text.strip(),
                    "metadata": {
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "title": title,
                        "page": 0,  # 0 indicates full document
                        "total_pages": total_pages,
                        "section": "full_document"
                    }
                })
        except Exception as e:
            print(f"Warning: Error extracting text from {pdf_path}: {e}")
    
    return results


def load_pdfs_from_directory(
    pdf_directory: str,
    page_chunks: bool = True,
    recursive: bool = False
) -> list[dict]:
    """
    Load and extract text from all PDF files in a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        page_chunks: If True, chunk by page. If False, full document per PDF.
        recursive: If True, search subdirectories recursively.
    
    Returns:
        List of dictionaries containing extracted text and metadata from all PDFs.
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    if not pdf_dir.is_dir():
        raise ValueError(f"Path is not a directory: {pdf_directory}")
    
    # Find all PDF files
    if recursive:
        pdf_files = list(pdf_dir.rglob("*.pdf"))
    else:
        pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {pdf_directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s) in {pdf_directory}")
    
    all_documents = []
    for pdf_path in tqdm(pdf_files, desc="Extracting text from PDFs"):
        try:
            docs = extract_text_from_pdf(str(pdf_path), page_chunks=page_chunks)
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    print(f"Extracted {len(all_documents)} document chunks from {len(pdf_files)} PDF(s)")
    return all_documents


def get_pdf_retrieval_context(
    batch_features: list[str],
    category: str,
    retriever,
    num_docs: int = 3
) -> str:
    """
    Retrieve relevant context from PDF vectorstore for a batch of features.
    
    Args:
        batch_features: List of feature names (e.g., gene names) to search for.
        category: The category/domain context (e.g., cancer type).
        retriever: Vector store retriever object.
        num_docs: Number of documents to retrieve per query.
    
    Returns:
        Concatenated string of relevant document contents.
    """
    all_docs = []
    seen_contents = set()
    
    for feature in batch_features:
        query = f"Information about {feature} related to {category}"
        try:
            docs = retriever.get_relevant_documents(query)[:num_docs]
            for doc in docs:
                # Deduplicate by content
                content_hash = hash(doc.page_content[:500])  # Use first 500 chars for hash
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)
        except Exception as e:
            print(f"Warning: Error retrieving docs for {feature}: {e}")
            continue
    
    if not all_docs:
        return ""
    
    # Format the context
    context_parts = []
    for i, doc in enumerate(all_docs):
        source = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Document {i+1} - {source}, Page {page}]\n{doc.page_content}"
        )
    
    return "\n\n".join(context_parts)

