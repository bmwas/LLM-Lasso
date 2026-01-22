"""
This script provides functionality to process PDF documents for RAG (Retrieval-Augmented Generation).
It extracts text from scientific papers using pymupdf4llm and prepares them for vectorstore ingestion.

Includes optional reference section filtering to avoid indexing bibliographies.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pymupdf4llm
import pymupdf
from tqdm import tqdm

# Set up module logger
logger = logging.getLogger("pdf_rag.process")


def extract_text_from_pdf(
    pdf_path: str, 
    page_chunks: bool = True,
    filter_references: bool = True
) -> Tuple[list[dict], Dict[str, Any]]:
    """
    Extract text from a PDF file using pymupdf4llm.
    
    Args:
        pdf_path: Path to the PDF file.
        page_chunks: If True, return text chunked by page. If False, return full document text.
        filter_references: If True, filter out pages that appear to be reference sections.
    
    Returns:
        Tuple of:
        - List of dictionaries containing extracted text and metadata.
          Each dict has keys: 'content', 'metadata' (with 'source', 'page', 'total_pages')
        - Dictionary with filtering statistics (empty if filter_references=False)
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
    filter_stats = {
        'pages_extracted': 0,
        'pages_filtered': 0,
        'filtered_pages': []
    }
    
    # Import reference filter if needed
    if filter_references:
        try:
            from llm_lasso.utils.reference_filter import is_reference_page
        except ImportError:
            logger.warning("reference_filter module not available, skipping reference filtering")
            filter_references = False
    
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
                    # Check if this page is a reference section
                    if filter_references:
                        is_ref, reason = is_reference_page(
                            md_text.strip(), 
                            page_num + 1,  # 1-indexed
                            total_pages
                        )
                        if is_ref:
                            filter_stats['pages_filtered'] += 1
                            filter_stats['filtered_pages'].append({
                                'page': page_num + 1,
                                'reason': reason
                            })
                            logger.debug(
                                f"Filtered page {page_num + 1}/{total_pages} from "
                                f"{pdf_path.name}: {reason}"
                            )
                            continue  # Skip this page
                    
                    filter_stats['pages_extracted'] += 1
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
                filter_stats['pages_extracted'] += 1
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
    
    if filter_references and filter_stats['pages_filtered'] > 0:
        logger.info(
            f"Filtered {filter_stats['pages_filtered']}/{total_pages} reference pages "
            f"from {pdf_path.name}"
        )
    
    return results, filter_stats


def load_pdfs_from_directory(
    pdf_directory: str,
    page_chunks: bool = True,
    recursive: bool = False,
    filter_references: bool = True
) -> Tuple[list[dict], Dict[str, Any]]:
    """
    Load and extract text from all PDF files in a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files.
        page_chunks: If True, chunk by page. If False, full document per PDF.
        recursive: If True, search subdirectories recursively.
        filter_references: If True, filter out pages that appear to be reference sections.
    
    Returns:
        Tuple of:
        - List of dictionaries containing extracted text and metadata from all PDFs.
        - Dictionary with combined filtering statistics
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
        return [], {}
    
    print(f"Found {len(pdf_files)} PDF file(s) in {pdf_directory}")
    if filter_references:
        print("Reference filtering: ENABLED")
    
    all_documents = []
    combined_stats = {
        'total_pdfs': len(pdf_files),
        'total_pages_extracted': 0,
        'total_pages_filtered': 0,
        'per_pdf_stats': {}
    }
    
    for pdf_path in tqdm(pdf_files, desc="Extracting text from PDFs"):
        try:
            docs, stats = extract_text_from_pdf(
                str(pdf_path), 
                page_chunks=page_chunks,
                filter_references=filter_references
            )
            all_documents.extend(docs)
            
            # Accumulate statistics
            combined_stats['total_pages_extracted'] += stats.get('pages_extracted', 0)
            combined_stats['total_pages_filtered'] += stats.get('pages_filtered', 0)
            combined_stats['per_pdf_stats'][pdf_path.name] = stats
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    # Print summary
    print(f"Extracted {len(all_documents)} document chunks from {len(pdf_files)} PDF(s)")
    if filter_references and combined_stats['total_pages_filtered'] > 0:
        total_pages = combined_stats['total_pages_extracted'] + combined_stats['total_pages_filtered']
        print(
            f"Reference filtering: {combined_stats['total_pages_filtered']}/{total_pages} "
            f"pages filtered ({100*combined_stats['total_pages_filtered']/max(total_pages,1):.1f}%)"
        )
    
    return all_documents, combined_stats


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

