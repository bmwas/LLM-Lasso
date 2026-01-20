"""
This script implements chunking for the .json file scraped down from Omim
and for PDF documents extracted via pymupdf4llm.
"""

import json
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Chunking omim json docs by attaching gene names and other metadata as metadata tags.
def chunk_by_gene(json_file, output_file, chunk_size=1000, chunk_overlap=200):
    """
    Chunk long fields (textSection, clinicalSynopsis, and geneMapData) from JSON objects,
    and include "full_name" in the metadata.
    Args:
        json_file (str): Path to the input JSON file.
        output_file (str): Path to save the chunked JSON output.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []

    # Count total lines in the input file for progress bar
    with open(json_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Process the JSON file line by line with tqdm
    with open(json_file, "r", encoding="utf-8") as f, tqdm(total=total_lines, desc="Processing JSON lines") as pbar:
        for line in f:
            # Parse the JSON object
            entry = json.loads(line)
            gene_name = entry.get("gene_name", "Unknown")
            preferred_title = entry.get("preferred_title", "Unknown Full Name")
            text_section = entry.get("text_description", "")
            clinical_synopsis = entry.get("clinical_synopsis", "")
            gene_map_data = entry.get("gene_map_data", "")

            # Chunk the text section
            if text_section:
                text_chunks = text_splitter.split_text(text_section)
                for chunk in text_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "text_description"
                        }
                    })

            # Chunk the clinical synopsis
            if clinical_synopsis:
                clinical_chunks = text_splitter.split_text(clinical_synopsis)
                for chunk in clinical_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "clinical_synopsis"
                        }
                    })

            # Chunk the gene map data
            if gene_map_data:
                gene_map_chunks = text_splitter.split_text(gene_map_data)
                for chunk in gene_map_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "gene_map_data"
                        }
                    })

            # Update progress bar
            pbar.update(1)

    # Save the chunks to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        for chunk in chunks:
            f_out.write(json.dumps(chunk) + "\n")

    print(f"Chunked data saved to {output_file}")


def chunk_pdf_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    output_file: Optional[str] = None
) -> list[dict]:
    """
    Chunk PDF documents extracted by pdf_RAG_process into smaller pieces for vectorstore ingestion.
    
    Args:
        documents: List of dictionaries from extract_text_from_pdf or load_pdfs_from_directory.
                   Each dict should have 'content' and 'metadata' keys.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.
        output_file: Optional path to save chunked output as JSON lines file.
    
    Returns:
        List of chunked documents with preserved and updated metadata.
    """
    # Initialize text splitter with separators appropriate for markdown/scientific text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",      # Major section headers (markdown)
            "\n### ",     # Subsection headers
            "\n#### ",    # Sub-subsection headers
            "\n\n",       # Paragraph breaks
            "\n",         # Line breaks
            ". ",         # Sentence boundaries
            ", ",         # Clause boundaries
            " ",          # Word boundaries
            ""            # Character fallback
        ]
    )
    
    chunks = []
    
    for doc in tqdm(documents, desc="Chunking PDF documents"):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {}).copy()
        
        if not content.strip():
            continue
        
        # Split the content into chunks
        text_chunks = text_splitter.split_text(content)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            # Create new metadata with chunk information
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["total_chunks"] = len(text_chunks)
            
            # Try to extract section header from chunk if present
            section_header = _extract_section_header(chunk)
            if section_header:
                chunk_metadata["section_header"] = section_header
            
            chunks.append({
                "content": chunk.strip(),
                "metadata": chunk_metadata
            })
    
    print(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
    
    # Optionally save to file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for chunk in chunks:
                f_out.write(json.dumps(chunk) + "\n")
        print(f"Chunked data saved to {output_file}")
    
    return chunks


def _extract_section_header(text: str) -> Optional[str]:
    """
    Extract a section header from the beginning of a text chunk if present.
    
    Args:
        text: Text chunk that may start with a markdown header.
    
    Returns:
        The section header text without markdown symbols, or None if not found.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None
    
    first_line = lines[0].strip()
    
    # Check for markdown headers
    if first_line.startswith('#'):
        # Remove leading # symbols and whitespace
        header = first_line.lstrip('#').strip()
        if header:
            return header
    
    return None