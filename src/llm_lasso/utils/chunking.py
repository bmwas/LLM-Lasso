"""
This script implements chunking for the .json file scraped down from Omim
and for PDF documents extracted via pymupdf4llm.

Includes optional reference section filtering at the chunk level.
"""

import json
import logging
from typing import Optional, Tuple, Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Set up module logger
logger = logging.getLogger("pdf_rag.chunking")

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


def _normalize_short_lines(content: str, max_short_line_chars: int) -> str:
    """
    Merge short lines (e.g. section headers) with the following non-empty line so they
    are not split into standalone chunks. Replaces the run of newlines between a short
    line and the next non-empty line with a single space.

    Consecutive short lines are merged into the next long line (e.g. "A\\nB\\nLong"
    becomes "A B Long" when A and B are short).

    Args:
        content: Raw text (e.g. markdown from a PDF page).
        max_short_line_chars: Lines with fewer than this many characters (after strip)
                             are considered short and merged with the next non-empty line.

    Returns:
        Text with short lines merged into the following paragraph.
    """
    if max_short_line_chars <= 0:
        return content
    lines = content.split("\n")
    output_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            output_lines.append("")
            i += 1
        elif len(stripped) >= max_short_line_chars:
            output_lines.append(stripped)
            i += 1
        else:
            # Short line: merge with following non-empty lines until we hit a long one or end
            buffer = stripped
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    j += 1
                    continue
                if len(next_stripped) >= max_short_line_chars:
                    output_lines.append(buffer + " " + next_stripped)
                    j += 1
                    break
                buffer = buffer + " " + next_stripped
                j += 1
            else:
                output_lines.append(buffer)
            i = j
    return "\n".join(output_lines)


def chunk_pdf_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    output_file: Optional[str] = None,
    filter_references: bool = True,
    min_chunk_length: int = 0,
    min_chunk_words: Optional[int] = None,
    normalize_newlines: bool = False,
    max_short_line_chars: int = 80
) -> Tuple[list[dict], Dict[str, Any]]:
    """
    Chunk PDF documents extracted by pdf_RAG_process into smaller pieces for vectorstore ingestion.
    
    Args:
        documents: List of dictionaries from extract_text_from_pdf or load_pdfs_from_directory.
                   Each dict should have 'content' and 'metadata' keys.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.
        output_file: Optional path to save chunked output as JSON lines file.
        filter_references: If True, filter out chunks that appear to be reference content.
        min_chunk_length: Minimum character length for a chunk to be indexed. Chunks shorter
                          than this (e.g. section headers like "SUICIDAL BEHAVIOR") are skipped
                          to reduce retrieval noise. Use 0 to disable. Default 0 for backward compat.
        min_chunk_words: Minimum word count for a chunk. If set, chunks with fewer words are
                         skipped. Use None to disable. Useful to drop 1–2 word headers.
        normalize_newlines: If True, merge short lines (e.g. headers) with the next paragraph so
                           they are not standalone chunks; preserves subtitles while reducing noise.
        max_short_line_chars: Lines shorter than this are merged with the next non-empty line
                              when normalize_newlines is True.
    
    Returns:
        Tuple of:
        - List of chunked documents with preserved and updated metadata.
        - Dictionary with filtering statistics
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
    
    # Import reference filter if needed
    is_reference_chunk = None
    if filter_references:
        try:
            from llm_lasso.utils.reference_filter import is_reference_chunk as _is_reference_chunk
            is_reference_chunk = _is_reference_chunk
        except ImportError:
            logger.warning("reference_filter module not available, skipping chunk-level reference filtering")
            filter_references = False
    
    chunks = []
    filter_stats = {
        'total_chunks_created': 0,
        'chunks_filtered': 0,
        'chunks_kept': 0,
        'bytes_filtered': 0,
        'filtered_chunks': [],
        'chunks_filtered_short': 0,
        'filtered_short_chunks': []
    }
    
    for doc in tqdm(documents, desc="Chunking PDF documents"):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {}).copy()
        
        if not content.strip():
            continue
        
        if normalize_newlines and max_short_line_chars > 0:
            content = _normalize_short_lines(content, max_short_line_chars)
        
        # Split the content into chunks
        text_chunks = text_splitter.split_text(content)
        
        for chunk_idx, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
            
            filter_stats['total_chunks_created'] += 1
            
            # Skip chunks that are too short (e.g. section headers, noise)
            chunk_stripped = chunk.strip()
            if min_chunk_length > 0 and len(chunk_stripped) < min_chunk_length:
                filter_stats['chunks_filtered_short'] += 1
                filter_stats['filtered_short_chunks'].append({
                    'source': metadata.get('filename', 'unknown'),
                    'page': metadata.get('page', 'unknown'),
                    'chunk_index': chunk_idx,
                    'reason': f'length {len(chunk_stripped)} < min_chunk_length {min_chunk_length}',
                    'content_preview': chunk_stripped[:80]
                })
                logger.debug(
                    f"Filtered short chunk {chunk_idx} from {metadata.get('filename', 'unknown')} "
                    f"page {metadata.get('page', '?')}: {len(chunk_stripped)} chars"
                )
                continue
            if min_chunk_words is not None and min_chunk_words > 0:
                word_count = len(chunk_stripped.split())
                if word_count < min_chunk_words:
                    filter_stats['chunks_filtered_short'] += 1
                    filter_stats['filtered_short_chunks'].append({
                        'source': metadata.get('filename', 'unknown'),
                        'page': metadata.get('page', 'unknown'),
                        'chunk_index': chunk_idx,
                        'reason': f'words {word_count} < min_chunk_words {min_chunk_words}',
                        'content_preview': chunk_stripped[:80]
                    })
                    logger.debug(
                        f"Filtered short chunk {chunk_idx} from {metadata.get('filename', 'unknown')} "
                        f"page {metadata.get('page', '?')}: {word_count} words"
                    )
                    continue
            
            # Check if this chunk is reference content
            if filter_references and is_reference_chunk is not None:
                is_ref, reason = is_reference_chunk(chunk.strip())
                if is_ref:
                    filter_stats['chunks_filtered'] += 1
                    filter_stats['bytes_filtered'] += len(chunk.encode('utf-8'))
                    filter_stats['filtered_chunks'].append({
                        'source': metadata.get('filename', 'unknown'),
                        'page': metadata.get('page', 'unknown'),
                        'chunk_index': chunk_idx,
                        'reason': reason
                    })
                    logger.debug(
                        f"Filtered chunk {chunk_idx} from {metadata.get('filename', 'unknown')} "
                        f"page {metadata.get('page', '?')}: {reason}"
                    )
                    continue  # Skip this chunk
            
            filter_stats['chunks_kept'] += 1
            
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
    
    # Print summary
    print(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
    if filter_references and filter_stats['chunks_filtered'] > 0:
        print(
            f"Chunk-level reference filtering: {filter_stats['chunks_filtered']}/{filter_stats['total_chunks_created']} "
            f"chunks filtered ({100*filter_stats['chunks_filtered']/max(filter_stats['total_chunks_created'],1):.1f}%)"
        )
    if filter_stats.get('chunks_filtered_short', 0) > 0:
        print(
            f"Short-chunk filtering: {filter_stats['chunks_filtered_short']}/{filter_stats['total_chunks_created']} "
            f"chunks skipped (below min length/words)"
        )
    
    # Optionally save to file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for chunk in chunks:
                f_out.write(json.dumps(chunk) + "\n")
        print(f"Chunked data saved to {output_file}")
    
    return chunks, filter_stats


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