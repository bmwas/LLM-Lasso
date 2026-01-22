"""
Reference Section Filter for PDF Documents

This module provides functionality to detect and filter out reference/bibliography
sections from scientific papers during PDF indexing. This prevents citation metadata
from polluting the RAG retrieval with non-substantive content.

The filtering operates at two levels:
1. Page-level: Detects entire pages that are primarily references
2. Chunk-level: Filters individual text chunks that contain reference content

Usage:
    from llm_lasso.utils.reference_filter import filter_reference_content
    
    filtered_docs, stats = filter_reference_content(documents)
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional

# Set up module logger
logger = logging.getLogger("pdf_rag.reference_filter")

# ==================== Reference Section Headers ====================
# Patterns that indicate the start of a reference section

REFERENCE_HEADER_PATTERNS = [
    # Standard headers (case insensitive)
    r'^\s*#{0,6}\s*references?\s*$',
    r'^\s*#{0,6}\s*bibliography\s*$',
    r'^\s*#{0,6}\s*works?\s+cited\s*$',
    r'^\s*#{0,6}\s*literature\s+cited\s*$',
    r'^\s*#{0,6}\s*literature\s*$',
    r'^\s*#{0,6}\s*citations?\s*$',
    r'^\s*#{0,6}\s*cited\s+literature\s*$',
    r'^\s*#{0,6}\s*reference\s+list\s*$',
    r'^\s*#{0,6}\s*sources?\s*$',
    r'^\s*#{0,6}\s*notes?\s+and\s+references?\s*$',
    # Numbered section headers
    r'^\s*\d+\.?\s*references?\s*$',
    r'^\s*\d+\.?\s*bibliography\s*$',
    # With colons
    r'^\s*references?\s*:\s*$',
    r'^\s*bibliography\s*:\s*$',
]

# Compile patterns for efficiency
COMPILED_HEADER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
    for pattern in REFERENCE_HEADER_PATTERNS
]


# ==================== Reference Entry Patterns ====================
# Patterns that identify individual reference entries

REFERENCE_ENTRY_PATTERNS = [
    # Numbered references: [1], 1., (1), 1)
    r'^\s*\[?\d{1,3}\]?[\.\)]\s+[A-Z]',
    
    # DOI patterns
    r'doi:\s*10\.\d{4,}',
    r'https?://doi\.org/10\.\d{4,}',
    r'https?://dx\.doi\.org/10\.\d{4,}',
    
    # PubMed/PMC IDs
    r'PMID:\s*\d+',
    r'PMC\d+',
    r'PubMed:\s*\d+',
    
    # arXiv
    r'arXiv:\s*\d{4}\.\d+',
    r'https?://arxiv\.org/abs/\d{4}\.\d+',
    
    # Journal citation patterns: Year;Volume(Issue):Pages
    r'\d{4}\s*;\s*\d+\s*\(\d+\)\s*:\s*\d+',
    r'\d{4}\s*;\s*\d+\s*:\s*\d+[-–]\d+',
    
    # Standard author patterns (multiple authors with initials)
    r'^[A-Z][a-z]+\s+[A-Z]{1,2},\s+[A-Z][a-z]+\s+[A-Z]{1,2}',
    r'^[A-Z][a-z]+\s+[A-Z]{1,2},\s+et\s+al\.',
    
    # URLs (common in references)
    r'Available\s+(?:from|at):\s*https?://',
    r'\[cited\s+\d{4}',
    r'\[Internet\]',
    r'\[Online\]',
    
    # Page ranges
    r'pp?\.\s*\d+[-–]\d+',
    r'pages?\s+\d+[-–]\d+',
    
    # ISBN/ISSN
    r'ISBN[:\s]*[\d\-X]+',
    r'ISSN[:\s]*[\d\-]+',
]

# Compile patterns for efficiency
COMPILED_ENTRY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) 
    for pattern in REFERENCE_ENTRY_PATTERNS
]


# ==================== Appendix/Supplementary Headers ====================
# Other sections that may contain non-substantive content

APPENDIX_HEADER_PATTERNS = [
    r'^\s*#{0,6}\s*appendix\s*[a-z]?\s*$',
    r'^\s*#{0,6}\s*appendices\s*$',
    r'^\s*#{0,6}\s*supplementary\s+(?:material|information|data)\s*$',
    r'^\s*#{0,6}\s*supporting\s+information\s*$',
    r'^\s*#{0,6}\s*acknowledgements?\s*$',
    r'^\s*#{0,6}\s*author\s+contributions?\s*$',
    r'^\s*#{0,6}\s*conflict\s+of\s+interest\s*$',
    r'^\s*#{0,6}\s*declaration\s+of\s+interests?\s*$',
    r'^\s*#{0,6}\s*funding\s*$',
    r'^\s*#{0,6}\s*data\s+availability\s*$',
]

COMPILED_APPENDIX_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
    for pattern in APPENDIX_HEADER_PATTERNS
]


# ==================== Detection Functions ====================

def is_reference_section_header(text: str) -> bool:
    """
    Check if text contains a reference section header.
    
    Args:
        text: Text to check (typically first few lines of a page/chunk)
    
    Returns:
        True if a reference header is detected
    """
    # Check each line separately
    lines = text.split('\n')[:10]  # Only check first 10 lines
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        for pattern in COMPILED_HEADER_PATTERNS:
            if pattern.search(line_stripped):
                return True
    
    return False


def is_appendix_section_header(text: str) -> bool:
    """
    Check if text contains an appendix/supplementary section header.
    
    Args:
        text: Text to check
    
    Returns:
        True if an appendix header is detected
    """
    lines = text.split('\n')[:10]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        for pattern in COMPILED_APPENDIX_PATTERNS:
            if pattern.search(line_stripped):
                return True
    
    return False


def count_reference_indicators(text: str) -> Dict[str, int]:
    """
    Count various reference indicators in text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with counts of different reference indicators
    """
    counts = {
        'numbered_refs': 0,
        'dois': 0,
        'urls': 0,
        'author_patterns': 0,
        'year_patterns': 0,
        'journal_patterns': 0,
        'total_lines': 0,
        'non_empty_lines': 0,
    }
    
    lines = text.split('\n')
    counts['total_lines'] = len(lines)
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        counts['non_empty_lines'] += 1
        
        # Check for numbered reference start
        if re.match(r'^\s*\[?\d{1,3}\]?[\.\)]\s+[A-Z]', line_stripped):
            counts['numbered_refs'] += 1
        
        # Check for DOIs
        if re.search(r'doi[:\s]*10\.\d{4,}', line_stripped, re.IGNORECASE):
            counts['dois'] += 1
        
        # Check for URLs
        if re.search(r'https?://', line_stripped):
            counts['urls'] += 1
        
        # Check for author patterns
        if re.search(r'[A-Z][a-z]+\s+[A-Z]{1,2}[,\.]', line_stripped):
            counts['author_patterns'] += 1
        
        # Check for year in parentheses or standalone
        if re.search(r'\((?:19|20)\d{2}\)', line_stripped):
            counts['year_patterns'] += 1
        
        # Check for journal citation patterns
        if re.search(r'\d{4}\s*;\s*\d+', line_stripped):
            counts['journal_patterns'] += 1
    
    return counts


def calculate_reference_density(text: str) -> float:
    """
    Calculate the proportion of text that appears to be reference content.
    
    Args:
        text: Text to analyze
    
    Returns:
        Float between 0 and 1 indicating reference density
    """
    counts = count_reference_indicators(text)
    
    if counts['non_empty_lines'] == 0:
        return 0.0
    
    # Calculate weighted score
    # Different indicators have different weights
    score = (
        counts['numbered_refs'] * 2.0 +  # Strong indicator
        counts['dois'] * 3.0 +            # Very strong indicator
        counts['journal_patterns'] * 2.0 + # Strong indicator
        counts['year_patterns'] * 0.5 +   # Weak indicator (years appear in regular text too)
        counts['author_patterns'] * 0.5   # Weak indicator
    )
    
    # Normalize by number of non-empty lines
    density = score / counts['non_empty_lines']
    
    # Cap at 1.0
    return min(density, 1.0)


def is_reference_line(line: str) -> bool:
    """
    Check if a single line looks like a reference entry.
    
    Args:
        line: Single line of text
    
    Returns:
        True if the line appears to be a reference
    """
    line_stripped = line.strip()
    
    if not line_stripped:
        return False
    
    # Check against compiled patterns
    for pattern in COMPILED_ENTRY_PATTERNS:
        if pattern.search(line_stripped):
            return True
    
    return False


def is_reference_page(
    content: str, 
    page_num: int, 
    total_pages: int,
    density_threshold: float = 0.3,
    position_weight: bool = True
) -> Tuple[bool, str]:
    """
    Determine if a page is primarily a reference page.
    
    Args:
        content: Page content
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages in document
        density_threshold: Minimum reference density to classify as reference page
        position_weight: If True, pages near the end are more likely to be references
    
    Returns:
        Tuple of (is_reference_page, reason)
    """
    # Check for explicit reference header
    if is_reference_section_header(content):
        return True, "Contains reference section header"
    
    # Calculate reference density
    density = calculate_reference_density(content)
    
    # Apply position weighting - references are more likely near the end
    effective_threshold = density_threshold
    if position_weight and total_pages > 5:
        # Pages in last 20% of document get lower threshold
        position_ratio = page_num / total_pages
        if position_ratio > 0.8:
            effective_threshold *= 0.7  # 30% easier to classify as reference
        elif position_ratio > 0.7:
            effective_threshold *= 0.85  # 15% easier
    
    if density >= effective_threshold:
        return True, f"High reference density ({density:.2f} >= {effective_threshold:.2f})"
    
    # Check for DOI concentration (strong signal)
    counts = count_reference_indicators(content)
    if counts['dois'] >= 3:
        return True, f"High DOI concentration ({counts['dois']} DOIs)"
    
    # Check for numbered reference patterns
    if counts['numbered_refs'] >= 5 and counts['non_empty_lines'] > 0:
        ref_ratio = counts['numbered_refs'] / counts['non_empty_lines']
        if ref_ratio >= 0.3:
            return True, f"High numbered reference ratio ({ref_ratio:.2f})"
    
    return False, "Does not appear to be a reference page"


def is_reference_chunk(
    content: str,
    density_threshold: float = 0.4
) -> Tuple[bool, str]:
    """
    Determine if a text chunk is primarily reference content.
    
    This is stricter than page-level detection since chunks are smaller.
    
    Args:
        content: Chunk content
        density_threshold: Minimum reference density to classify as reference
    
    Returns:
        Tuple of (is_reference_chunk, reason)
    """
    # Check for explicit reference header at start
    first_lines = '\n'.join(content.split('\n')[:3])
    if is_reference_section_header(first_lines):
        return True, "Starts with reference section header"
    
    # Calculate reference density
    density = calculate_reference_density(content)
    
    if density >= density_threshold:
        return True, f"High reference density ({density:.2f} >= {density_threshold:.2f})"
    
    # Check for DOI concentration
    counts = count_reference_indicators(content)
    if counts['dois'] >= 2:
        return True, f"Multiple DOIs in chunk ({counts['dois']} DOIs)"
    
    # Check for numbered reference patterns dominating the chunk
    if counts['numbered_refs'] >= 3 and counts['non_empty_lines'] > 0:
        ref_ratio = counts['numbered_refs'] / counts['non_empty_lines']
        if ref_ratio >= 0.4:
            return True, f"High numbered reference ratio ({ref_ratio:.2f})"
    
    return False, "Does not appear to be reference content"


# ==================== Main Filtering Functions ====================

def filter_reference_pages(
    documents: List[Dict],
    density_threshold: float = 0.3,
    position_weight: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Filter out pages that are primarily references.
    
    Args:
        documents: List of document dicts with 'content' and 'metadata'
        density_threshold: Threshold for reference density
        position_weight: Whether to weight by position in document
    
    Returns:
        Tuple of (filtered_documents, filter_statistics)
    """
    filtered = []
    stats = {
        'pages_analyzed': len(documents),
        'pages_filtered': 0,
        'pages_kept': 0,
        'filtered_reasons': [],
        'bytes_removed': 0,
    }
    
    for doc in documents:
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        page_num = metadata.get('page', 1)
        total_pages = metadata.get('total_pages', 1)
        
        is_ref, reason = is_reference_page(
            content, page_num, total_pages, 
            density_threshold, position_weight
        )
        
        if is_ref:
            stats['pages_filtered'] += 1
            stats['bytes_removed'] += len(content.encode('utf-8'))
            stats['filtered_reasons'].append({
                'source': metadata.get('filename', 'unknown'),
                'page': page_num,
                'reason': reason
            })
            logger.debug(f"Filtered page {page_num} from {metadata.get('filename', 'unknown')}: {reason}")
        else:
            filtered.append(doc)
            stats['pages_kept'] += 1
    
    return filtered, stats


def filter_reference_chunks(
    chunks: List[Dict],
    density_threshold: float = 0.4
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Filter out chunks that are primarily references.
    
    Args:
        chunks: List of chunk dicts with 'content' and 'metadata'
        density_threshold: Threshold for reference density
    
    Returns:
        Tuple of (filtered_chunks, filter_statistics)
    """
    filtered = []
    stats = {
        'chunks_analyzed': len(chunks),
        'chunks_filtered': 0,
        'chunks_kept': 0,
        'filtered_reasons': [],
        'bytes_removed': 0,
    }
    
    for chunk in chunks:
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        is_ref, reason = is_reference_chunk(content, density_threshold)
        
        if is_ref:
            stats['chunks_filtered'] += 1
            stats['bytes_removed'] += len(content.encode('utf-8'))
            stats['filtered_reasons'].append({
                'source': metadata.get('filename', 'unknown'),
                'page': metadata.get('page', 'unknown'),
                'chunk_index': metadata.get('chunk_index', 'unknown'),
                'reason': reason
            })
            logger.debug(
                f"Filtered chunk {metadata.get('chunk_index', '?')} from "
                f"{metadata.get('filename', 'unknown')} page {metadata.get('page', '?')}: {reason}"
            )
        else:
            filtered.append(chunk)
            stats['chunks_kept'] += 1
    
    return filtered, stats


def filter_reference_content(
    documents: List[Dict],
    filter_pages: bool = True,
    filter_chunks: bool = True,
    page_density_threshold: float = 0.3,
    chunk_density_threshold: float = 0.4,
    position_weight: bool = True,
    log_filtered: bool = True
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Main filtering function that applies both page-level and chunk-level filtering.
    
    Args:
        documents: List of document dicts with 'content' and 'metadata'
        filter_pages: Whether to apply page-level filtering
        filter_chunks: Whether to apply chunk-level filtering
        page_density_threshold: Threshold for page-level reference density
        chunk_density_threshold: Threshold for chunk-level reference density
        position_weight: Whether to weight page filtering by position
        log_filtered: Whether to log filtering statistics
    
    Returns:
        Tuple of (filtered_documents, combined_statistics)
    """
    combined_stats = {
        'input_documents': len(documents),
        'output_documents': 0,
        'page_filtering': None,
        'chunk_filtering': None,
        'total_bytes_removed': 0,
    }
    
    current_docs = documents
    
    # Stage 1: Page-level filtering
    if filter_pages:
        current_docs, page_stats = filter_reference_pages(
            current_docs, page_density_threshold, position_weight
        )
        combined_stats['page_filtering'] = page_stats
        combined_stats['total_bytes_removed'] += page_stats['bytes_removed']
        
        if log_filtered:
            logger.info(
                f"Page filtering: {page_stats['pages_filtered']}/{page_stats['pages_analyzed']} "
                f"pages removed ({page_stats['bytes_removed']:,} bytes)"
            )
    
    # Stage 2: Chunk-level filtering
    if filter_chunks:
        current_docs, chunk_stats = filter_reference_chunks(
            current_docs, chunk_density_threshold
        )
        combined_stats['chunk_filtering'] = chunk_stats
        combined_stats['total_bytes_removed'] += chunk_stats['bytes_removed']
        
        if log_filtered:
            logger.info(
                f"Chunk filtering: {chunk_stats['chunks_filtered']}/{chunk_stats['chunks_analyzed']} "
                f"chunks removed ({chunk_stats['bytes_removed']:,} bytes)"
            )
    
    combined_stats['output_documents'] = len(current_docs)
    
    if log_filtered:
        logger.info(
            f"Reference filtering complete: {combined_stats['input_documents']} -> "
            f"{combined_stats['output_documents']} documents "
            f"({combined_stats['total_bytes_removed']:,} bytes removed)"
        )
    
    return current_docs, combined_stats


def get_filter_summary(stats: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of filtering statistics.
    
    Args:
        stats: Statistics dictionary from filter_reference_content
    
    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 50,
        "Reference Filtering Summary",
        "=" * 50,
        f"Input documents: {stats['input_documents']}",
        f"Output documents: {stats['output_documents']}",
        f"Total bytes removed: {stats['total_bytes_removed']:,}",
        "",
    ]
    
    if stats.get('page_filtering'):
        ps = stats['page_filtering']
        lines.extend([
            "Page-level filtering:",
            f"  Pages analyzed: {ps['pages_analyzed']}",
            f"  Pages filtered: {ps['pages_filtered']} ({100*ps['pages_filtered']/max(ps['pages_analyzed'],1):.1f}%)",
            f"  Pages kept: {ps['pages_kept']}",
            "",
        ])
    
    if stats.get('chunk_filtering'):
        cs = stats['chunk_filtering']
        lines.extend([
            "Chunk-level filtering:",
            f"  Chunks analyzed: {cs['chunks_analyzed']}",
            f"  Chunks filtered: {cs['chunks_filtered']} ({100*cs['chunks_filtered']/max(cs['chunks_analyzed'],1):.1f}%)",
            f"  Chunks kept: {cs['chunks_kept']}",
            "",
        ])
    
    lines.append("=" * 50)
    
    return '\n'.join(lines)
