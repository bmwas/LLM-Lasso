#!/usr/bin/env python3
"""
RAG Document Embedding Visualizer

This script analyzes and visualizes RAG retrieved documents from rag_retrieved_documents.json.
It embeds document content using OpenAI embeddings (same as RAG), projects to 2D using
UMAP and t-SNE, clusters using K-Means and HDBSCAN, and generates publication-quality
visualizations with feature query labels.

Usage:
    python scripts/visualize_rag_embeddings.py \
        --input /path/to/rag_retrieved_documents.json \
        --output /path/to/output_directory

Author: LLM-Lasso Team
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==================== Logging Setup ====================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging with colored output."""
    logger = logging.getLogger("rag_visualizer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ==================== Data Loading ====================

def load_rag_documents(input_path: str, logger: logging.Logger) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Load and parse RAG documents from JSON file.
    
    Args:
        input_path: Path to rag_retrieved_documents.json
        logger: Logger instance
    
    Returns:
        Tuple of (documents list, doc_to_features mapping)
    """
    logger.info(f"Loading RAG documents from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    metadata = data.get("metadata", {})
    logger.info(f"  Category: {metadata.get('category', 'Unknown')}")
    logger.info(f"  Features queried: {metadata.get('num_features_queried', 'Unknown')}")
    logger.info(f"  Docs per query: {metadata.get('docs_per_query', 'Unknown')}")
    
    # Build document list and track which features retrieved each document
    documents = []
    doc_to_features = defaultdict(list)  # doc_id -> list of features that retrieved it
    doc_contents = {}  # doc_id -> content (for deduplication)
    
    queries = data.get("queries", {})
    
    for feature_name, query_data in queries.items():
        for doc in query_data.get("documents", []):
            doc_id = doc.get("doc_id")
            content = doc.get("full_content", "")
            
            if not content:
                continue
            
            # Track which features retrieved this document
            doc_to_features[doc_id].append(feature_name)
            
            # Store document if not seen before
            if doc_id not in doc_contents:
                doc_contents[doc_id] = content
                documents.append({
                    "doc_id": doc_id,
                    "content": content,
                    "source_file": doc.get("source_file", "Unknown"),
                    "page": doc.get("page", "N/A"),
                    "content_length": len(content)
                })
    
    logger.info(f"  Unique documents: {len(documents)}")
    logger.info(f"  Total feature-document associations: {sum(len(f) for f in doc_to_features.values())}")
    
    return documents, dict(doc_to_features)


# ==================== Embedding ====================

def compute_embeddings(
    documents: List[Dict],
    logger: logging.Logger,
    batch_size: int = 100
) -> np.ndarray:
    """
    Compute OpenAI embeddings for document content.
    Uses the same embedding method as the RAG pipeline.
    
    Args:
        documents: List of document dictionaries with 'content' key
        logger: Logger instance
        batch_size: Batch size for embedding API calls
    
    Returns:
        numpy array of embeddings (n_docs, embedding_dim)
    """
    logger.info("Computing OpenAI embeddings...")
    logger.info(f"  Documents to embed: {len(documents)}")
    
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
        raise
    
    # Initialize embeddings (same as RAG)
    embeddings_model = OpenAIEmbeddings()
    
    # Extract content
    texts = [doc["content"] for doc in documents]
    
    # Embed in batches
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"  Embedding batch {batch_num}/{n_batches} ({len(batch)} documents)...")
        
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    
    embeddings_array = np.array(all_embeddings)
    logger.info(f"  Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array


# ==================== Dimensionality Reduction ====================

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str,
    logger: logging.Logger,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensions to 2D.
    
    Args:
        embeddings: High-dimensional embeddings
        method: 'umap' or 'tsne'
        logger: Logger instance
        random_state: Random seed for reproducibility
    
    Returns:
        2D coordinates (n_docs, 2)
    """
    logger.info(f"Reducing dimensions with {method.upper()}...")
    
    if method == 'umap':
        try:
            import umap
        except ImportError:
            logger.error("umap-learn not installed. Install with: pip install umap-learn")
            raise
        
        reducer = umap.UMAP(
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=random_state,
            verbose=False
        )
        coords = reducer.fit_transform(embeddings)
        
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        
        # Adjust perplexity based on number of samples
        perplexity = min(30, max(5, len(embeddings) // 4))
        
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate='auto',
            init='pca'
        )
        coords = reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
    
    logger.info(f"  Output shape: {coords.shape}")
    return coords


# ==================== Clustering ====================

def cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    logger: logging.Logger,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster embeddings using K-Means or HDBSCAN.
    
    Args:
        embeddings: Embeddings (can be high-dim or 2D)
        method: 'kmeans' or 'hdbscan'
        logger: Logger instance
        random_state: Random seed
    
    Returns:
        Tuple of (cluster labels, clustering metadata)
    """
    logger.info(f"Clustering with {method.upper()}...")
    
    metadata = {"method": method}
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Find optimal k using silhouette score
        max_k = min(15, len(embeddings) - 1)
        min_k = 2
        
        best_k = min_k
        best_score = -1
        scores = {}
        
        logger.info(f"  Finding optimal k (range: {min_k}-{max_k})...")
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                scores[k] = score
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        logger.info(f"  Optimal k: {best_k} (silhouette: {best_score:.3f})")
        
        # Fit with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        metadata.update({
            "n_clusters": best_k,
            "silhouette_score": float(best_score),
            "silhouette_scores_by_k": {str(k): float(v) for k, v in scores.items()},
            "inertia": float(kmeans.inertia_)
        })
        
    elif method == 'hdbscan':
        try:
            import hdbscan
        except ImportError:
            logger.error("hdbscan not installed. Install with: pip install hdbscan")
            raise
        
        # Adjust min_cluster_size based on data size
        min_cluster_size = max(3, len(embeddings) // 20)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
        
        metadata.update({
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "noise_percentage": float(100 * n_noise / len(labels)),
            "min_cluster_size": min_cluster_size
        })
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'hdbscan'")
    
    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {str(int(u)): int(c) for u, c in zip(unique, counts)}
    metadata["cluster_sizes"] = cluster_sizes
    
    logger.info(f"  Cluster sizes: {cluster_sizes}")
    
    return labels, metadata


# ==================== Visualization ====================

def generate_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    documents: List[Dict],
    doc_to_features: Dict[str, List[str]],
    dim_method: str,
    cluster_method: str,
    cluster_metadata: Dict[str, Any],
    output_path: str,
    logger: logging.Logger
) -> None:
    """
    Generate a publication-quality scatter plot.
    
    Args:
        coords: 2D coordinates (n_docs, 2)
        labels: Cluster labels
        documents: Document list
        doc_to_features: Mapping of doc_id to feature names
        dim_method: Dimensionality reduction method name
        cluster_method: Clustering method name
        cluster_metadata: Clustering statistics
        output_path: Path to save the plot
        logger: Logger instance
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    
    logger.info(f"Generating plot: {os.path.basename(output_path)}")
    
    # Publication-quality settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Create figure with two panels: main plot + legend
    fig = plt.figure(figsize=(16, 10))
    
    # Main scatter plot
    ax_main = fig.add_axes([0.08, 0.1, 0.6, 0.8])
    
    # Get unique clusters and create colormap
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Use a qualitative colormap
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral
    
    # Plot each cluster
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        
        if cluster_id == -1:
            # Noise points (HDBSCAN)
            color = '#CCCCCC'
            label = 'Noise'
            alpha = 0.4
            marker = 'x'
            s = 50
        else:
            color = cmap(idx / max(n_clusters - 1, 1))
            label = f'Cluster {cluster_id}'
            alpha = 0.7
            marker = 'o'
            s = 80
        
        ax_main.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=label, alpha=alpha,
            s=s, marker=marker, edgecolors='white', linewidth=0.5
        )
    
    # Add feature labels for each point
    # Collect all feature associations for annotation
    feature_annotations = []
    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        features = doc_to_features.get(doc_id, [])
        if features:
            # Use first feature or join multiple with comma
            if len(features) <= 2:
                feature_text = ", ".join(features)
            else:
                feature_text = f"{features[0]} +{len(features)-1}"
            feature_annotations.append((coords[i, 0], coords[i, 1], feature_text))
    
    # Add annotations (limit to avoid clutter)
    # If too many points, only annotate cluster centroids
    if len(feature_annotations) <= 50:
        for x, y, text in feature_annotations:
            ax_main.annotate(
                text, (x, y), fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords='offset points',
                ha='left', va='bottom'
            )
    else:
        # Annotate cluster centroids with most common features
        logger.info("  Too many points - annotating cluster centroids only")
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            centroid_x = coords[mask, 0].mean()
            centroid_y = coords[mask, 1].mean()
            
            # Get most common feature in this cluster
            cluster_features = []
            for i, doc in enumerate(documents):
                if mask[i]:
                    cluster_features.extend(doc_to_features.get(doc["doc_id"], []))
            
            if cluster_features:
                from collections import Counter
                most_common = Counter(cluster_features).most_common(3)
                feature_text = ", ".join([f[0] for f in most_common])
                ax_main.annotate(
                    feature_text, (centroid_x, centroid_y), fontsize=8, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
                )
    
    # Styling
    ax_main.set_xlabel(f'{dim_method.upper()} Dimension 1', fontweight='bold')
    ax_main.set_ylabel(f'{dim_method.upper()} Dimension 2', fontweight='bold')
    ax_main.set_title(
        f'RAG Document Embeddings\n{dim_method.upper()} + {cluster_method.upper()}',
        fontweight='bold', fontsize=14, pad=10
    )
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend on the right side
    ax_legend = fig.add_axes([0.72, 0.1, 0.26, 0.8])
    ax_legend.axis('off')
    
    # Create legend elements
    legend_elements = []
    for idx, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            color = '#CCCCCC'
            label = f'Noise (n={(labels == -1).sum()})'
        else:
            color = cmap(idx / max(n_clusters - 1, 1))
            label = f'Cluster {cluster_id} (n={(labels == cluster_id).sum()})'
        legend_elements.append(Patch(facecolor=color, edgecolor='gray', label=label))
    
    ax_legend.legend(
        handles=legend_elements, loc='upper left', frameon=True,
        fancybox=True, shadow=True, title='Clusters', title_fontsize=11
    )
    
    # Add clustering stats as text
    stats_text = f"Clustering Statistics:\n"
    stats_text += f"  Method: {cluster_method.upper()}\n"
    stats_text += f"  Clusters: {cluster_metadata.get('n_clusters', 'N/A')}\n"
    
    if 'silhouette_score' in cluster_metadata:
        stats_text += f"  Silhouette: {cluster_metadata['silhouette_score']:.3f}\n"
    if 'noise_percentage' in cluster_metadata:
        stats_text += f"  Noise: {cluster_metadata['noise_percentage']:.1f}%\n"
    
    stats_text += f"\nTotal Documents: {len(documents)}"
    
    ax_legend.text(
        0.02, 0.3, stats_text, transform=ax_legend.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='gray')
    )
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


def generate_combined_dashboard(
    all_coords: Dict[str, np.ndarray],
    all_labels: Dict[str, np.ndarray],
    documents: List[Dict],
    doc_to_features: Dict[str, List[str]],
    output_path: str,
    logger: logging.Logger
) -> None:
    """
    Generate a combined dashboard showing all 4 visualizations.
    
    Args:
        all_coords: Dict mapping 'umap'/'tsne' to coordinates
        all_labels: Dict mapping 'umap_kmeans', etc. to labels
        documents: Document list
        doc_to_features: Feature mapping
        output_path: Output path
        logger: Logger instance
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    logger.info("Generating combined dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })
    
    configs = [
        ('umap', 'kmeans', 'UMAP + K-Means', axes[0, 0]),
        ('umap', 'hdbscan', 'UMAP + HDBSCAN', axes[0, 1]),
        ('tsne', 'kmeans', 't-SNE + K-Means', axes[1, 0]),
        ('tsne', 'hdbscan', 't-SNE + HDBSCAN', axes[1, 1]),
    ]
    
    for dim_method, cluster_method, title, ax in configs:
        coords = all_coords[dim_method]
        labels = all_labels[f"{dim_method}_{cluster_method}"]
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters <= 10:
            cmap = plt.cm.tab10
        elif n_clusters <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral
        
        for idx, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            
            if cluster_id == -1:
                color = '#CCCCCC'
                alpha = 0.4
                marker = 'x'
                s = 30
            else:
                color = cmap(idx / max(n_clusters - 1, 1))
                alpha = 0.7
                marker = 'o'
                s = 40
            
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[color], alpha=alpha, s=s, marker=marker,
                edgecolors='white', linewidth=0.3
            )
        
        ax.set_xlabel(f'{dim_method.upper()} Dim 1')
        ax.set_ylabel(f'{dim_method.upper()} Dim 2')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add cluster count annotation
        n_actual_clusters = len([l for l in unique_labels if l != -1])
        n_noise = (labels == -1).sum() if -1 in labels else 0
        ax.annotate(
            f'Clusters: {n_actual_clusters}' + (f', Noise: {n_noise}' if n_noise > 0 else ''),
            xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=9, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle(
        'RAG Document Embedding Analysis - Comparison Dashboard',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ==================== Results Saving ====================

def save_results(
    documents: List[Dict],
    doc_to_features: Dict[str, List[str]],
    embeddings: np.ndarray,
    all_coords: Dict[str, np.ndarray],
    all_labels: Dict[str, np.ndarray],
    all_cluster_metadata: Dict[str, Dict],
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    Save embeddings data and summary to files.
    
    Args:
        documents: Document list
        doc_to_features: Feature mapping
        embeddings: Original embeddings
        all_coords: 2D coordinates for each method
        all_labels: Cluster labels for each method combination
        all_cluster_metadata: Clustering metadata
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("Saving results...")
    
    # Create DataFrame with all information
    rows = []
    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        features = doc_to_features.get(doc_id, [])
        
        row = {
            "doc_id": doc_id,
            "source_file": doc["source_file"],
            "page": doc["page"],
            "content_length": doc["content_length"],
            "features": "; ".join(features),
            "num_features": len(features),
            "umap_x": all_coords["umap"][i, 0],
            "umap_y": all_coords["umap"][i, 1],
            "tsne_x": all_coords["tsne"][i, 0],
            "tsne_y": all_coords["tsne"][i, 1],
            "cluster_umap_kmeans": all_labels["umap_kmeans"][i],
            "cluster_umap_hdbscan": all_labels["umap_hdbscan"][i],
            "cluster_tsne_kmeans": all_labels["tsne_kmeans"][i],
            "cluster_tsne_hdbscan": all_labels["tsne_hdbscan"][i],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "rag_embeddings_data.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_documents": len(documents),
        "embedding_dim": embeddings.shape[1],
        "unique_source_files": list(set(doc["source_file"] for doc in documents)),
        "feature_coverage": {
            "total_features": len(set(f for feats in doc_to_features.values() for f in feats)),
            "avg_features_per_doc": np.mean([len(f) for f in doc_to_features.values()]),
            "docs_with_multiple_features": sum(1 for f in doc_to_features.values() if len(f) > 1)
        },
        "clustering_results": all_cluster_metadata
    }
    
    summary_path = os.path.join(output_dir, "rag_embeddings_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {summary_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize RAG retrieved documents using embeddings and clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/visualize_rag_embeddings.py \\
        --input /path/to/rag_retrieved_documents.json \\
        --output /path/to/output_directory
    
    python scripts/visualize_rag_embeddings.py \\
        --input ./rag_retrieved_documents.json \\
        --output ./visualizations \\
        --log_level DEBUG
        """
    )
    
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to rag_retrieved_documents.json file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for plots and data files"
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("RAG DOCUMENT EMBEDDING VISUALIZER")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info("")
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Step 1: Load documents
        logger.info("STEP 1: Loading RAG documents")
        logger.info("-" * 40)
        documents, doc_to_features = load_rag_documents(args.input, logger)
        
        if len(documents) < 3:
            logger.error(f"Not enough documents for clustering (found {len(documents)}, need at least 3)")
            sys.exit(1)
        
        # Step 2: Compute embeddings
        logger.info("")
        logger.info("STEP 2: Computing embeddings")
        logger.info("-" * 40)
        embeddings = compute_embeddings(documents, logger)
        
        # Step 3: Dimensionality reduction
        logger.info("")
        logger.info("STEP 3: Dimensionality reduction")
        logger.info("-" * 40)
        
        all_coords = {}
        for method in ['umap', 'tsne']:
            all_coords[method] = reduce_dimensions(
                embeddings, method, logger, args.random_state
            )
        
        # Step 4: Clustering
        logger.info("")
        logger.info("STEP 4: Clustering")
        logger.info("-" * 40)
        
        all_labels = {}
        all_cluster_metadata = {}
        
        for dim_method in ['umap', 'tsne']:
            for cluster_method in ['kmeans', 'hdbscan']:
                key = f"{dim_method}_{cluster_method}"
                logger.info(f"\n{dim_method.upper()} + {cluster_method.upper()}:")
                
                # Cluster on 2D coordinates
                labels, metadata = cluster_embeddings(
                    all_coords[dim_method], cluster_method, logger, args.random_state
                )
                all_labels[key] = labels
                all_cluster_metadata[key] = metadata
        
        # Step 5: Generate plots
        logger.info("")
        logger.info("STEP 5: Generating plots")
        logger.info("-" * 40)
        
        for dim_method in ['umap', 'tsne']:
            for cluster_method in ['kmeans', 'hdbscan']:
                key = f"{dim_method}_{cluster_method}"
                output_path = os.path.join(args.output, f"rag_embeddings_{key}.png")
                
                generate_plot(
                    coords=all_coords[dim_method],
                    labels=all_labels[key],
                    documents=documents,
                    doc_to_features=doc_to_features,
                    dim_method=dim_method,
                    cluster_method=cluster_method,
                    cluster_metadata=all_cluster_metadata[key],
                    output_path=output_path,
                    logger=logger
                )
        
        # Generate combined dashboard
        dashboard_path = os.path.join(args.output, "rag_embeddings_dashboard.png")
        generate_combined_dashboard(
            all_coords, all_labels, documents, doc_to_features,
            dashboard_path, logger
        )
        
        # Step 6: Save results
        logger.info("")
        logger.info("STEP 6: Saving results")
        logger.info("-" * 40)
        
        save_results(
            documents, doc_to_features, embeddings,
            all_coords, all_labels, all_cluster_metadata,
            args.output, logger
        )
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {args.output}")
        logger.info("")
        logger.info("Generated files:")
        for f in sorted(os.listdir(args.output)):
            logger.info(f"  - {f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
