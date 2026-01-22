#!/usr/bin/env python3
"""
PBD LLM-Lasso Pipeline with PDF RAG

Complete end-to-end pipeline for running LLM-Lasso on the Pediatric Bipolar Disorder
dataset with PDF RAG context retrieval.

Features:
- Data loading with validation
- Missing value imputation (configurable strategy)
- PDF vectorstore loading for RAG
- LLM penalty factor generation
- Lasso classification with LLM-generated penalties
- Comprehensive DEBUG/INFO logging throughout

Usage:
    python scripts/run_pbd_llm_lasso.py \
        --prompt-filename prompts/pbd_normal.txt \
        --feature_names_path examples/example_data/pbd_focal_variables.txt \
        --dataset_path /path/to/B1afocal.csv \
        --target_column target_var \
        --category "Suicidal Ideation" \
        --pdf_rag \
        --save_dir results/pbd_experiment \
        --model-type gpt-4o
"""

import sys
import os
import logging
import time
import json
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
    matthews_corrcoef, balanced_accuracy_score
)

# SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    env_file = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

# Try to import constants
try:
    import constants
    HAS_CONSTANTS = True
except (ImportError, ModuleNotFoundError):
    HAS_CONSTANTS = False


# ==================== Logging Setup ====================

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
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.colored_levelname = f"{color}{record.levelname:8}{self.RESET}"
        record.short_time = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        record.location = f"{record.filename}:{record.lineno}"
        return super().format(record)


def setup_logging(log_level: str = "DEBUG", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up comprehensive logging with console and optional file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("pbd_llm_lasso")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # Console formatter (colored)
    console_format = "%(short_time)s | %(colored_levelname)s | %(location)-30s | %(message)s"
    console_formatter = ColoredFormatter(console_format)
    
    # File formatter (no colors, more detail)
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# ==================== Data Processing ====================

def load_dataset(
    dataset_path: str,
    target_column: str,
    feature_names: List[str],
    logger: logging.Logger
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load dataset from CSV and validate against feature names.
    
    Args:
        dataset_path: Path to CSV file
        target_column: Name of target column
        feature_names: List of expected feature names
        logger: Logger instance
    
    Returns:
        Tuple of (X, y, validated_feature_names)
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    start_time = time.time()
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    load_time = time.time() - start_time
    
    logger.info(f"Dataset loaded in {load_time:.2f}s")
    logger.info(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.debug(f"Columns: {list(df.columns)}")
    
    # Validate target column
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataset")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Extract target
    y = df[target_column]
    logger.info(f"Target column: {target_column}")
    logger.info(f"Target distribution: {dict(y.value_counts())}")
    logger.debug(f"Target dtype: {y.dtype}")
    
    # Validate feature names against dataset columns
    dataset_features = [col for col in df.columns if col != target_column]
    logger.debug(f"Dataset features ({len(dataset_features)}): {dataset_features}")
    logger.debug(f"Expected features ({len(feature_names)}): {feature_names}")
    
    # Find matching features
    matching_features = [f for f in feature_names if f in dataset_features]
    missing_features = [f for f in feature_names if f not in dataset_features]
    extra_features = [f for f in dataset_features if f not in feature_names]
    
    logger.info(f"Matching features: {len(matching_features)}/{len(feature_names)}")
    
    if missing_features:
        logger.warning(f"Features in feature file but NOT in dataset ({len(missing_features)}): {missing_features}")
    
    if extra_features:
        logger.warning(f"Features in dataset but NOT in feature file ({len(extra_features)}): {extra_features}")
    
    if not matching_features:
        logger.error("No matching features found between feature file and dataset!")
        raise ValueError("No matching features found between feature file and dataset")
    
    # Use matching features
    X = df[matching_features]
    logger.info(f"Using {len(matching_features)} features for modeling")
    
    return X, y, matching_features


def impute_missing_values(
    X: pd.DataFrame,
    strategy: str,
    logger: logging.Logger
) -> tuple[pd.DataFrame, SimpleImputer]:
    """
    Impute missing values in the dataset.
    
    Args:
        X: Feature DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        logger: Logger instance
    
    Returns:
        Tuple of (imputed DataFrame, fitted imputer)
    """
    logger.info("=" * 60)
    logger.info("MISSING VALUE IMPUTATION")
    logger.info("=" * 60)
    
    # Check for missing values
    missing_counts = X.isnull().sum()
    total_missing = missing_counts.sum()
    
    logger.info(f"Total missing values: {total_missing}")
    logger.info(f"Missing values per column:")
    
    for col in X.columns:
        if missing_counts[col] > 0:
            pct = 100 * missing_counts[col] / len(X)
            logger.info(f"  {col}: {missing_counts[col]} ({pct:.1f}%)")
    
    if total_missing == 0:
        logger.info("No missing values found - skipping imputation")
        return X, None
    
    logger.info(f"Imputation strategy: {strategy}")
    
    start_time = time.time()
    imputer = SimpleImputer(strategy=strategy)
    
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    impute_time = time.time() - start_time
    logger.info(f"Imputation completed in {impute_time:.2f}s")
    
    # Verify no missing values remain
    remaining_missing = X_imputed.isnull().sum().sum()
    if remaining_missing > 0:
        logger.error(f"Imputation failed: {remaining_missing} missing values remain")
        raise RuntimeError("Imputation did not resolve all missing values")
    
    logger.info("All missing values successfully imputed")
    logger.debug(f"Imputed values per feature: {dict(zip(X.columns, imputer.statistics_))}")
    
    return X_imputed, imputer


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 60)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 60)
    
    logger.info(f"Test size: {test_size}")
    logger.info(f"Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.debug(f"Training target distribution: {dict(y_train.value_counts())}")
    logger.debug(f"Test target distribution: {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test


# ==================== PDF RAG Setup ====================

def setup_pdf_vectorstore(
    persist_directory: str,
    collection_name: str,
    logger: logging.Logger
):
    """
    Load PDF vectorstore for RAG.
    
    Args:
        persist_directory: Path to vectorstore directory
        collection_name: Name of the collection
        logger: Logger instance
    
    Returns:
        Tuple of (vectorstore, embeddings)
    """
    logger.info("=" * 60)
    logger.info("PDF VECTORSTORE SETUP")
    logger.info("=" * 60)
    
    logger.info(f"Persist directory: {persist_directory}")
    logger.info(f"Collection name: {collection_name}")
    
    # Check if vectorstore exists
    if not os.path.exists(persist_directory):
        logger.error(f"Vectorstore directory not found: {persist_directory}")
        logger.error("Run 'python playground/interactive_pdf_RAG.py' to create the vectorstore first")
        raise FileNotFoundError(f"Vectorstore directory not found: {persist_directory}")
    
    # Set up API key
    if HAS_CONSTANTS:
        api_key = getattr(constants, 'OPENAI_API', None)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            logger.debug("API key loaded from constants")
    
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        if "OPENAI_API" in os.environ and os.environ["OPENAI_API"]:
            os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]
            logger.debug("API key loaded from OPENAI_API environment variable")
    
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        logger.error("OPENAI_API_KEY not set")
        raise ValueError("OPENAI_API_KEY must be set in environment or constants")
    
    logger.info("Initializing OpenAI embeddings...")
    start_time = time.time()
    
    from langchain_openai import OpenAIEmbeddings
    from llm_lasso.llm_penalty.rag import load_pdf_vectorstore
    
    embeddings = OpenAIEmbeddings()
    embed_time = time.time() - start_time
    logger.info(f"Embeddings initialized in {embed_time:.2f}s")
    
    logger.info("Loading PDF vectorstore...")
    start_time = time.time()
    
    vectorstore = load_pdf_vectorstore(
        persist_directory=persist_directory,
        embedding_model=embeddings,
        collection_name=collection_name
    )
    
    load_time = time.time() - start_time
    logger.info(f"Vectorstore loaded in {load_time:.2f}s")
    
    # Log vectorstore stats
    collection = vectorstore._collection
    doc_count = collection.count()
    logger.info(f"Vectorstore contains {doc_count} document chunks")
    
    return vectorstore, embeddings


# ==================== LLM Penalty Generation ====================

def generate_llm_penalties(
    feature_names: List[str],
    category: str,
    prompt_file: str,
    save_dir: str,
    pdf_vectorstore,
    model_type: str,
    model_name: Optional[str],
    temperature: float,
    top_p: float,
    n_trials: int,
    batch_size: int,
    pdf_rag_num_docs: int,
    wipe: bool,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Generate LLM penalty factors using PDF RAG.
    
    Args:
        feature_names: List of feature names
        category: Category/context for the query
        prompt_file: Path to prompt file
        save_dir: Directory to save results
        pdf_vectorstore: Loaded PDF vectorstore
        model_type: LLM model type
        model_name: Optional specific model name
        temperature: LLM temperature
        top_p: LLM top_p
        n_trials: Number of trials
        batch_size: Batch size for LLM queries
        pdf_rag_num_docs: Number of PDF documents to retrieve
        wipe: Whether to wipe existing results
        logger: Logger instance
    
    Returns:
        Dictionary with penalty scores
    """
    logger.info("=" * 60)
    logger.info("LLM PENALTY GENERATION")
    logger.info("=" * 60)
    
    logger.info(f"Category: {category}")
    logger.info(f"Prompt file: {prompt_file}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-p: {top_p}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"PDF RAG documents: {pdf_rag_num_docs}")
    logger.info(f"Features to process: {len(feature_names)}")
    
    # Validate prompt file
    if not os.path.exists(prompt_file):
        logger.error(f"Prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    logger.debug(f"Feature names: {feature_names}")
    
    # Import required modules
    from llm_lasso.llm_penalty.penalty_collection import collect_penalties, PenaltyCollectionParams
    from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory, LLMType
    
    # Set up LLM
    logger.info("Initializing LLM...")
    
    if model_type == "gpt-4o":
        llm_type = LLMType.GPT4O
        api_key = os.environ.get("OPENAI_API_KEY")
    elif model_type == "o1":
        llm_type = LLMType.O1
        api_key = os.environ.get("OPENAI_API_KEY")
    elif model_type == "o1-pro":
        llm_type = LLMType.O1PRO
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        llm_type = LLMType.OPENROUTER
        api_key = os.environ.get("OPEN_ROUTER", os.environ.get("OPENAI_API_KEY"))
    
    actual_model_name = model_name if model_name else model_type
    logger.info(f"Using LLM: {actual_model_name} (type: {llm_type})")
    
    model = LLMQueryWrapperWithMemory(
        llm_type=llm_type,
        llm_name=actual_model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.0
    )
    
    # Set up penalty collection parameters
    params = PenaltyCollectionParams(
        batch_size=batch_size,
        n_trials=n_trials,
        wipe=wipe,
        pdf_rag=True,
        pdf_rag_num_docs=pdf_rag_num_docs,
        enable_memory=True,
        memory_size=200
    )
    
    logger.info("Starting penalty collection...")
    start_time = time.time()
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    results, all_scores = collect_penalties(
        category=category,
        feature_names=feature_names,
        prompt_file=prompt_file,
        save_dir=save_dir,
        vectorstore=None,  # No OMIM vectorstore
        model=model,
        params=params,
        omim_api_key="",  # Not using OMIM
        pdf_vectorstore=pdf_vectorstore,
        parallel=False,
        n_threads=1
    )
    
    collection_time = time.time() - start_time
    logger.info(f"Penalty collection completed in {collection_time:.2f}s")
    logger.info(f"Collected {len(all_scores)} penalty scores")
    
    # Validate scores
    if len(all_scores) != len(feature_names):
        logger.error(f"Score count mismatch: {len(all_scores)} scores vs {len(feature_names)} features")
        raise ValueError(f"Score count mismatch: {len(all_scores)} vs {len(feature_names)}")
    
    # Log score distribution
    scores_array = np.array(all_scores)
    logger.info(f"Score statistics:")
    logger.info(f"  Min: {scores_array.min():.2f}")
    logger.info(f"  Max: {scores_array.max():.2f}")
    logger.info(f"  Mean: {scores_array.mean():.2f}")
    logger.info(f"  Std: {scores_array.std():.2f}")
    
    # Create score dictionary
    score_dict = {name: score for name, score in zip(feature_names, all_scores)}
    logger.debug(f"Penalty scores: {score_dict}")
    
    # Save scores
    scores_file = os.path.join(save_dir, "penalty_scores.json")
    with open(scores_file, 'w') as f:
        json.dump(score_dict, f, indent=2)
    logger.info(f"Penalty scores saved to: {scores_file}")
    
    return {
        "scores": all_scores,
        "score_dict": score_dict,
        "results": results
    }


# ==================== Lasso Training ====================

# Check if adelie is available
ADELIE_AVAILABLE = False
try:
    from llm_lasso.task_specific_lasso.llm_lasso import llm_lasso_cv, PenaltyType, scale_cols
    import adelie as ad
    from adelie.cv import cv_grpnet
    from adelie import grpnet
    from adelie.diagnostic import predict
    ADELIE_AVAILABLE = True
except ImportError:
    pass


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_dir: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation plots and compute all metrics.
    
    Creates publication-quality figures including:
    - ROC Curve with AUC
    - Precision-Recall Curve with AP
    - Confusion Matrix (counts and normalized)
    - Probability Distribution by Class
    - Calibration Curve
    - Metrics Summary Bar Chart
    
    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities
        save_dir: Directory to save plots
        logger: Logger instance
    
    Returns:
        Dictionary with all computed metrics
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATING EVALUATION PLOTS AND METRICS")
    logger.info("=" * 60)
    
    # Compute predictions at 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    
    # ==================== Compute All Metrics ====================
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Probabilistic metrics
    metrics['auroc'] = roc_auc_score(y_true, y_prob)
    metrics['average_precision'] = average_precision_score(y_true, y_prob)
    metrics['log_loss'] = log_loss(y_true, y_prob)
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Log metrics
    logger.info("Classification Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  Precision (PPV): {metrics['precision']:.4f}")
    logger.info(f"  NPV: {metrics['npv']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  MCC: {metrics['mcc']:.4f}")
    logger.info(f"  AUROC: {metrics['auroc']:.4f}")
    logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    # Define color palette
    colors = {
        'primary': '#2C3E50',
        'secondary': '#E74C3C',
        'accent': '#3498DB',
        'success': '#27AE60',
        'warning': '#F39C12',
        'light': '#ECF0F1',
        'class0': '#3498DB',
        'class1': '#E74C3C'
    }
    
    # ==================== Figure 1: ROC Curve ====================
    logger.info("Generating ROC Curve...")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax1.plot(fpr, tpr, color=colors['secondary'], lw=2.5, 
             label=f'LLM-Lasso (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color=colors['primary'], lw=1.5, 
             linestyle='--', alpha=0.7, label='Random Classifier')
    
    # Fill area under curve
    ax1.fill_between(fpr, tpr, alpha=0.15, color=colors['secondary'])
    
    # Styling
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                marker='o', s=100, c=colors['success'], zorder=5,
                label=f'Optimal (threshold={thresholds[optimal_idx]:.2f})')
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    roc_file = os.path.join(save_dir, 'roc_curve.png')
    fig1.savefig(roc_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    logger.info(f"  Saved: {roc_file}")
    
    # ==================== Figure 2: Precision-Recall Curve ====================
    logger.info("Generating Precision-Recall Curve...")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    # Baseline (proportion of positive class)
    baseline = np.mean(y_true)
    
    ax2.plot(recall, precision, color=colors['accent'], lw=2.5,
             label=f'LLM-Lasso (AP = {ap:.3f})')
    ax2.axhline(y=baseline, color=colors['primary'], lw=1.5, 
                linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')
    
    ax2.fill_between(recall, precision, alpha=0.15, color=colors['accent'])
    
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    pr_file = os.path.join(save_dir, 'precision_recall_curve.png')
    fig2.savefig(pr_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    logger.info(f"  Saved: {pr_file}")
    
    # ==================== Figure 3: Confusion Matrix ====================
    logger.info("Generating Confusion Matrix...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes3[0],
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                annot_kws={'size': 20, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    axes3[0].set_xlabel('Predicted Label')
    axes3[0].set_ylabel('True Label')
    axes3[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes3[1],
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                annot_kws={'size': 18, 'weight': 'bold'},
                cbar_kws={'label': 'Proportion'})
    axes3[1].set_xlabel('Predicted Label')
    axes3[1].set_ylabel('True Label')
    axes3[1].set_title('Confusion Matrix (Normalized by Row)')
    
    plt.tight_layout()
    cm_file = os.path.join(save_dir, 'confusion_matrix.png')
    fig3.savefig(cm_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    logger.info(f"  Saved: {cm_file}")
    
    # ==================== Figure 4: Probability Distribution ====================
    logger.info("Generating Probability Distribution...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Separate probabilities by true class
    prob_class0 = y_prob[y_true == 0]
    prob_class1 = y_prob[y_true == 1]
    
    # Plot histograms
    bins = np.linspace(0, 1, 25)
    ax4.hist(prob_class0, bins=bins, alpha=0.7, label=f'True Negative (n={len(prob_class0)})',
             color=colors['class0'], edgecolor='white', linewidth=1.2)
    ax4.hist(prob_class1, bins=bins, alpha=0.7, label=f'True Positive (n={len(prob_class1)})',
             color=colors['class1'], edgecolor='white', linewidth=1.2)
    
    # Add threshold line
    ax4.axvline(x=0.5, color=colors['primary'], linestyle='--', lw=2, 
                label='Decision Threshold (0.5)')
    
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Predicted Probabilities by True Class')
    ax4.legend(loc='upper center', frameon=True, fancybox=True, shadow=True)
    ax4.set_xlim([-0.02, 1.02])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    dist_file = os.path.join(save_dir, 'probability_distribution.png')
    fig4.savefig(dist_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    logger.info(f"  Saved: {dist_file}")
    
    # ==================== Figure 5: Calibration Curve ====================
    logger.info("Generating Calibration Curve...")
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    
    # Compute calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_proportions = []
    mean_predicted = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() > 0:
            true_proportions.append(y_true[mask].mean())
            mean_predicted.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            true_proportions.append(np.nan)
            mean_predicted.append(np.nan)
            bin_counts.append(0)
    
    true_proportions = np.array(true_proportions)
    mean_predicted = np.array(mean_predicted)
    bin_counts = np.array(bin_counts)
    
    # Plot calibration
    valid = ~np.isnan(true_proportions)
    ax5.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated', alpha=0.7)
    ax5.scatter(mean_predicted[valid], true_proportions[valid], 
                s=bin_counts[valid]*3, c=colors['accent'], alpha=0.7, 
                edgecolors='white', linewidth=1.5)
    ax5.plot(mean_predicted[valid], true_proportions[valid], 
             color=colors['accent'], lw=2, alpha=0.8, label='LLM-Lasso')
    
    ax5.set_xlabel('Mean Predicted Probability')
    ax5.set_ylabel('Fraction of Positives')
    ax5.set_title('Calibration Curve (Reliability Diagram)')
    ax5.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax5.set_xlim([-0.02, 1.02])
    ax5.set_ylim([-0.02, 1.02])
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    cal_file = os.path.join(save_dir, 'calibration_curve.png')
    fig5.savefig(cal_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    logger.info(f"  Saved: {cal_file}")
    
    # ==================== Figure 6: Metrics Summary Bar Chart ====================
    logger.info("Generating Metrics Summary...")
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    
    # Select key metrics for visualization
    metric_names = ['Accuracy', 'Balanced\nAccuracy', 'Sensitivity', 'Specificity', 
                    'Precision', 'F1 Score', 'AUROC', 'Avg\nPrecision']
    metric_values = [
        metrics['accuracy'], metrics['balanced_accuracy'], 
        metrics['sensitivity'], metrics['specificity'],
        metrics['precision'], metrics['f1_score'], 
        metrics['auroc'], metrics['average_precision']
    ]
    
    # Create bars
    bars = ax6.bar(range(len(metric_names)), metric_values, 
                   color=[colors['accent'] if v >= 0.7 else colors['warning'] if v >= 0.5 else colors['secondary'] 
                          for v in metric_values],
                   edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax6.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax6.set_xticks(range(len(metric_names)))
    ax6.set_xticklabels(metric_names, fontsize=11)
    ax6.set_ylabel('Score')
    ax6.set_title('Classification Performance Metrics Summary')
    ax6.set_ylim([0, 1.15])
    ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance Level')
    ax6.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Good Performance')
    ax6.legend(loc='upper right', frameon=True)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_file = os.path.join(save_dir, 'metrics_summary.png')
    fig6.savefig(summary_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig6)
    logger.info(f"  Saved: {summary_file}")
    
    # ==================== Figure 7: Combined Dashboard ====================
    logger.info("Generating Combined Dashboard...")
    fig7 = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig7.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ROC Curve (top-left)
    ax7_1 = fig7.add_subplot(gs[0, 0])
    ax7_1.plot(fpr, tpr, color=colors['secondary'], lw=2.5)
    ax7_1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax7_1.fill_between(fpr, tpr, alpha=0.15, color=colors['secondary'])
    ax7_1.set_xlabel('FPR')
    ax7_1.set_ylabel('TPR')
    ax7_1.set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    ax7_1.set_aspect('equal')
    ax7_1.grid(True, alpha=0.3)
    
    # PR Curve (top-center)
    ax7_2 = fig7.add_subplot(gs[0, 1])
    ax7_2.plot(recall, precision, color=colors['accent'], lw=2.5)
    ax7_2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5)
    ax7_2.fill_between(recall, precision, alpha=0.15, color=colors['accent'])
    ax7_2.set_xlabel('Recall')
    ax7_2.set_ylabel('Precision')
    ax7_2.set_title(f'PR Curve (AP = {ap:.3f})')
    ax7_2.grid(True, alpha=0.3)
    
    # Confusion Matrix (top-right)
    ax7_3 = fig7.add_subplot(gs[0, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7_3,
                xticklabels=['0', '1'], yticklabels=['0', '1'],
                annot_kws={'size': 16, 'weight': 'bold'}, cbar=False)
    ax7_3.set_xlabel('Predicted')
    ax7_3.set_ylabel('True')
    ax7_3.set_title('Confusion Matrix')
    
    # Probability Distribution (bottom-left)
    ax7_4 = fig7.add_subplot(gs[1, 0])
    ax7_4.hist(prob_class0, bins=bins, alpha=0.7, label='Class 0', color=colors['class0'])
    ax7_4.hist(prob_class1, bins=bins, alpha=0.7, label='Class 1', color=colors['class1'])
    ax7_4.axvline(x=0.5, color='k', linestyle='--', lw=1.5)
    ax7_4.set_xlabel('Predicted Probability')
    ax7_4.set_ylabel('Count')
    ax7_4.set_title('Probability Distribution')
    ax7_4.legend()
    
    # Metrics bars (bottom-center and right span)
    ax7_5 = fig7.add_subplot(gs[1, 1:])
    short_names = ['ACC', 'BAL', 'SENS', 'SPEC', 'PREC', 'F1', 'AUROC', 'AP']
    bars = ax7_5.bar(short_names, metric_values,
                     color=[colors['accent'] if v >= 0.7 else colors['warning'] for v in metric_values],
                     edgecolor='white', linewidth=1.5, alpha=0.85)
    for bar, val in zip(bars, metric_values):
        ax7_5.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                      xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    ax7_5.set_ylim([0, 1.1])
    ax7_5.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax7_5.set_ylabel('Score')
    ax7_5.set_title('Performance Metrics')
    ax7_5.grid(True, alpha=0.3, axis='y')
    
    fig7.suptitle('LLM-Lasso LOO Cross-Validation Performance Dashboard', 
                  fontsize=18, fontweight='bold', y=0.98)
    
    dashboard_file = os.path.join(save_dir, 'performance_dashboard.png')
    fig7.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig7)
    logger.info(f"  Saved: {dashboard_file}")
    
    # ==================== Save Metrics to Files ====================
    # Save detailed metrics JSON
    metrics_file = os.path.join(save_dir, 'detailed_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved: {metrics_file}")
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    report_file = os.path.join(save_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("LLM-Lasso Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nAdditional Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"AUROC: {metrics['auroc']:.4f}\n")
        f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        f.write(f"Log Loss: {metrics['log_loss']:.4f}\n")
        f.write(f"Brier Score: {metrics['brier_score']:.4f}\n")
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
    logger.info(f"  Saved: {report_file}")
    
    logger.info("")
    logger.info(f"Generated 7 publication-quality figures in {save_dir}")
    
    return metrics


def run_lasso_with_loo(
    X: pd.DataFrame,
    y: pd.Series,
    penalty_scores: np.ndarray,
    save_dir: str,
    inner_cv_folds: int,
    n_threads: int,
    logger: logging.Logger,
    participant_ids: Optional[pd.Index] = None,
    use_smote: bool = False,
    smote_random_state: int = 42
) -> Dict[str, Any]:
    """
    Run Lasso classification with Leave-One-Out outer CV and inner k-fold CV for hyperparameter selection.
    
    This implements a nested cross-validation scheme:
    - Outer loop: Leave-One-Out (LOO) - each sample is held out once for testing
    - Inner loop: k-fold CV (default 10-fold) for lambda hyperparameter selection
    
    LLM RAG penalty scores are computed ONCE before this function is called,
    then reused for all LOO folds.
    
    SMOTE (if enabled) is applied ONLY to training data within each LOO fold,
    never to test data, ensuring unbiased evaluation.
    
    Args:
        X: Feature matrix (all samples)
        y: Target labels (all samples)
        penalty_scores: LLM-generated penalty scores (computed once)
        save_dir: Directory to save results
        inner_cv_folds: Number of inner CV folds for hyperparameter selection
        n_threads: Number of threads
        logger: Logger instance
        participant_ids: Optional participant IDs (uses index if not provided)
        use_smote: Whether to apply SMOTE to balance training data
        smote_random_state: Random state for SMOTE reproducibility
    
    Returns:
        Dictionary with predictions and evaluation results
    """
    logger.info("=" * 60)
    logger.info("LASSO WITH LEAVE-ONE-OUT CROSS-VALIDATION")
    logger.info("=" * 60)
    
    if not ADELIE_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("ADELIE NOT INSTALLED - SKIPPING LASSO TRAINING")
        logger.warning("=" * 60)
        logger.warning("To enable Lasso training, install adelie:")
        logger.warning("  cd adelie-fork && pip install -e .")
        return {"predictions": None, "summary": {"lasso_skipped": True}}
    
    # Check SMOTE availability
    if use_smote and not SMOTE_AVAILABLE:
        logger.warning("SMOTE requested but imbalanced-learn not installed!")
        logger.warning("Install with: pip install imbalanced-learn")
        logger.warning("Proceeding without SMOTE...")
        use_smote = False
    
    n_samples = len(X)
    class_counts = y.value_counts().sort_index()
    
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Class imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1")
    logger.info(f"Inner CV folds: {inner_cv_folds}")
    logger.info(f"Outer CV: Leave-One-Out ({n_samples} iterations)")
    logger.info(f"Threads: {n_threads}")
    logger.info(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    
    # Use index as participant IDs if not provided
    if participant_ids is None:
        participant_ids = X.index
    
    # Convert penalty factors to importances (inverse relationship)
    penalty_scores_arr = np.array(penalty_scores)
    importances = 1.0 / penalty_scores_arr
    
    # Storage for predictions
    predictions = []
    
    # LOO outer loop
    logger.info("")
    logger.info("Starting LOO cross-validation...")
    start_time = time.time()
    
    for i in range(n_samples):
        # Create LOO split
        test_idx = [i]
        train_idx = [j for j in range(n_samples) if j != i]
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        # Scale features (before SMOTE to maintain consistent scaling)
        train_center = X_train.mean(axis=0)
        train_scale = X_train.std(axis=0)
        X_train_scaled = scale_cols(X_train)
        X_test_scaled = scale_cols(X_test, center=train_center, scale=train_scale)
        
        # Apply SMOTE to balance training data (if enabled)
        if use_smote:
            try:
                smote = SMOTE(random_state=smote_random_state + i, k_neighbors=min(5, y_train.value_counts().min() - 1))
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train_scaled.to_numpy(), 
                    y_train.to_numpy()
                )
            except Exception as e:
                # Fall back to original data if SMOTE fails (e.g., too few samples)
                logger.debug(f"  LOO fold {i+1}: SMOTE failed ({e}), using original data")
                X_train_resampled = X_train_scaled.to_numpy()
                y_train_resampled = y_train.to_numpy()
        else:
            X_train_resampled = X_train_scaled.to_numpy()
            y_train_resampled = y_train.to_numpy()
        
        # Initialize GLM for binomial classification (with possibly resampled data)
        glm_train = ad.glm.binomial(y=y_train_resampled, dtype=np.float64)
        
        # Compute penalty factors from importance scores
        # Using 1/imp^1 as default (can be tuned)
        pf = 1.0 / importances
        pf = pf / np.sum(pf) * X_train_resampled.shape[1]  # Normalize
        
        try:
            # Inner CV for lambda selection (on resampled data if SMOTE enabled)
            fit = cv_grpnet(
                X=X_train_resampled,
                glm=glm_train,
                seed=42 + i,
                n_folds=inner_cv_folds,
                min_ratio=0.01,
                alpha=1.0,  # Pure L1
                penalty=pf,
                n_threads=n_threads,
                progress_bar=False
            )
            
            # Get best lambda index (minimum CV error)
            best_lambda_idx = np.argmin(fit.test_error)
            
            # Train final model with best lambda on all (resampled) training data
            model = grpnet(
                X=X_train_resampled,
                glm=glm_train,
                ddev_tol=0,
                early_exit=False,
                n_threads=n_threads,
                min_ratio=0.01,
                progress_bar=False,
                alpha=1.0,
                penalty=pf,
            )
            
            # Predict on test sample (get linear predictor eta)
            # Note: test data is NEVER resampled - only training data
            etas = predict(
                X=X_test_scaled.to_numpy(),
                betas=model.betas,
                intercepts=model.intercepts,
                n_threads=n_threads,
            )
            
            # Convert eta to probability using sigmoid for binomial
            # eta is shape (n_lambdas, n_samples), use best lambda
            eta_best = etas[best_lambda_idx, 0]
            prob = 1.0 / (1.0 + np.exp(-eta_best))
            
        except Exception as e:
            logger.warning(f"  LOO fold {i+1}/{n_samples}: Error - {e}, using 0.5 probability")
            prob = 0.5
        
        # Store prediction
        predictions.append({
            'participant_id': participant_ids[i],
            'actual_label': int(y_test.iloc[0]),
            'predicted_probability': float(prob)
        })
        
        # Progress logging every 10%
        if (i + 1) % max(1, n_samples // 10) == 0 or i == n_samples - 1:
            logger.info(f"  LOO progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
    
    loo_time = time.time() - start_time
    logger.info(f"LOO cross-validation completed in {loo_time:.2f}s")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Save predictions CSV
    predictions_file = os.path.join(save_dir, "loo_predictions.csv")
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Extract arrays for evaluation
    y_true = predictions_df['actual_label'].values
    y_prob = predictions_df['predicted_probability'].values
    
    # Generate comprehensive evaluation plots and compute all metrics
    metrics = generate_evaluation_plots(y_true, y_prob, save_dir, logger)
    
    # Create summary with all metrics
    class_counts = y.value_counts().sort_index()
    summary = {
        "cv_method": "Leave-One-Out",
        "inner_cv_folds": inner_cv_folds,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_imbalance_ratio": float(class_counts.max() / class_counts.min()),
        "smote_enabled": use_smote,
        "loo_time_seconds": float(loo_time),
        **metrics  # Include all computed metrics
    }
    
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    # Log final results
    logger.info("")
    logger.info("=" * 60)
    logger.info("LOO CROSS-VALIDATION FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({int(metrics['accuracy'] * n_samples)}/{n_samples} correct)")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"MCC: {metrics['mcc']:.4f}")
    
    return {
        "predictions": predictions_df,
        "summary": summary
    }


def run_lasso_with_penalties(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    penalty_scores: np.ndarray,
    save_dir: str,
    n_threads: int,
    folds_cv: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run Lasso classification with LLM-generated penalty factors.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        penalty_scores: LLM-generated penalty scores
        save_dir: Directory to save results
        n_threads: Number of threads
        folds_cv: Number of CV folds
        logger: Logger instance
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("LASSO CLASSIFICATION")
    logger.info("=" * 60)
    
    if not ADELIE_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("ADELIE NOT INSTALLED - SKIPPING LASSO TRAINING")
        logger.warning("=" * 60)
        logger.warning("To enable Lasso training, install adelie:")
        logger.warning("  cd adelie-fork && pip install -e .")
        logger.warning("")
        logger.warning("Penalty scores have been generated and saved.")
        logger.warning("You can run Lasso training separately after installing adelie.")
        
        # Save a summary without Lasso results
        summary = {
            "lasso_skipped": True,
            "reason": "adelie not installed",
            "total_features": X_train.shape[1],
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "penalty_scores_generated": True
        }
        
        summary_file = os.path.join(save_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
        
        return {
            "results": None,
            "summary": summary
        }
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"CV folds: {folds_cv}")
    logger.info(f"Threads: {n_threads}")
    
    from llm_lasso.task_specific_lasso.llm_lasso import llm_lasso_cv, PenaltyType
    
    logger.info("Running LLM-Lasso CV...")
    start_time = time.time()
    
    results = llm_lasso_cv(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        score=np.array(penalty_scores),
        regression=False,  # Classification
        score_type=PenaltyType.PF,  # Penalty factors
        folds_cv=folds_cv,
        seed=42,
        n_threads=n_threads,
        alpha=1.0,  # Pure L1
        max_imp_pow=5
    )
    
    lasso_time = time.time() - start_time
    logger.info(f"Lasso training completed in {lasso_time:.2f}s")
    
    # Log results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Results shape: {results.shape}")
    logger.debug(f"Results columns: {list(results.columns)}")
    
    # Find best result
    best_idx = results['test_error'].idxmin()
    best_result = results.loc[best_idx]
    
    logger.info(f"Best Model:")
    logger.info(f"  Method: {best_result.get('method', 'N/A')}")
    logger.info(f"  Best method/model: {best_result.get('best_method_model', 'N/A')}")
    logger.info(f"  Test error: {best_result['test_error']:.4f}")
    logger.info(f"  AUROC: {best_result.get('auroc', 'N/A')}")
    logger.info(f"  Number of features: {best_result['n_features']}")
    
    # Save results
    results_file = os.path.join(save_dir, "lasso_results.csv")
    results.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Save summary
    summary = {
        "best_method": str(best_result.get('method', 'N/A')),
        "best_method_model": str(best_result.get('best_method_model', 'N/A')),
        "test_error": float(best_result['test_error']),
        "auroc": float(best_result.get('auroc', 0)) if best_result.get('auroc') is not None else None,
        "n_features": int(best_result['n_features']),
        "total_features": X_train.shape[1],
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    return {
        "results": results,
        "summary": summary
    }


# ==================== Main Pipeline ====================

@dataclass
class PipelineArguments:
    """Arguments for the PBD LLM-Lasso pipeline."""
    
    prompt_filename: str = field(metadata={
        "help": "Path to the prompt file"
    })
    feature_names_path: str = field(metadata={
        "help": "Path to the file containing feature names (.pkl or .txt)"
    })
    dataset_path: str = field(metadata={
        "help": "Path to the CSV dataset file"
    })
    target_column: str = field(metadata={
        "help": "Name of the target column in the dataset"
    })
    category: str = field(metadata={
        "help": "Category/context for the LLM query (e.g., 'Suicidal Ideation')"
    })
    save_dir: str = field(default=None, metadata={
        "help": "Directory to save results. If not specified, saves to same directory as dataset file."
    })
    
    # PDF RAG options
    pdf_rag: bool = field(default=True, metadata={
        "help": "Enable PDF RAG for context retrieval"
    })
    pdf_persist_directory: str = field(default="pdf_vectorstore", metadata={
        "help": "Path to PDF vectorstore directory"
    })
    pdf_collection_name: str = field(default="scientific_papers", metadata={
        "help": "Name of the PDF vectorstore collection"
    })
    pdf_rag_num_docs: int = field(default=3, metadata={
        "help": "Number of PDF documents to retrieve for RAG"
    })
    
    # LLM options
    model_type: str = field(default="gpt-4o", metadata={
        "help": "LLM model type: gpt-4o, o1, o1-pro, openrouter",
        "choices": ["gpt-4o", "o1", "o1-pro", "openrouter"]
    })
    model_name: Optional[str] = field(default=None, metadata={
        "help": "Specific model name (optional)"
    })
    temp: float = field(default=0.0, metadata={
        "help": "LLM temperature"
    })
    top_p: float = field(default=0.9, metadata={
        "help": "LLM top-p sampling parameter"
    })
    
    # Data options
    test_size: float = field(default=0.3, metadata={
        "help": "Proportion of data for test set"
    })
    imputation_strategy: str = field(default="median", metadata={
        "help": "Imputation strategy: mean, median, most_frequent",
        "choices": ["mean", "median", "most_frequent"]
    })
    random_state: int = field(default=42, metadata={
        "help": "Random seed for reproducibility"
    })
    
    # Training options
    n_trials: int = field(default=1, metadata={
        "help": "Number of trials for penalty generation"
    })
    batch_size: int = field(default=10, metadata={
        "help": "Batch size for LLM queries"
    })
    folds_cv: int = field(default=5, metadata={
        "help": "Number of cross-validation folds for hyperparameter selection"
    })
    n_threads: int = field(default=4, metadata={
        "help": "Number of threads for Lasso training"
    })
    wipe: bool = field(default=False, metadata={
        "help": "Wipe existing results before starting"
    })
    
    # Nested CV options
    use_loo: bool = field(default=False, metadata={
        "help": "Use Leave-One-Out cross-validation for outer testing loop"
    })
    inner_cv_folds: int = field(default=10, metadata={
        "help": "Number of inner CV folds for hyperparameter selection (used with --use_loo)"
    })
    
    # Class imbalance handling
    use_smote: bool = field(default=False, metadata={
        "help": "Use SMOTE to balance classes in training data (applied within each CV fold)"
    })
    smote_random_state: int = field(default=42, metadata={
        "help": "Random state for SMOTE reproducibility"
    })
    
    # Logging options
    log_level: str = field(default="DEBUG", metadata={
        "help": "Logging level: DEBUG, INFO, WARNING, ERROR"
    })
    log_file: Optional[str] = field(default=None, metadata={
        "help": "Path to log file (optional)"
    })


def main():
    """Main entry point for the PBD LLM-Lasso pipeline."""
    
    from transformers import HfArgumentParser
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Parse arguments
    parser = HfArgumentParser([PipelineArguments])
    args = parser.parse_args_into_dataclasses()[0]
    
    # Default save_dir to the same directory as the dataset file
    if args.save_dir is None:
        args.save_dir = os.path.dirname(os.path.abspath(args.dataset_path))
    
    # Set up logging
    log_file = args.log_file
    if log_file is None:
        # Default log file in save_dir
        os.makedirs(args.save_dir, exist_ok=True)
        log_file = os.path.join(args.save_dir, "pipeline.log")
    
    logger = setup_logging(args.log_level, log_file)
    
    # Log pipeline start
    logger.info("=" * 80)
    logger.info("PBD LLM-LASSO PIPELINE WITH PDF RAG")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("")
    
    # Log all arguments
    logger.info("Pipeline Arguments:")
    for field_name, field_value in vars(args).items():
        logger.info(f"  {field_name}: {field_value}")
    logger.info("")
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Load feature names
        logger.info("=" * 60)
        logger.info("STEP 1: LOAD FEATURE NAMES")
        logger.info("=" * 60)
        
        if args.feature_names_path.endswith(".pkl"):
            with open(args.feature_names_path, 'rb') as f:
                feature_names = pkl.load(f)
        elif args.feature_names_path.endswith(".txt"):
            with open(args.feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Feature names file must be .pkl or .txt")
        
        logger.info(f"Loaded {len(feature_names)} feature names from {args.feature_names_path}")
        logger.debug(f"Feature names: {feature_names}")
        
        # Step 2: Load and validate dataset
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: LOAD AND VALIDATE DATASET")
        logger.info("=" * 60)
        
        X, y, validated_features = load_dataset(
            args.dataset_path,
            args.target_column,
            feature_names,
            logger
        )
        
        # Step 3: Impute missing values
        logger.info("")
        X_imputed, imputer = impute_missing_values(X, args.imputation_strategy, logger)
        
        # Step 4: Split data (skip if using LOO)
        logger.info("")
        if args.use_loo:
            logger.info("=" * 60)
            logger.info("USING LEAVE-ONE-OUT CROSS-VALIDATION")
            logger.info("=" * 60)
            logger.info("Skipping train/test split - LOO will use all samples")
            logger.info(f"Outer CV: Leave-One-Out ({len(X_imputed)} iterations)")
            logger.info(f"Inner CV: {args.inner_cv_folds}-fold for hyperparameter selection")
            X_train, X_test, y_train, y_test = X_imputed, None, y, None
        else:
            X_train, X_test, y_train, y_test = split_data(
                X_imputed, y,
                args.test_size,
                args.random_state,
                logger
            )
        
        # Step 5: Set up PDF vectorstore
        logger.info("")
        if args.pdf_rag:
            pdf_vectorstore, embeddings = setup_pdf_vectorstore(
                args.pdf_persist_directory,
                args.pdf_collection_name,
                logger
            )
        else:
            pdf_vectorstore = None
            logger.info("PDF RAG disabled - using LLM without retrieval")
        
        # Step 6: Generate LLM penalties (done ONCE, before any CV)
        logger.info("")
        penalty_result = generate_llm_penalties(
            feature_names=validated_features,
            category=args.category,
            prompt_file=args.prompt_filename,
            save_dir=args.save_dir,
            pdf_vectorstore=pdf_vectorstore,
            model_type=args.model_type,
            model_name=args.model_name,
            temperature=args.temp,
            top_p=args.top_p,
            n_trials=args.n_trials,
            batch_size=args.batch_size,
            pdf_rag_num_docs=args.pdf_rag_num_docs,
            wipe=args.wipe,
            logger=logger
        )
        
        # Step 7: Run Lasso with penalties
        logger.info("")
        if args.use_loo:
            # Use Leave-One-Out cross-validation
            lasso_result = run_lasso_with_loo(
                X=X_imputed,
                y=y,
                penalty_scores=penalty_result["scores"],
                save_dir=args.save_dir,
                inner_cv_folds=args.inner_cv_folds,
                n_threads=args.n_threads,
                logger=logger,
                participant_ids=X_imputed.index,
                use_smote=args.use_smote,
                smote_random_state=args.smote_random_state
            )
        else:
            # Use standard train/test split
            lasso_result = run_lasso_with_penalties(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                penalty_scores=penalty_result["scores"],
                save_dir=args.save_dir,
                n_threads=args.n_threads,
                folds_cv=args.folds_cv,
                logger=logger
            )
        
        # Pipeline complete
        pipeline_time = time.time() - pipeline_start
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total pipeline time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - {os.path.join(args.save_dir, 'penalty_scores.json')}")
        if args.use_loo:
            logger.info(f"  - {os.path.join(args.save_dir, 'loo_predictions.csv')}")
        elif lasso_result.get('results') is not None:
            logger.info(f"  - {os.path.join(args.save_dir, 'lasso_results.csv')}")
        logger.info(f"  - {os.path.join(args.save_dir, 'summary.json')}")
        logger.info(f"  - {log_file}")
        logger.info("")
        
        if lasso_result['summary'].get('lasso_skipped'):
            logger.info("Penalty scores generated successfully!")
            logger.info("Lasso training was skipped (adelie not installed)")
            logger.info("To run Lasso, install adelie: cd adelie-fork && pip install -e .")
        elif args.use_loo:
            summary = lasso_result['summary']
            logger.info(f"LOO Cross-Validation Results:")
            logger.info(f"  AUROC: {summary['auroc']:.4f}")
            logger.info(f"  Accuracy: {summary['accuracy']:.4f} ({summary['true_positives'] + summary['true_negatives']}/{summary['n_samples']} correct)")
            logger.info(f"  Balanced Accuracy: {summary['balanced_accuracy']:.4f}")
            logger.info(f"  Sensitivity: {summary['sensitivity']:.4f}")
            logger.info(f"  Specificity: {summary['specificity']:.4f}")
            logger.info(f"  F1 Score: {summary['f1_score']:.4f}")
            logger.info(f"  MCC: {summary['mcc']:.4f}")
            logger.info(f"  Log Loss: {summary['log_loss']:.4f}")
            logger.info(f"  Brier Score: {summary['brier_score']:.4f}")
            logger.info(f"")
            logger.info(f"Generated files:")
            logger.info(f"  - loo_predictions.csv (per-sample predictions)")
            logger.info(f"  - roc_curve.png")
            logger.info(f"  - precision_recall_curve.png") 
            logger.info(f"  - confusion_matrix.png")
            logger.info(f"  - probability_distribution.png")
            logger.info(f"  - calibration_curve.png")
            logger.info(f"  - metrics_summary.png")
            logger.info(f"  - performance_dashboard.png")
            logger.info(f"  - detailed_metrics.json")
            logger.info(f"  - classification_report.txt")
        else:
            logger.info(f"Best model test error: {lasso_result['summary']['test_error']:.4f}")
            logger.info(f"Best model AUROC: {lasso_result['summary']['auroc']}")
            logger.info(f"Features selected: {lasso_result['summary']['n_features']}/{lasso_result['summary']['total_features']}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
