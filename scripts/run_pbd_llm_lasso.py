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


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that may contain numpy types
    
    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Try to convert to string as fallback
        try:
            return str(obj)
        except:
            return repr(obj)


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


def validate_loo_data(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    dataset_path: str,
    logger: logging.Logger,
) -> None:
    """
    Validate data for LOO pipeline. On failure, prints exact row/column locations
    to the console and exits with code 1. Checks:
    1) No NaN in target column (reports exact row indices and index values).
    2) No Inf in target column (reports exact rows).
    3) Target has exactly 2 unique classes for binary classification (reports unexpected values/rows).
    """
    has_error = False
    # Use a list of (title, lines) to print once at the end
    error_sections = []

    # ----- 1. Check for NaN in target -----
    nan_mask = y.isna()
    if nan_mask.any():
        has_error = True
        nan_positions = np.where(nan_mask)[0].tolist()
        lines = [
            "",
            "  ISSUE: Missing values (NaN) in the target column.",
            f"  Column name: '{target_column}'",
            f"  Dataset: {dataset_path}",
            f"  Total rows with NaN: {nan_mask.sum()}",
            "",
            "  Rows with NaN (0-based row number = position in CSV, 1-based = line number in file):",
        ]
        for pos in nan_positions:
            index_val = y.index[pos] if pos < len(y.index) else "?"
            lines.append(f"    - 0-based row: {pos}  |  1-based row: {pos + 1}  |  DataFrame index: {index_val!r}  |  value: NaN")
        lines.append("")
        lines.append("  FIX: Remove these rows from the dataset, or impute the target before running.")
        error_sections.append(("MISSING VALUES IN TARGET COLUMN", lines))

    # ----- 2. Check for Inf in target -----
    try:
        y_numeric = pd.to_numeric(y, errors="coerce")
        inf_mask = np.isinf(y_numeric)
        if inf_mask.any():
            has_error = True
            inf_positions = np.where(inf_mask)[0].tolist()
            lines = [
                "",
                "  ISSUE: Infinite values (Inf/-Inf) in the target column.",
                f"  Column name: '{target_column}'",
                f"  Total rows with Inf: {inf_mask.sum()}",
                "",
                "  Rows with Inf:",
            ]
            for pos in inf_positions:
                index_val = y.index[pos] if pos < len(y.index) else "?"
                val = y.iloc[pos]
                lines.append(f"    - 0-based row: {pos}  |  1-based row: {pos + 1}  |  DataFrame index: {index_val!r}  |  value: {val}")
            lines.append("")
            lines.append("  FIX: Replace or remove these rows so the target has only finite values.")
            error_sections.append(("INFINITE VALUES IN TARGET COLUMN", lines))
    except Exception:
        pass

    # ----- 3. Check binary target (exactly 2 classes) -----
    y_valid = y.dropna()
    unique_vals = np.unique(y_valid.astype(str))
    if len(unique_vals) != 2:
        has_error = True
        lines = [
            "",
            "  ISSUE: Target must have exactly 2 classes for binary classification.",
            f"  Column name: '{target_column}'",
            f"  Number of unique values (after dropping NaN): {len(unique_vals)}",
            f"  Unique values: {list(unique_vals)}",
            "",
            "  Value counts:",
        ]
        for v, cnt in y_valid.value_counts().items():
            lines.append(f"    - {v!r}: {cnt} rows")
        lines.append("")
        lines.append("  FIX: Ensure the target column has exactly two distinct values (e.g. 0/1 or Yes/No).")
        error_sections.append(("TARGET NOT BINARY", lines))

    if not has_error:
        return

    # ----- Print to console and log, then exit -----
    sep = "=" * 70
    header = "DATA VALIDATION FAILED — fix the issues below and re-run"
    full_lines = [sep, header, sep]
    for title, section_lines in error_sections:
        full_lines.append("")
        full_lines.append(f"  [{title}]")
        full_lines.extend(section_lines)
    full_lines.append("")
    full_lines.append(sep)
    full_lines.append("Exiting with code 1.")
    full_lines.append(sep)
    msg = "\n".join(full_lines)

    # Ensure it appears on console
    print(msg, file=sys.stderr)
    for line in full_lines:
        logger.error(line)
    logger.error("Validation failed. Exiting.")
    sys.exit(1)


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
    logger: logging.Logger,
    llm_backend: str = "openai"
):
    """
    Load PDF vectorstore for RAG.
    
    Args:
        persist_directory: Path to vectorstore directory
        collection_name: Name of the collection
        logger: Logger instance
        llm_backend: LLM backend to use ("openai" or "vllm")
    
    Returns:
        Tuple of (vectorstore, embeddings)
    """
    logger.info("=" * 60)
    logger.info("PDF VECTORSTORE SETUP")
    logger.info("=" * 60)
    
    logger.info(f"Persist directory: {persist_directory}")
    logger.info(f"Collection name: {collection_name}")
    logger.info(f"LLM backend: {llm_backend}")
    
    # Check if vectorstore exists
    if not os.path.exists(persist_directory):
        logger.error(f"Vectorstore directory not found: {persist_directory}")
        logger.error("Create the vectorstore first using:")
        logger.error(f"  python scripts/index_pdf_vectorstore.py --pdf-directory /path/to/pdfs --persist-directory {persist_directory} --embedding-backend {llm_backend}")
        raise FileNotFoundError(f"Vectorstore directory not found: {persist_directory}")
    
    from llm_lasso.llm_penalty.rag import load_pdf_vectorstore
    from llm_lasso.llm_penalty.embeddings import get_embeddings
    
    if llm_backend == "vllm":
        # vLLM backend - uses local vLLM server
        logger.info("Initializing vLLM embeddings...")
        
        # Check for vLLM configuration
        # Priority: EMBED_BASE_URL > VLLM_EMBED_BASE_URL > default
        vllm_embed_url = os.environ.get(
            "EMBED_BASE_URL",
            os.environ.get("VLLM_EMBED_BASE_URL", "http://localhost:8001/v1")
        )
        vllm_model = os.environ.get("VLLM_EMBED_MODEL", "qwen3-embed")
        
        # Log which env var was used
        if "EMBED_BASE_URL" in os.environ:
            logger.info(f"vLLM Embed URL: {vllm_embed_url} (from EMBED_BASE_URL)")
        elif "VLLM_EMBED_BASE_URL" in os.environ:
            logger.info(f"vLLM Embed URL: {vllm_embed_url} (from VLLM_EMBED_BASE_URL)")
        else:
            logger.info(f"vLLM Embed URL: {vllm_embed_url} (default)")
        logger.info(f"vLLM Embed Model: {vllm_model}")
        
        start_time = time.time()
        embeddings = get_embeddings(backend="vllm")
        embed_time = time.time() - start_time
        logger.info(f"vLLM embeddings initialized in {embed_time:.2f}s")
    else:
        # OpenAI backend (default)
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
        
        embeddings = get_embeddings(backend="openai")
        embed_time = time.time() - start_time
        logger.info(f"OpenAI embeddings initialized in {embed_time:.2f}s")
    
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
    logger: logging.Logger,
    llm_backend: str = "openai"
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
        llm_backend: LLM backend to use ("openai" or "vllm")
    
    Returns:
        Dictionary with penalty scores
    """
    logger.info("=" * 60)
    logger.info("LLM PENALTY GENERATION")
    logger.info("=" * 60)
    
    logger.info(f"Category: {category}")
    logger.info(f"Prompt file: {prompt_file}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"LLM backend: {llm_backend}")
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
    
    # Set up LLM based on backend
    logger.info("Initializing LLM...")
    
    if llm_backend == "vllm":
        # vLLM backend - local open-source models
        llm_type = LLMType.VLLM
        api_key = os.environ.get("VLLM_API_KEY", "")
        
        # Get model name from env or argument
        if model_name:
            actual_model_name = model_name
        else:
            actual_model_name = os.environ.get("VLLM_CHAT_MODEL", "qwen3-thinking")
        
        # Check multiple environment variable names for flexibility
        # Priority: CHAT_BASE_URL > VLLM_CHAT_BASE_URL > default
        vllm_url = os.environ.get(
            "CHAT_BASE_URL",
            os.environ.get("VLLM_CHAT_BASE_URL", "http://localhost:8000/v1")
        )
        
        # Log which env var was used
        if "CHAT_BASE_URL" in os.environ:
            logger.info(f"vLLM Chat URL: {vllm_url} (from CHAT_BASE_URL)")
        elif "VLLM_CHAT_BASE_URL" in os.environ:
            logger.info(f"vLLM Chat URL: {vllm_url} (from VLLM_CHAT_BASE_URL)")
        else:
            logger.info(f"vLLM Chat URL: {vllm_url} (default)")
        logger.info(f"vLLM Chat Model: {actual_model_name}")
    else:
        # OpenAI backend (default)
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
    
    logger.info(f"Using LLM: {actual_model_name} (type: {llm_type}, backend: {llm_backend})")
    
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
    
    # Save detailed LLM reasoning log
    reasoning_log_file = os.path.join(save_dir, "penalty_llm_reasoning.log")
    with open(reasoning_log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM PENALTY FACTOR REASONING LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LLM Backend: {llm_backend}\n")
        f.write(f"Model: {actual_model_name}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Prompt file: {prompt_file}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("PENALTY FACTOR SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Feature':<45} {'Penalty':>8}\n")
        f.write("-" * 60 + "\n")
        for name, score in score_dict.items():
            f.write(f"{name:<45} {score:>8.1f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Mean':<45} {scores_array.mean():>8.2f}\n")
        f.write(f"{'Min':<45} {scores_array.min():>8.2f}\n")
        f.write(f"{'Max':<45} {scores_array.max():>8.2f}\n")
        f.write(f"{'Std':<45} {scores_array.std():>8.2f}\n")
        f.write("\n")
        
        # Full LLM responses with reasoning
        f.write("=" * 80 + "\n")
        f.write("DETAILED LLM REASONING\n")
        f.write("=" * 80 + "\n\n")
        
        for i, response in enumerate(results):
            f.write(f"--- BATCH {i + 1} ---\n")
            f.write(response + "\n")
            f.write("\n" + "-" * 40 + "\n\n")
    
    logger.info(f"LLM reasoning log saved to: {reasoning_log_file}")
    
    return {
        "scores": all_scores,
        "score_dict": score_dict,
        "results": results
    }


def save_rag_retrieved_documents(
    feature_names: List[str],
    category: str,
    pdf_vectorstore,
    pdf_rag_num_docs: int,
    save_dir: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Retrieve and save all RAG documents with their sources to a JSON file.
    This save is MANDATORY when PDF RAG is enabled so that retrieved documents
    are always persisted to rag_retrieved_documents.json.

    Args:
        feature_names: List of feature names used for queries
        category: Category/domain context
        pdf_vectorstore: ChromaDB vectorstore with PDF embeddings
        pdf_rag_num_docs: Number of documents to retrieve per query
        save_dir: Directory to save the output
        logger: Logger instance

    Returns:
        Dictionary with retrieved documents information
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING RAG RETRIEVED DOCUMENTS")
    logger.info("=" * 60)

    if pdf_vectorstore is None:
        logger.warning("No PDF vectorstore available - skipping RAG document saving")
        return {}

    # Log collection size so we can see if we're hitting an empty collection (e.g. wrong collection name)
    try:
        coll = pdf_vectorstore._collection
        chunk_count = coll.count()
        logger.info(f"RAG vectorstore collection has {chunk_count} document chunks (save path: {os.path.abspath(save_dir)})")
        if chunk_count == 0:
            logger.warning(
                "RAG vectorstore has 0 chunks - rag_retrieved_documents.json will be empty. "
                "Ensure --pdf_persist_directory and --pdf_collection_name match how you indexed "
                "(index_pdf_vectorstore.py uses --collection-name pdf_documents by default)."
            )
    except Exception as e:
        logger.warning(f"Could not get vectorstore chunk count: {e}")

    retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": pdf_rag_num_docs})

    rag_documents = {
        "metadata": {
            "category": category,
            "num_features_queried": len(feature_names),
            "docs_per_query": pdf_rag_num_docs,
            "timestamp": datetime.now().isoformat()
        },
        "queries": {},
        "all_documents": []
    }

    seen_contents = set()
    doc_id = 1

    num_with_docs = 0
    for idx, feature in enumerate(feature_names):
        query = f"Information about {feature} related to {category}"
        try:
            docs = retriever.get_relevant_documents(query)[:pdf_rag_num_docs]
            if docs:
                num_with_docs += 1
            if idx < 3:
                logger.info(f"  Feature '{feature[:50]}...' retrieved {len(docs)} doc(s)")
            feature_docs = []
            for doc in docs:
                # LangChain Document: page_content is the main text; fallback for compatibility
                page_content = getattr(doc, "page_content", None) or getattr(doc, "content", "") or ""
                content_preview = page_content[:500] if len(page_content) > 500 else page_content
                content_hash = hash(content_preview)

                doc_info = {
                    "doc_id": f"doc_{doc_id}",
                    "source_file": doc.metadata.get("filename", doc.metadata.get("source", "Unknown")),
                    "page": doc.metadata.get("page", "N/A"),
                    "chunk_index": doc.metadata.get("chunk_index", "N/A"),
                    "content_preview": content_preview + ("..." if len(page_content) > 500 else ""),
                    "full_content": page_content,
                    "content_length": len(page_content),
                    "metadata": {k: str(v) for k, v in doc.metadata.items()}
                }

                feature_docs.append(doc_info)

                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    rag_documents["all_documents"].append(doc_info)
                    doc_id += 1

            rag_documents["queries"][feature] = {
                "query": query,
                "num_docs_retrieved": len(feature_docs),
                "documents": feature_docs
            }

        except Exception as e:
            if idx < 3:
                logger.warning(f"  Feature '{feature[:50]}...' retrieval failed: {e}")
            logger.warning(f"Error retrieving docs for {feature}: {e}")
            rag_documents["queries"][feature] = {
                "query": query,
                "num_docs_retrieved": 0,
                "documents": [],
                "error": str(e)
            }

    # Summary statistics
    rag_documents["summary"] = {
        "total_unique_documents": len(rag_documents["all_documents"]),
        "total_queries": len(feature_names),
        "successful_queries": sum(1 for q in rag_documents["queries"].values() if q["num_docs_retrieved"] > 0),
        "unique_source_files": list(set(
            d["source_file"] for d in rag_documents["all_documents"]
        ))
    }

    # MANDATORY: Always write RAG retrieved documents to disk (atomic write + sync)
    os.makedirs(save_dir, exist_ok=True)
    rag_file = os.path.join(save_dir, "rag_retrieved_documents.json")
    rag_file_tmp = rag_file + ".tmp"
    try:
        with open(rag_file_tmp, "w", encoding="utf-8") as f:
            json.dump(rag_documents, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(rag_file_tmp, rag_file)
    except Exception as e:
        logger.error(f"Failed to save RAG retrieved documents to {rag_file}: {e}")
        raise
    finally:
        if os.path.exists(rag_file_tmp):
            try:
                os.remove(rag_file_tmp)
            except OSError:
                pass

    logger.info(f"RAG retrieval summary: {num_with_docs}/{len(feature_names)} features had at least 1 doc")
    logger.info(f"Total unique documents retrieved: {rag_documents['summary']['total_unique_documents']}")
    logger.info(f"Unique source files: {len(rag_documents['summary']['unique_source_files'])}")
    for src in rag_documents['summary']['unique_source_files'][:5]:  # Show first 5
        logger.info(f"  - {src}")
    if len(rag_documents['summary']['unique_source_files']) > 5:
        logger.info(f"  ... and {len(rag_documents['summary']['unique_source_files']) - 5} more")
    logger.info(f"RAG documents saved to: {rag_file}")
    
    return rag_documents


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


def generate_coefficient_plot(
    feature_names: List[str],
    coefficients: np.ndarray,
    mean_coefficients: np.ndarray,
    std_coefficients: np.ndarray,
    selection_frequency: np.ndarray,
    save_dir: str,
    logger: logging.Logger
) -> None:
    """
    Generate publication-quality bar plots of Lasso coefficients.
    
    Creates multiple visualizations:
    1. Final model coefficients (non-zero only) - sorted by signed value
    2. All coefficients with selection frequency
    3. Coefficient stability plot (mean ± std from LOO)
    
    Args:
        feature_names: List of feature names
        coefficients: Final model coefficients
        mean_coefficients: Mean coefficients from LOO folds
        std_coefficients: Std of coefficients from LOO folds
        selection_frequency: Proportion of folds where each feature was selected
        save_dir: Directory to save plots
        logger: Logger instance
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    
    logger.info("")
    logger.info("Generating coefficient plots...")
    
    # Publication-quality settings - enhanced for high quality
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Create DataFrame for easier manipulation
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Mean_Coefficient': mean_coefficients,
        'Std_Coefficient': std_coefficients,
        'Selection_Frequency': selection_frequency
    })
    
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    
    # =========================================================================
    # PLOT 1: Non-zero coefficients bar plot - SORTED BY SIGNED VALUE
    # =========================================================================
    nonzero_df = coef_df[coef_df['Coefficient'] != 0].copy()
    # Sort by signed coefficient value (ascending: most negative at bottom, most positive at top)
    nonzero_df = nonzero_df.sort_values('Coefficient', ascending=True)
    
    if len(nonzero_df) > 0:
        n_features = len(nonzero_df)
        fig_height = max(6, n_features * 0.45)  # More space per bar
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # Color by sign
        colors = ['#E74C3C' if c < 0 else '#27AE60' for c in nonzero_df['Coefficient']]
        
        y_positions = range(n_features)
        bars = ax.barh(
            y=y_positions,
            width=nonzero_df['Coefficient'],
            color=colors,
            edgecolor='#2C3E50',
            linewidth=0.8,
            alpha=0.9,
            height=0.7
        )
        
        # Calculate x-axis limits with extra padding for labels
        coef_min = nonzero_df['Coefficient'].min()
        coef_max = nonzero_df['Coefficient'].max()
        coef_range = coef_max - coef_min
        
        # Add substantial padding for labels (30% on each side)
        x_min = coef_min - 0.35 * abs(coef_min) if coef_min < 0 else -0.1 * coef_range
        x_max = coef_max + 0.35 * abs(coef_max) if coef_max > 0 else 0.1 * coef_range
        ax.set_xlim(x_min, x_max)
        
        # Add value labels OUTSIDE the bars with proper positioning
        for i, (idx, row) in enumerate(nonzero_df.iterrows()):
            val = row['Coefficient']
            if val >= 0:
                # Positive: label to the right of bar
                label_x = val + 0.03 * coef_range
                ha = 'left'
            else:
                # Negative: label to the left of bar
                label_x = val - 0.03 * coef_range
                ha = 'right'
            
            ax.text(label_x, i, f'{val:.3f}', 
                    va='center', ha=ha, fontsize=9, fontweight='bold',
                    color='#2C3E50')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(nonzero_df['Feature'], fontsize=10)
        ax.axvline(x=0, color='#2C3E50', linewidth=1.5, linestyle='-', zorder=1)
        ax.set_xlabel('Coefficient Value', fontweight='bold', fontsize=12)
        ax.set_title('Lasso Coefficients (Non-Zero Features)\nSorted by Coefficient Value', 
                     fontweight='bold', fontsize=14, pad=15)
        
        # Add legend with better positioning
        legend_elements = [
            Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Positive (↑ risk)', linewidth=0.8),
            Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Negative (↓ risk)', linewidth=0.8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, 
                  edgecolor='#BDC3C7', fancybox=True)
        
        # Add subtle grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
        ax.set_axisbelow(True)
        
        # Add y-axis padding
        ax.set_ylim(-0.5, n_features - 0.5)
        
        plt.tight_layout()
        coef_plot_file = os.path.join(save_dir, "coefficients_nonzero.png")
        plt.savefig(coef_plot_file, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.3)
        plt.close()
        logger.info(f"  Saved: {coef_plot_file}")
    else:
        logger.warning("  No non-zero coefficients to plot!")
    
    # =========================================================================
    # PLOT 2: All coefficients with selection frequency - SORTED BY SIGNED VALUE
    # =========================================================================
    # Sort by signed mean coefficient
    coef_df_mean_sorted = coef_df.sort_values('Mean_Coefficient', ascending=True)
    n_all_features = len(coef_df_mean_sorted)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, n_all_features * 0.38)), 
                              gridspec_kw={'width_ratios': [3.5, 1], 'wspace': 0.08})
    
    # Left panel: Mean coefficients with error bars
    ax1 = axes[0]
    colors = ['#E74C3C' if c < 0 else '#27AE60' if c > 0 else '#95A5A6' 
              for c in coef_df_mean_sorted['Mean_Coefficient']]
    
    y_positions = range(n_all_features)
    ax1.barh(
        y=y_positions,
        width=coef_df_mean_sorted['Mean_Coefficient'],
        xerr=coef_df_mean_sorted['Std_Coefficient'],
        color=colors,
        edgecolor='#2C3E50',
        linewidth=0.6,
        alpha=0.85,
        capsize=3,
        height=0.7,
        error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#7F8C8D'}
    )
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(coef_df_mean_sorted['Feature'], fontsize=9)
    ax1.axvline(x=0, color='#2C3E50', linewidth=1.5)
    ax1.set_xlabel('Mean Coefficient ± SD (across LOO folds)', fontweight='bold', fontsize=11)
    ax1.set_title('Coefficient Stability', fontweight='bold', fontsize=13)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
    ax1.set_axisbelow(True)
    ax1.set_ylim(-0.5, n_all_features - 0.5)
    
    # Extend x-axis for error bars
    mean_min = (coef_df_mean_sorted['Mean_Coefficient'] - coef_df_mean_sorted['Std_Coefficient']).min()
    mean_max = (coef_df_mean_sorted['Mean_Coefficient'] + coef_df_mean_sorted['Std_Coefficient']).max()
    mean_range = mean_max - mean_min
    ax1.set_xlim(mean_min - 0.15 * abs(mean_range), mean_max + 0.15 * abs(mean_range))
    
    # Right panel: Selection frequency with colormap
    ax2 = axes[1]
    freq_values = coef_df_mean_sorted['Selection_Frequency'].values
    
    # Use a diverging colormap based on frequency
    cmap = plt.cm.RdYlGn
    freq_colors = [cmap(f) for f in freq_values]
    
    bars2 = ax2.barh(
        y=y_positions,
        width=freq_values,
        color=freq_colors,
        edgecolor='#2C3E50',
        linewidth=0.4,
        alpha=0.9,
        height=0.7
    )
    
    # Add frequency labels inside or outside bars
    for i, freq in enumerate(freq_values):
        if freq > 0.15:
            ax2.text(freq - 0.02, i, f'{freq*100:.0f}%', va='center', ha='right', 
                    fontsize=7, fontweight='bold', color='white')
        else:
            ax2.text(freq + 0.02, i, f'{freq*100:.0f}%', va='center', ha='left', 
                    fontsize=7, fontweight='bold', color='#2C3E50')
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Selection\nFrequency', fontweight='bold', fontsize=10)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(x=0.5, color='#E74C3C', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
    ax2.set_axisbelow(True)
    ax2.set_ylim(-0.5, n_all_features - 0.5)
    ax2.set_title('Selection\nFrequency', fontweight='bold', fontsize=11)
    
    plt.suptitle('Feature Coefficients and Selection Stability (LOO Cross-Validation)', 
                 fontweight='bold', fontsize=15, y=1.01)
    
    plt.tight_layout()
    stability_plot_file = os.path.join(save_dir, "coefficients_stability.png")
    plt.savefig(stability_plot_file, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.3)
    plt.close()
    logger.info(f"  Saved: {stability_plot_file}")
    
    # =========================================================================
    # PLOT 3: Top features by importance - SORTED BY SIGNED VALUE
    # =========================================================================
    top_n = min(20, len(feature_names))  # Show top 20 or all if fewer
    top_features = coef_df.nlargest(top_n, 'Abs_Coefficient').copy()
    # Sort by signed coefficient value
    top_features = top_features.sort_values('Coefficient', ascending=True)
    
    n_top = len(top_features)
    fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.5)))
    
    colors = ['#E74C3C' if c < 0 else '#27AE60' for c in top_features['Coefficient']]
    
    y_positions = range(n_top)
    bars = ax.barh(
        y=y_positions,
        width=top_features['Coefficient'],
        color=colors,
        edgecolor='#2C3E50',
        linewidth=0.8,
        alpha=0.9,
        height=0.7
    )
    
    # Calculate x-axis limits
    top_min = top_features['Coefficient'].min()
    top_max = top_features['Coefficient'].max()
    top_range = top_max - top_min
    
    x_min = top_min - 0.35 * abs(top_min) if top_min < 0 else -0.1 * top_range
    x_max = top_max + 0.35 * abs(top_max) if top_max > 0 else 0.1 * top_range
    ax.set_xlim(x_min, x_max)
    
    # Add coefficient labels and selection frequency
    for i, (idx, row) in enumerate(top_features.iterrows()):
        coef = row['Coefficient']
        freq = row['Selection_Frequency']
        
        # Coefficient value label
        if coef >= 0:
            label_x = coef + 0.03 * top_range
            ha = 'left'
        else:
            label_x = coef - 0.03 * top_range
            ha = 'right'
        
        ax.text(label_x, i, f'{coef:.3f}', va='center', ha=ha, 
                fontsize=9, fontweight='bold', color='#2C3E50')
        
        # Selection frequency as small annotation on the right margin
        ax.annotate(f'({freq*100:.0f}%)', xy=(x_max, i), 
                    fontsize=8, color='#7F8C8D', style='italic',
                    va='center', ha='left')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features['Feature'], fontsize=10)
    ax.axvline(x=0, color='#2C3E50', linewidth=1.5)
    ax.set_xlabel('Coefficient Value', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {n_top} Features by Coefficient Magnitude\nSorted by Coefficient Value (selection freq. in parentheses)', 
                 fontweight='bold', fontsize=14, pad=15)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Positive Effect (↑ risk)', linewidth=0.8),
        Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Negative Effect (↓ risk)', linewidth=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, 
              edgecolor='#BDC3C7', fancybox=True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, n_top - 0.5)
    
    plt.tight_layout()
    top_coef_file = os.path.join(save_dir, "coefficients_top_features.png")
    plt.savefig(top_coef_file, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.3)
    plt.close()
    logger.info(f"  Saved: {top_coef_file}")
    
    # =========================================================================
    # PLOT 4: Waterfall/Tornado chart - clear visualization of all coefficients
    # =========================================================================
    # Sort all features by signed coefficient
    all_sorted = coef_df.sort_values('Coefficient', ascending=True)
    n_all = len(all_sorted)
    
    fig, ax = plt.subplots(figsize=(14, max(10, n_all * 0.4)))
    
    colors = ['#E74C3C' if c < 0 else '#27AE60' if c > 0 else '#BDC3C7' 
              for c in all_sorted['Coefficient']]
    
    y_positions = range(n_all)
    bars = ax.barh(
        y=y_positions,
        width=all_sorted['Coefficient'],
        color=colors,
        edgecolor='#2C3E50',
        linewidth=0.6,
        alpha=0.9,
        height=0.75
    )
    
    # Calculate x-axis limits
    all_min = all_sorted['Coefficient'].min()
    all_max = all_sorted['Coefficient'].max()
    all_range = max(abs(all_min), abs(all_max))
    
    # Symmetric x-axis with padding
    x_limit = all_range * 1.4
    ax.set_xlim(-x_limit, x_limit)
    
    # Add coefficient labels on the outside
    for i, (idx, row) in enumerate(all_sorted.iterrows()):
        coef = row['Coefficient']
        if coef == 0:
            continue
        
        if coef >= 0:
            label_x = coef + 0.03 * all_range
            ha = 'left'
        else:
            label_x = coef - 0.03 * all_range
            ha = 'right'
        
        ax.text(label_x, i, f'{coef:.3f}', va='center', ha=ha, 
                fontsize=8, fontweight='bold', color='#2C3E50')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_sorted['Feature'], fontsize=9)
    ax.axvline(x=0, color='#2C3E50', linewidth=2, linestyle='-')
    ax.set_xlabel('Coefficient Value', fontweight='bold', fontsize=12)
    ax.set_title('All Feature Coefficients (Tornado Plot)\nSorted by Coefficient Value', 
                 fontweight='bold', fontsize=14, pad=15)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Positive (↑ risk)', linewidth=0.8),
        Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Negative (↓ risk)', linewidth=0.8),
        Patch(facecolor='#BDC3C7', edgecolor='#2C3E50', label='Zero (not selected)', linewidth=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, 
              edgecolor='#BDC3C7', fancybox=True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
    ax.set_axisbelow(True)
    ax.set_ylim(-0.5, n_all - 0.5)
    
    plt.tight_layout()
    tornado_file = os.path.join(save_dir, "coefficients_tornado.png")
    plt.savefig(tornado_file, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.3)
    plt.close()
    logger.info(f"  Saved: {tornado_file}")
    
    logger.info(f"Generated 4 coefficient plots in {save_dir}")


def generate_comparison_plots(
    llm_lasso_results: Dict[str, Any],
    standard_lasso_results: Dict[str, Any],
    save_dir: str,
    logger: logging.Logger,
    use_loo: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive comparison plots between LLM-Lasso and Standard Lasso.
    
    Creates publication-quality figures including:
    - Metrics comparison bar chart
    - ROC curves overlay
    - Confusion matrices side-by-side
    - Coefficient comparison heatmap
    
    Args:
        llm_lasso_results: Results dictionary from LLM-Lasso run
        standard_lasso_results: Results dictionary from Standard Lasso run
        save_dir: Directory to save comparison plots
        logger: Logger instance
        use_loo: Whether LOO cross-validation was used
    
    Returns:
        Dictionary with comparison statistics
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATING COMPARISON PLOTS")
    logger.info("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract summaries
    llm_summary = llm_lasso_results.get('summary', {})
    std_summary = standard_lasso_results.get('summary', {})
    
    # Check if either method was skipped
    if llm_summary.get('lasso_skipped') or std_summary.get('lasso_skipped'):
        logger.warning("Cannot generate comparison plots - one or both methods were skipped")
        return {"comparison_skipped": True}
    
    # Define color palette
    colors = {
        'llm_lasso': '#E74C3C',  # Red
        'standard_lasso': '#3498DB',  # Blue
        'neutral': '#2C3E50'
    }
    
    comparison_stats = {}
    
    # ==================== Figure 1: Metrics Comparison Bar Chart ====================
    logger.info("Generating metrics comparison bar chart...")
    
    if use_loo:
        # LOO mode: use metrics from summary
        metrics_config = [
            ('Accuracy', 'accuracy'),
            ('Sensitivity', 'sensitivity'),
            ('Specificity', 'specificity'),
            ('Balanced Acc.', 'balanced_accuracy'),
            ('F1 Score', 'f1_score'),
            ('AUROC', 'auroc'),
            ('MCC', 'mcc'),
            ('Brier Score', 'brier_score'),
        ]
        
        llm_values = []
        std_values = []
        metric_names = []
        
        for name, key in metrics_config:
            llm_val = llm_summary.get(key)
            std_val = std_summary.get(key)
            if llm_val is not None and std_val is not None:
                llm_values.append(llm_val)
                std_values.append(std_val)
                metric_names.append(name)
        
        if metric_names:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, llm_values, width, label='LLM-Lasso', 
                          color=colors['llm_lasso'], edgecolor='white', linewidth=1.5)
            bars2 = ax.bar(x + width/2, std_values, width, label='Standard Lasso', 
                          color=colors['standard_lasso'], edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars1, llm_values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=colors['llm_lasso'])
            
            for bar, val in zip(bars2, std_values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=colors['standard_lasso'])
            
            ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title('LLM-Lasso vs Standard Lasso: Performance Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, 1.15)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            bar_file = os.path.join(save_dir, 'metrics_comparison_bar.png')
            fig.savefig(bar_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"  Saved: {bar_file}")
            
            # Calculate differences
            for name, llm_val, std_val in zip(metric_names, llm_values, std_values):
                diff = float(llm_val - std_val)
                is_llm_better = bool(diff > 0) if name != 'Brier Score' else bool(diff < 0)
                comparison_stats[name.lower().replace(' ', '_').replace('.', '')] = {
                    'llm_lasso': float(llm_val),
                    'standard_lasso': float(std_val),
                    'difference': diff,
                    'llm_better': is_llm_better
                }
                logger.debug(f"  {name}: LLM={llm_val:.4f}, STD={std_val:.4f}, Diff={diff:+.4f}, LLM Better: {is_llm_better}")
    
    else:
        # Non-LOO mode: use test_error and auroc
        metrics_config = [
            ('Accuracy', lambda s: 1 - s.get('test_error', 0) if s.get('test_error') is not None else s.get('accuracy')),
            ('AUROC', lambda s: s.get('auroc')),
        ]
        
        llm_values = []
        std_values = []
        metric_names = []
        
        for name, getter in metrics_config:
            llm_val = getter(llm_summary)
            std_val = getter(std_summary)
            if llm_val is not None and std_val is not None:
                llm_values.append(llm_val)
                std_values.append(std_val)
                metric_names.append(name)
        
        if metric_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(metric_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, llm_values, width, label='LLM-Lasso', 
                          color=colors['llm_lasso'], edgecolor='white', linewidth=1.5)
            bars2 = ax.bar(x + width/2, std_values, width, label='Standard Lasso', 
                          color=colors['standard_lasso'], edgecolor='white', linewidth=1.5)
            
            for bar, val in zip(bars1, llm_values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            for bar, val in zip(bars2, std_values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax.set_title('LLM-Lasso vs Standard Lasso: Performance Comparison', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax.set_ylim(0, 1.15)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            bar_file = os.path.join(save_dir, 'metrics_comparison_bar.png')
            fig.savefig(bar_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"  Saved: {bar_file}")
    
    # ==================== Figure 2: ROC Curves Comparison ====================
    if use_loo:
        logger.info("Generating ROC curves comparison...")
        
        llm_preds = llm_lasso_results.get('predictions')
        std_preds = standard_lasso_results.get('predictions')
        
        if llm_preds is not None and std_preds is not None:
            try:
                llm_y_true = llm_preds['actual_label'].values
                llm_y_prob = llm_preds['predicted_probability'].values
                std_y_true = std_preds['actual_label'].values
                std_y_prob = std_preds['predicted_probability'].values
                
                llm_fpr, llm_tpr, _ = roc_curve(llm_y_true, llm_y_prob)
                llm_auc = auc(llm_fpr, llm_tpr)
                std_fpr, std_tpr, _ = roc_curve(std_y_true, std_y_prob)
                std_auc = auc(std_fpr, std_tpr)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Plot ROC curves
                ax.plot(llm_fpr, llm_tpr, color=colors['llm_lasso'], lw=2.5, 
                       label=f'LLM-Lasso (AUC = {llm_auc:.3f})')
                ax.plot(std_fpr, std_tpr, color=colors['standard_lasso'], lw=2.5, 
                       label=f'Standard Lasso (AUC = {std_auc:.3f})')
                ax.plot([0, 1], [0, 1], color=colors['neutral'], lw=1.5, 
                       linestyle='--', alpha=0.7, label='Random Classifier')
                
                # Fill between to show difference
                ax.fill_between(llm_fpr, llm_tpr, alpha=0.1, color=colors['llm_lasso'])
                ax.fill_between(std_fpr, std_tpr, alpha=0.1, color=colors['standard_lasso'])
                
                ax.set_xlim([-0.02, 1.02])
                ax.set_ylim([-0.02, 1.02])
                ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
                ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
                ax.set_title('ROC Curve Comparison: LLM-Lasso vs Standard Lasso', 
                            fontsize=14, fontweight='bold')
                ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                plt.tight_layout()
                roc_file = os.path.join(save_dir, 'roc_curves_comparison.png')
                fig.savefig(roc_file, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                logger.info(f"  Saved: {roc_file}")
                
                comparison_stats['auc_difference'] = float(llm_auc - std_auc)
                logger.debug(f"  AUC Difference: LLM ({llm_auc:.4f}) - STD ({std_auc:.4f}) = {llm_auc - std_auc:+.4f}")
                
            except Exception as e:
                logger.warning(f"Could not generate ROC comparison: {e}")
    
    # ==================== Figure 3: Confusion Matrices Comparison ====================
    if use_loo:
        logger.info("Generating confusion matrices comparison...")
        
        llm_preds = llm_lasso_results.get('predictions')
        std_preds = standard_lasso_results.get('predictions')
        
        if llm_preds is not None and std_preds is not None:
            try:
                llm_y_true = llm_preds['actual_label'].values
                llm_y_pred = (llm_preds['predicted_probability'].values >= 0.5).astype(int)
                std_y_true = std_preds['actual_label'].values
                std_y_pred = (std_preds['predicted_probability'].values >= 0.5).astype(int)
                
                llm_cm = confusion_matrix(llm_y_true, llm_y_pred)
                std_cm = confusion_matrix(std_y_true, std_y_pred)
                
                # Normalize confusion matrices
                llm_cm_norm = llm_cm.astype('float') / llm_cm.sum(axis=1)[:, np.newaxis]
                std_cm_norm = std_cm.astype('float') / std_cm.sum(axis=1)[:, np.newaxis]
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # LLM-Lasso confusion matrix
                sns.heatmap(llm_cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                           xticklabels=['Negative (0)', 'Positive (1)'],
                           yticklabels=['Negative (0)', 'Positive (1)'],
                           annot_kws={'size': 16, 'weight': 'bold'},
                           cbar_kws={'label': 'Count'})
                axes[0].set_xlabel('Predicted Label', fontsize=11)
                axes[0].set_ylabel('True Label', fontsize=11)
                axes[0].set_title(f'LLM-Lasso\n(Acc: {llm_summary.get("accuracy", 0):.3f})', 
                                 fontsize=12, fontweight='bold')
                
                # Standard Lasso confusion matrix
                sns.heatmap(std_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                           xticklabels=['Negative (0)', 'Positive (1)'],
                           yticklabels=['Negative (0)', 'Positive (1)'],
                           annot_kws={'size': 16, 'weight': 'bold'},
                           cbar_kws={'label': 'Count'})
                axes[1].set_xlabel('Predicted Label', fontsize=11)
                axes[1].set_ylabel('True Label', fontsize=11)
                axes[1].set_title(f'Standard Lasso\n(Acc: {std_summary.get("accuracy", 0):.3f})', 
                                 fontsize=12, fontweight='bold')
                
                fig.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                cm_file = os.path.join(save_dir, 'confusion_matrices_comparison.png')
                fig.savefig(cm_file, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                logger.info(f"  Saved: {cm_file}")
                
            except Exception as e:
                logger.warning(f"Could not generate confusion matrix comparison: {e}")
    
    # ==================== Figure 4: Feature Selection Summary ====================
    logger.info("Generating feature selection comparison...")
    
    llm_n_features = llm_summary.get('n_features', llm_summary.get('n_nonzero', 0))
    std_n_features = std_summary.get('n_features', std_summary.get('n_nonzero', 0))
    total_features = llm_summary.get('n_features_total', llm_summary.get('total_features', 
                                     std_summary.get('n_features_total', std_summary.get('total_features', 1))))
    
    # Try to get from coefficients if available
    try:
        # Read coefficients JSON files
        llm_coef_file = os.path.join(os.path.dirname(save_dir), 'model_coefficients.json')
        std_coef_file = os.path.join(os.path.dirname(save_dir), 'standard_lasso', 'model_coefficients.json')
        
        if os.path.exists(llm_coef_file):
            with open(llm_coef_file, 'r') as f:
                llm_coef_data = json.load(f)
                llm_n_features = llm_coef_data.get('final_model', {}).get('n_nonzero', llm_n_features)
                total_features = len(llm_coef_data.get('feature_names', [total_features]))
        
        if os.path.exists(std_coef_file):
            with open(std_coef_file, 'r') as f:
                std_coef_data = json.load(f)
                std_n_features = std_coef_data.get('final_model', {}).get('n_nonzero', std_n_features)
    except Exception as e:
        logger.debug(f"Could not read coefficient files: {e}")
    
    if llm_n_features and std_n_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = ['LLM-Lasso', 'Standard Lasso']
        selected = [llm_n_features, std_n_features]
        not_selected = [total_features - llm_n_features, total_features - std_n_features]
        
        x = np.arange(len(methods))
        width = 0.5
        
        bars1 = ax.bar(x, selected, width, label='Selected Features', 
                      color=[colors['llm_lasso'], colors['standard_lasso']], 
                      edgecolor='white', linewidth=2)
        bars2 = ax.bar(x, not_selected, width, bottom=selected, label='Not Selected', 
                      color=['#FADBD8', '#D4E6F1'], edgecolor='white', linewidth=2)
        
        # Add labels
        for i, (s, ns) in enumerate(zip(selected, not_selected)):
            ax.annotate(f'{s}', xy=(x[i], s/2), ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='white')
            ax.annotate(f'{ns}', xy=(x[i], s + ns/2), ha='center', va='center', 
                       fontsize=12, color='gray')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Selection Comparison\n(Total: {total_features} features)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        ax.set_ylim(0, total_features * 1.1)
        
        plt.tight_layout()
        feat_file = os.path.join(save_dir, 'feature_selection_comparison.png')
        fig.savefig(feat_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  Saved: {feat_file}")
        
        comparison_stats['feature_selection'] = {
            'llm_lasso': int(llm_n_features),
            'standard_lasso': int(std_n_features),
            'total_features': int(total_features)
        }
        logger.debug(f"  Feature Selection: LLM={llm_n_features}, STD={std_n_features}, Total={total_features}")
    
    # ==================== Figure 5: Summary Dashboard ====================
    logger.info("Generating comparison dashboard...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Key metrics comparison (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    if use_loo:
        key_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUROC']
        key_keys = ['accuracy', 'sensitivity', 'specificity', 'auroc']
        llm_vals = [llm_summary.get(k, 0) for k in key_keys]
        std_vals = [std_summary.get(k, 0) for k in key_keys]
    else:
        key_metrics = ['Accuracy', 'AUROC']
        llm_vals = [1 - llm_summary.get('test_error', 0), llm_summary.get('auroc', 0)]
        std_vals = [1 - std_summary.get('test_error', 0), std_summary.get('auroc', 0)]
    
    x = np.arange(len(key_metrics))
    width = 0.35
    
    ax1.bar(x - width/2, llm_vals, width, label='LLM-Lasso', color=colors['llm_lasso'])
    ax1.bar(x + width/2, std_vals, width, label='Standard Lasso', color=colors['standard_lasso'])
    ax1.set_ylabel('Value')
    ax1.set_title('Key Performance Metrics', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(key_metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Summary text (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Determine winner for each metric
    summary_text = "COMPARISON SUMMARY\n" + "=" * 25 + "\n\n"
    
    llm_wins = 0
    std_wins = 0
    
    for i, metric in enumerate(key_metrics):
        llm_v = llm_vals[i] if llm_vals[i] is not None else 0
        std_v = std_vals[i] if std_vals[i] is not None else 0
        diff = llm_v - std_v
        
        if abs(diff) < 0.001:
            winner = "TIE"
        elif diff > 0:
            winner = "LLM-Lasso"
            llm_wins += 1
        else:
            winner = "Standard"
            std_wins += 1
        
        summary_text += f"{metric}:\n"
        summary_text += f"  LLM: {llm_v:.4f}\n"
        summary_text += f"  Std: {std_v:.4f}\n"
        summary_text += f"  Winner: {winner}\n\n"
    
    summary_text += "=" * 25 + "\n"
    summary_text += f"LLM-Lasso wins: {llm_wins}\n"
    summary_text += f"Standard wins: {std_wins}\n"
    
    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    comparison_stats['overall'] = {
        'llm_lasso_wins': int(llm_wins),
        'standard_lasso_wins': int(std_wins)
    }
    logger.debug(f"  Overall: LLM wins={llm_wins}, STD wins={std_wins}")
    
    # Feature count comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    if llm_n_features and std_n_features:
        methods = ['LLM-Lasso', 'Standard']
        counts = [llm_n_features, std_n_features]
        bars = ax3.bar(methods, counts, color=[colors['llm_lasso'], colors['standard_lasso']])
        ax3.set_ylabel('Features Selected')
        ax3.set_title('Sparsity Comparison', fontweight='bold')
        for bar, count in zip(bars, counts):
            ax3.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Feature counts\nnot available', ha='center', va='center')
        ax3.set_title('Sparsity Comparison', fontweight='bold')
    
    # Performance improvement (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if use_loo:
        improvements = [(llm_vals[i] - std_vals[i]) * 100 for i in range(len(key_metrics))]
        bar_colors = [colors['llm_lasso'] if imp > 0 else colors['standard_lasso'] for imp in improvements]
        bars = ax4.bar(key_metrics, improvements, color=bar_colors)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('LLM-Lasso Improvement over Standard', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Detailed metrics\nnot available', ha='center', va='center')
        ax4.set_title('Improvement Analysis', fontweight='bold')
    
    # Method info (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    info_text = "ANALYSIS DETAILS\n" + "=" * 25 + "\n\n"
    info_text += f"CV Method: {'LOO' if use_loo else 'Train/Test Split'}\n"
    info_text += f"Samples: {llm_summary.get('n_samples', 'N/A')}\n"
    info_text += f"Features: {total_features}\n"
    info_text += f"Inner CV Folds: {llm_summary.get('inner_cv_folds', 'N/A')}\n"
    info_text += f"SMOTE: {'Yes' if llm_summary.get('smote_enabled') else 'No'}\n"
    
    ax5.text(0.1, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle('LLM-Lasso vs Standard Lasso: Comprehensive Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    dashboard_file = os.path.join(save_dir, 'comparison_dashboard.png')
    fig.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {dashboard_file}")
    
    logger.info(f"Generated comparison plots in {save_dir}")
    
    return comparison_stats


def save_gridsearch_plots(
    cv_result,
    save_dir: str,
    logger: logging.Logger,
    fold_label: str = "final",
    optimize_metric: str = "accuracy"
) -> None:
    """
    Save gridsearch parameter search plots showing lambda vs metrics.
    
    Args:
        cv_result: CVGrpnetResult from cv_grpnet
        save_dir: Directory to save plots (gridsearch subfolder will be created)
        logger: Logger instance
        fold_label: Label for this gridsearch (e.g., "final", "fold_1")
        optimize_metric: Metric used for hyperparameter optimization (for highlighting)
    """
    # Create gridsearch subfolder
    gridsearch_dir = os.path.join(save_dir, "gridsearch")
    os.makedirs(gridsearch_dir, exist_ok=True)
    
    lmdas = cv_result.lmdas
    n_lambdas = len(lmdas)
    
    # Compute accuracy from test_error (test_error is Hamming loss = 1 - accuracy)
    accuracy = 1.0 - cv_result.test_error
    
    # Get AUC-ROC if available
    roc_auc = cv_result.roc_auc if cv_result.roc_auc is not None else None
    
    # Get extended metrics if available
    sensitivity = cv_result.sensitivity if hasattr(cv_result, 'sensitivity') and cv_result.sensitivity is not None else None
    specificity = cv_result.specificity if hasattr(cv_result, 'specificity') and cv_result.specificity is not None else None
    balanced_accuracy = cv_result.balanced_accuracy if hasattr(cv_result, 'balanced_accuracy') and cv_result.balanced_accuracy is not None else None
    f1 = cv_result.f1 if hasattr(cv_result, 'f1') and cv_result.f1 is not None else None
    
    # Get best lambda index (this should already use the optimize_metric)
    best_idx = cv_result.best_idx
    best_lambda = lmdas[best_idx]
    
    # Use -log(lambda) for better visualization (common practice)
    neg_log_lambda = -np.log10(lmdas)
    
    logger.info(f"Saving gridsearch plots ({n_lambdas} lambda values) to {gridsearch_dir}")
    logger.info(f"  Optimization metric: {optimize_metric}")
    
    # Plot 1: Lambda vs Accuracy
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(neg_log_lambda, accuracy, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5, 
                label=f'Best λ = {best_lambda:.2e}')
    ax1.scatter([-np.log10(best_lambda)], [accuracy[best_idx]], color='r', s=100, zorder=5,
                label=f'Best Accuracy = {accuracy[best_idx]:.4f}')
    ax1.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Grid Search: Lambda vs Accuracy ({fold_label})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    plt.tight_layout()
    fig1.savefig(os.path.join(gridsearch_dir, f'lambda_vs_accuracy_{fold_label}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Lambda vs AUC-ROC (if available)
    if roc_auc is not None and not np.all(roc_auc == 0):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(neg_log_lambda, roc_auc, 'g-', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax2.axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5,
                    label=f'Best λ = {best_lambda:.2e}')
        ax2.scatter([-np.log10(best_lambda)], [roc_auc[best_idx]], color='r', s=100, zorder=5,
                    label=f'AUC at Best λ = {roc_auc[best_idx]:.4f}')
        ax2.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
        ax2.set_ylabel('AUC-ROC', fontsize=12)
        ax2.set_title(f'Grid Search: Lambda vs AUC-ROC ({fold_label})', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        plt.tight_layout()
        fig2.savefig(os.path.join(gridsearch_dir, f'lambda_vs_auc_{fold_label}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    # Plot 3: Lambda vs CV Loss
    avg_losses = cv_result.avg_losses
    std_losses = np.std(cv_result.losses, axis=0, ddof=0) if cv_result.losses is not None else None
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if std_losses is not None:
        ax3.errorbar(neg_log_lambda, avg_losses, yerr=std_losses, fmt='m-', linewidth=1.5,
                     marker='d', markersize=3, alpha=0.7, capsize=2, elinewidth=0.5, ecolor='gray')
    else:
        ax3.plot(neg_log_lambda, avg_losses, 'm-', linewidth=2, marker='d', markersize=3, alpha=0.7)
    ax3.axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5,
                label=f'Best λ = {best_lambda:.2e}')
    ax3.scatter([-np.log10(best_lambda)], [avg_losses[best_idx]], color='r', s=100, zorder=5,
                label=f'Min Loss = {avg_losses[best_idx]:.4f}')
    ax3.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
    ax3.set_ylabel('CV Loss (with std)', fontsize=12)
    ax3.set_title(f'Grid Search: Lambda vs CV Loss ({fold_label})', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(gridsearch_dir, f'lambda_vs_loss_{fold_label}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Combined metrics plot
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(neg_log_lambda, accuracy, 'b-', linewidth=2, marker='o', markersize=2, alpha=0.7)
    axes[0].axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].scatter([-np.log10(best_lambda)], [accuracy[best_idx]], color='r', s=80, zorder=5)
    axes[0].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # AUC-ROC
    if roc_auc is not None and not np.all(roc_auc == 0):
        axes[1].plot(neg_log_lambda, roc_auc, 'g-', linewidth=2, marker='s', markersize=2, alpha=0.7)
        axes[1].axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1].scatter([-np.log10(best_lambda)], [roc_auc[best_idx]], color='r', s=80, zorder=5)
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_ylim([0, 1.05])
    else:
        axes[1].text(0.5, 0.5, 'AUC-ROC\nNot Available', ha='center', va='center', fontsize=12)
    axes[1].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[1].set_title('AUC-ROC')
    axes[1].grid(True, alpha=0.3)
    
    # CV Loss
    axes[2].plot(neg_log_lambda, avg_losses, 'm-', linewidth=2, marker='d', markersize=2, alpha=0.7)
    axes[2].axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].scatter([-np.log10(best_lambda)], [avg_losses[best_idx]], color='r', s=80, zorder=5)
    axes[2].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[2].set_ylabel('CV Loss')
    axes[2].set_title('CV Loss')
    axes[2].grid(True, alpha=0.3)
    
    fig4.suptitle(f'Grid Search Summary: {n_lambdas} Lambda Values ({fold_label})', 
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4.savefig(os.path.join(gridsearch_dir, f'gridsearch_summary_{fold_label}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # Plot 5: Extended metrics (sensitivity, specificity, balanced accuracy, F1) if available
    if sensitivity is not None and specificity is not None:
        fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_data = [
            (sensitivity, 'Sensitivity', 'darkorange'),
            (specificity, 'Specificity', 'purple'),
            (balanced_accuracy, 'Balanced Accuracy', 'teal'),
            (f1, 'F1 Score', 'brown')
        ]
        
        for ax, (metric, name, color) in zip(axes5.flat, metrics_data):
            if metric is not None:
                ax.plot(neg_log_lambda, metric, linestyle='-', linewidth=2, marker='o', markersize=2, alpha=0.7, color=color)
                ax.axvline(x=-np.log10(best_lambda), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.scatter([-np.log10(best_lambda)], [metric[best_idx]], color='r', s=80, zorder=5)
                ax.set_xlabel(r'$-\log_{10}(\lambda)$')
                ax.set_ylabel(name)
                ax.set_title(f'{name} (at best λ: {metric[best_idx]:.4f})')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                
                # Highlight if this is the optimization metric
                if name.lower().replace(' ', '_') == optimize_metric:
                    ax.set_title(f'{name} (at best λ: {metric[best_idx]:.4f}) ★ OPTIMIZED', color='red')
            else:
                ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center', fontsize=12)
                ax.set_xlabel(r'$-\log_{10}(\lambda)$')
        
        fig5.suptitle(f'Extended Metrics: {n_lambdas} Lambda Values ({fold_label})\nOptimization Metric: {optimize_metric}', 
                      fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig5.savefig(os.path.join(gridsearch_dir, f'extended_metrics_{fold_label}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig5)
    
    # Save gridsearch data to CSV
    gridsearch_data = pd.DataFrame({
        'lambda': lmdas,
        'neg_log_lambda': neg_log_lambda,
        'accuracy': accuracy,
        'cv_loss': avg_losses
    })
    if roc_auc is not None:
        gridsearch_data['auc_roc'] = roc_auc
    if sensitivity is not None:
        gridsearch_data['sensitivity'] = sensitivity
    if specificity is not None:
        gridsearch_data['specificity'] = specificity
    if balanced_accuracy is not None:
        gridsearch_data['balanced_accuracy'] = balanced_accuracy
    if f1 is not None:
        gridsearch_data['f1'] = f1
    gridsearch_data['is_best'] = np.arange(len(lmdas)) == best_idx
    gridsearch_data['optimize_metric'] = optimize_metric
    gridsearch_data.to_csv(os.path.join(gridsearch_dir, f'gridsearch_results_{fold_label}.csv'), index=False)
    
    logger.info(f"  Saved gridsearch plots and data for {fold_label} ({n_lambdas} lambda values)")


def save_loo_gridsearch_plots(
    loo_gridsearch_results: List[Dict],
    save_dir: str,
    logger: logging.Logger,
    max_individual_plots: Optional[int] = None,
    optimize_metric: str = "accuracy"
) -> None:
    """
    Save gridsearch plots for LOO folds: individual fold plots and aggregate summary.
    
    Args:
        loo_gridsearch_results: List of dicts with gridsearch results from each LOO fold
        save_dir: Directory to save plots
        logger: Logger instance
        max_individual_plots: Maximum number of individual fold plots to save. 
                              If None, saves ALL folds. Set a limit to avoid too many files.
        optimize_metric: Metric used for hyperparameter optimization (for highlighting)
    """
    if not loo_gridsearch_results:
        logger.warning("No LOO gridsearch results to save")
        return
    
    # Create gridsearch subfolder with loo_folds subfolder
    gridsearch_dir = os.path.join(save_dir, "gridsearch")
    loo_folds_dir = os.path.join(gridsearch_dir, "loo_folds")
    os.makedirs(loo_folds_dir, exist_ok=True)
    
    n_folds = len(loo_gridsearch_results)
    logger.info(f"  Saving gridsearch results for {n_folds} LOO folds...")
    
    # Get common lambda values (assuming all folds use the same lambda path)
    lmdas = loo_gridsearch_results[0]['lmdas']
    n_lambdas = len(lmdas)
    neg_log_lambda = -np.log10(lmdas)
    
    # Collect metrics across all folds
    all_accuracy = []
    all_auc = []
    all_loss = []
    all_best_lambdas = []
    all_sensitivity = []
    all_specificity = []
    all_balanced_accuracy = []
    all_f1 = []
    
    for result in loo_gridsearch_results:
        accuracy = 1.0 - result['test_error']  # Convert Hamming loss to accuracy
        all_accuracy.append(accuracy)
        all_loss.append(result['avg_losses'])
        all_best_lambdas.append(result['best_lambda'])
        
        if result['roc_auc'] is not None:
            all_auc.append(result['roc_auc'])
        
        # Collect extended metrics if available
        if result.get('sensitivity') is not None:
            all_sensitivity.append(result['sensitivity'])
        if result.get('specificity') is not None:
            all_specificity.append(result['specificity'])
        if result.get('balanced_accuracy') is not None:
            all_balanced_accuracy.append(result['balanced_accuracy'])
        if result.get('f1') is not None:
            all_f1.append(result['f1'])
    
    all_accuracy = np.array(all_accuracy)  # Shape: (n_folds, n_lambdas)
    all_loss = np.array(all_loss)
    all_auc = np.array(all_auc) if all_auc else None
    
    # Compute mean and std across folds
    mean_accuracy = np.mean(all_accuracy, axis=0)
    std_accuracy = np.std(all_accuracy, axis=0)
    mean_loss = np.mean(all_loss, axis=0)
    std_loss = np.std(all_loss, axis=0)
    
    if all_auc is not None and len(all_auc) > 0:
        mean_auc = np.mean(all_auc, axis=0)
        std_auc = np.std(all_auc, axis=0)
    else:
        mean_auc = None
        std_auc = None
    
    # Aggregate extended metrics
    if len(all_sensitivity) > 0:
        mean_sensitivity = np.mean(np.array(all_sensitivity), axis=0)
        std_sensitivity = np.std(np.array(all_sensitivity), axis=0)
    else:
        mean_sensitivity = None
        std_sensitivity = None
        
    if len(all_specificity) > 0:
        mean_specificity = np.mean(np.array(all_specificity), axis=0)
        std_specificity = np.std(np.array(all_specificity), axis=0)
    else:
        mean_specificity = None
        std_specificity = None
        
    if len(all_balanced_accuracy) > 0:
        mean_balanced_accuracy = np.mean(np.array(all_balanced_accuracy), axis=0)
        std_balanced_accuracy = np.std(np.array(all_balanced_accuracy), axis=0)
    else:
        mean_balanced_accuracy = None
        std_balanced_accuracy = None
        
    if len(all_f1) > 0:
        mean_f1 = np.mean(np.array(all_f1), axis=0)
        std_f1 = np.std(np.array(all_f1), axis=0)
    else:
        mean_f1 = None
        std_f1 = None
    
    # Find the optimal lambda based on mean accuracy
    best_idx_aggregate = np.argmax(mean_accuracy)
    best_lambda_aggregate = lmdas[best_idx_aggregate]
    
    # ==================== AGGREGATE PLOTS ====================
    
    # Plot 1: Aggregate Accuracy with confidence band
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot individual fold traces (light gray)
    for i, acc in enumerate(all_accuracy):
        ax1.plot(neg_log_lambda, acc, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # Plot mean with confidence band
    ax1.fill_between(neg_log_lambda, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
                     alpha=0.3, color='blue', label='±1 std')
    ax1.plot(neg_log_lambda, mean_accuracy, 'b-', linewidth=2.5, label=f'Mean (n={n_folds} folds)')
    
    # Mark best lambda
    ax1.axvline(x=-np.log10(best_lambda_aggregate), color='r', linestyle='--', linewidth=1.5,
                label=f'Best λ = {best_lambda_aggregate:.2e}')
    ax1.scatter([-np.log10(best_lambda_aggregate)], [mean_accuracy[best_idx_aggregate]], 
                color='r', s=100, zorder=5, label=f'Best Accuracy = {mean_accuracy[best_idx_aggregate]:.4f}')
    
    ax1.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'LOO Cross-Validation: Lambda vs Accuracy\n(Aggregated across {n_folds} folds, {n_lambdas} λ values)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    plt.tight_layout()
    fig1.savefig(os.path.join(gridsearch_dir, 'lambda_vs_accuracy_loo_aggregate.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Aggregate AUC-ROC (if available)
    if mean_auc is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        
        for i, auc in enumerate(all_auc):
            ax2.plot(neg_log_lambda, auc, color='lightgray', linewidth=0.5, alpha=0.5)
        
        ax2.fill_between(neg_log_lambda, mean_auc - std_auc, mean_auc + std_auc,
                         alpha=0.3, color='green', label='±1 std')
        ax2.plot(neg_log_lambda, mean_auc, 'g-', linewidth=2.5, label=f'Mean (n={n_folds} folds)')
        
        ax2.axvline(x=-np.log10(best_lambda_aggregate), color='r', linestyle='--', linewidth=1.5)
        ax2.scatter([-np.log10(best_lambda_aggregate)], [mean_auc[best_idx_aggregate]], 
                    color='r', s=100, zorder=5)
        
        ax2.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
        ax2.set_ylabel('AUC-ROC', fontsize=12)
        ax2.set_title(f'LOO Cross-Validation: Lambda vs AUC-ROC\n(Aggregated across {n_folds} folds)', 
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        plt.tight_layout()
        fig2.savefig(os.path.join(gridsearch_dir, 'lambda_vs_auc_loo_aggregate.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    # Plot 3: Aggregate CV Loss
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    for i, loss in enumerate(all_loss):
        ax3.plot(neg_log_lambda, loss, color='lightgray', linewidth=0.5, alpha=0.5)
    
    ax3.fill_between(neg_log_lambda, mean_loss - std_loss, mean_loss + std_loss,
                     alpha=0.3, color='purple', label='±1 std')
    ax3.plot(neg_log_lambda, mean_loss, 'm-', linewidth=2.5, label=f'Mean (n={n_folds} folds)')
    
    best_loss_idx = np.argmin(mean_loss)
    ax3.axvline(x=-np.log10(lmdas[best_loss_idx]), color='r', linestyle='--', linewidth=1.5,
                label=f'Best λ = {lmdas[best_loss_idx]:.2e}')
    ax3.scatter([-np.log10(lmdas[best_loss_idx])], [mean_loss[best_loss_idx]], 
                color='r', s=100, zorder=5, label=f'Min Loss = {mean_loss[best_loss_idx]:.4f}')
    
    ax3.set_xlabel(r'$-\log_{10}(\lambda)$', fontsize=12)
    ax3.set_ylabel('CV Loss', fontsize=12)
    ax3.set_title(f'LOO Cross-Validation: Lambda vs CV Loss\n(Aggregated across {n_folds} folds)', 
                  fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(gridsearch_dir, 'lambda_vs_loss_loo_aggregate.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Combined summary for LOO aggregate
    fig4, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy
    axes[0].fill_between(neg_log_lambda, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
                         alpha=0.3, color='blue')
    axes[0].plot(neg_log_lambda, mean_accuracy, 'b-', linewidth=2)
    axes[0].axvline(x=-np.log10(best_lambda_aggregate), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].scatter([-np.log10(best_lambda_aggregate)], [mean_accuracy[best_idx_aggregate]], color='r', s=80, zorder=5)
    axes[0].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[0].set_ylabel('Accuracy (mean ± std)')
    axes[0].set_title(f'Accuracy\nBest: {mean_accuracy[best_idx_aggregate]:.3f}±{std_accuracy[best_idx_aggregate]:.3f}')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # AUC-ROC
    if mean_auc is not None:
        axes[1].fill_between(neg_log_lambda, mean_auc - std_auc, mean_auc + std_auc,
                             alpha=0.3, color='green')
        axes[1].plot(neg_log_lambda, mean_auc, 'g-', linewidth=2)
        axes[1].axvline(x=-np.log10(best_lambda_aggregate), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1].scatter([-np.log10(best_lambda_aggregate)], [mean_auc[best_idx_aggregate]], color='r', s=80, zorder=5)
        axes[1].set_ylabel('AUC-ROC (mean ± std)')
        axes[1].set_title(f'AUC-ROC\nAt best λ: {mean_auc[best_idx_aggregate]:.3f}±{std_auc[best_idx_aggregate]:.3f}')
        axes[1].set_ylim([0, 1.05])
    else:
        axes[1].text(0.5, 0.5, 'AUC-ROC\nNot Available', ha='center', va='center', fontsize=12)
    axes[1].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[1].grid(True, alpha=0.3)
    
    # CV Loss
    axes[2].fill_between(neg_log_lambda, mean_loss - std_loss, mean_loss + std_loss,
                         alpha=0.3, color='purple')
    axes[2].plot(neg_log_lambda, mean_loss, 'm-', linewidth=2)
    axes[2].axvline(x=-np.log10(lmdas[best_loss_idx]), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[2].scatter([-np.log10(lmdas[best_loss_idx])], [mean_loss[best_loss_idx]], color='r', s=80, zorder=5)
    axes[2].set_xlabel(r'$-\log_{10}(\lambda)$')
    axes[2].set_ylabel('CV Loss (mean ± std)')
    axes[2].set_title(f'CV Loss\nMin: {mean_loss[best_loss_idx]:.4f}±{std_loss[best_loss_idx]:.4f}')
    axes[2].grid(True, alpha=0.3)
    
    fig4.suptitle(f'LOO Grid Search Summary: {n_folds} Folds × {n_lambdas} Lambda Values\nOptimization Metric: {optimize_metric}', 
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4.savefig(os.path.join(gridsearch_dir, 'gridsearch_summary_loo_aggregate.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # Plot 4b: Extended metrics (sensitivity, specificity, balanced_accuracy, f1) if available
    if mean_sensitivity is not None and mean_specificity is not None:
        fig4b, axes4b = plt.subplots(2, 2, figsize=(14, 10))
        
        ext_metrics_data = [
            (mean_sensitivity, std_sensitivity, 'Sensitivity', 'darkorange'),
            (mean_specificity, std_specificity, 'Specificity', 'purple'),
            (mean_balanced_accuracy, std_balanced_accuracy, 'Balanced Accuracy', 'teal'),
            (mean_f1, std_f1, 'F1 Score', 'brown')
        ]
        
        for ax, (mean_metric, std_metric, name, color) in zip(axes4b.flat, ext_metrics_data):
            if mean_metric is not None:
                ax.fill_between(neg_log_lambda, mean_metric - std_metric, mean_metric + std_metric,
                               alpha=0.3, color=color)
                ax.plot(neg_log_lambda, mean_metric, '-', linewidth=2, color=color, label=f'Mean (n={n_folds})')
                ax.axvline(x=-np.log10(best_lambda_aggregate), color='r', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.scatter([-np.log10(best_lambda_aggregate)], [mean_metric[best_idx_aggregate]], 
                          color='r', s=80, zorder=5)
                ax.set_xlabel(r'$-\log_{10}(\lambda)$')
                ax.set_ylabel(f'{name} (mean ± std)')
                title = f'{name}\nAt best λ: {mean_metric[best_idx_aggregate]:.3f}±{std_metric[best_idx_aggregate]:.3f}'
                
                # Highlight if this is the optimization metric
                if name.lower().replace(' ', '_') == optimize_metric:
                    title += ' ★ OPTIMIZED'
                    ax.set_title(title, color='red', fontweight='bold')
                else:
                    ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                ax.legend(loc='best')
            else:
                ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center', fontsize=12)
                ax.set_xlabel(r'$-\log_{10}(\lambda)$')
        
        fig4b.suptitle(f'LOO Grid Search Extended Metrics: {n_folds} Folds\nOptimization Metric: {optimize_metric}', 
                      fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig4b.savefig(os.path.join(gridsearch_dir, 'extended_metrics_loo_aggregate.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4b)
    
    # Plot 5: Distribution of best lambdas across folds
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(-np.log10(all_best_lambdas), bins=min(30, n_folds // 2 + 1), edgecolor='white', alpha=0.7, color='steelblue')
    ax5.axvline(x=-np.log10(np.median(all_best_lambdas)), color='r', linestyle='--', linewidth=2,
                label=f'Median λ = {np.median(all_best_lambdas):.2e}')
    ax5.axvline(x=-np.log10(np.mean(all_best_lambdas)), color='orange', linestyle='--', linewidth=2,
                label=f'Mean λ = {np.mean(all_best_lambdas):.2e}')
    ax5.set_xlabel(r'$-\log_{10}(\lambda_{best})$', fontsize=12)
    ax5.set_ylabel('Count (folds)', fontsize=12)
    ax5.set_title(f'Distribution of Selected λ Values Across {n_folds} LOO Folds', fontsize=14, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5.savefig(os.path.join(gridsearch_dir, 'best_lambda_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    # Save aggregate data to CSV
    aggregate_data = pd.DataFrame({
        'lambda': lmdas,
        'neg_log_lambda': neg_log_lambda,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_cv_loss': mean_loss,
        'std_cv_loss': std_loss
    })
    if mean_auc is not None:
        aggregate_data['mean_auc_roc'] = mean_auc
        aggregate_data['std_auc_roc'] = std_auc
    if mean_sensitivity is not None:
        aggregate_data['mean_sensitivity'] = mean_sensitivity
        aggregate_data['std_sensitivity'] = std_sensitivity
    if mean_specificity is not None:
        aggregate_data['mean_specificity'] = mean_specificity
        aggregate_data['std_specificity'] = std_specificity
    if mean_balanced_accuracy is not None:
        aggregate_data['mean_balanced_accuracy'] = mean_balanced_accuracy
        aggregate_data['std_balanced_accuracy'] = std_balanced_accuracy
    if mean_f1 is not None:
        aggregate_data['mean_f1'] = mean_f1
        aggregate_data['std_f1'] = std_f1
    aggregate_data['optimize_metric'] = optimize_metric
    aggregate_data['is_best_accuracy'] = np.arange(n_lambdas) == best_idx_aggregate
    aggregate_data['is_best_loss'] = np.arange(n_lambdas) == best_loss_idx
    aggregate_data.to_csv(os.path.join(gridsearch_dir, 'gridsearch_results_loo_aggregate.csv'), index=False)
    
    # Save best lambda per fold
    best_lambda_data = pd.DataFrame({
        'fold': [r['fold'] for r in loo_gridsearch_results],
        'best_lambda': all_best_lambdas,
        'neg_log_best_lambda': -np.log10(all_best_lambdas),
        'best_accuracy': [1.0 - r['test_error'][r['best_idx']] for r in loo_gridsearch_results]
    })
    best_lambda_data.to_csv(os.path.join(gridsearch_dir, 'best_lambda_per_fold.csv'), index=False)
    
    logger.info(f"  Saved aggregate gridsearch plots and data")
    
    # ==================== INDIVIDUAL FOLD PLOTS ====================
    
    # Determine which folds to save
    if max_individual_plots is None:
        # Save ALL folds
        folds_to_save = range(n_folds)
        logger.info(f"  Saving individual plots for ALL {n_folds} LOO folds")
    elif n_folds <= max_individual_plots:
        folds_to_save = range(n_folds)
        logger.info(f"  Saving individual plots for all {n_folds} folds")
    else:
        # Save evenly spaced folds (limited)
        step = n_folds // max_individual_plots
        folds_to_save = list(range(0, n_folds, step))[:max_individual_plots]
        logger.info(f"  Saving individual plots for {len(folds_to_save)} of {n_folds} folds (every {step}th fold)")
    
    for fold_idx in folds_to_save:
        result = loo_gridsearch_results[fold_idx]
        fold_lmdas = result['lmdas']
        fold_accuracy = 1.0 - result['test_error']
        fold_loss = result['avg_losses']
        fold_best_idx = result['best_idx']
        fold_auc = result['roc_auc']
        
        neg_log_lmda = -np.log10(fold_lmdas)
        
        # Create combined plot for this fold
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(neg_log_lmda, fold_accuracy, 'b-', linewidth=2, marker='o', markersize=2)
        axes[0].axvline(x=neg_log_lmda[fold_best_idx], color='r', linestyle='--', linewidth=1.5)
        axes[0].scatter([neg_log_lmda[fold_best_idx]], [fold_accuracy[fold_best_idx]], color='r', s=80, zorder=5)
        axes[0].set_xlabel(r'$-\log_{10}(\lambda)$')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Accuracy: {fold_accuracy[fold_best_idx]:.4f}')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # AUC-ROC
        if fold_auc is not None and not np.all(fold_auc == 0):
            axes[1].plot(neg_log_lmda, fold_auc, 'g-', linewidth=2, marker='s', markersize=2)
            axes[1].axvline(x=neg_log_lmda[fold_best_idx], color='r', linestyle='--', linewidth=1.5)
            axes[1].scatter([neg_log_lmda[fold_best_idx]], [fold_auc[fold_best_idx]], color='r', s=80, zorder=5)
            axes[1].set_ylabel('AUC-ROC')
            axes[1].set_title(f'AUC: {fold_auc[fold_best_idx]:.4f}')
            axes[1].set_ylim([0, 1.05])
        else:
            axes[1].text(0.5, 0.5, 'AUC-ROC\nNot Available', ha='center', va='center')
        axes[1].set_xlabel(r'$-\log_{10}(\lambda)$')
        axes[1].grid(True, alpha=0.3)
        
        # CV Loss
        axes[2].plot(neg_log_lmda, fold_loss, 'm-', linewidth=2, marker='d', markersize=2)
        axes[2].axvline(x=neg_log_lmda[fold_best_idx], color='r', linestyle='--', linewidth=1.5)
        axes[2].scatter([neg_log_lmda[fold_best_idx]], [fold_loss[fold_best_idx]], color='r', s=80, zorder=5)
        axes[2].set_xlabel(r'$-\log_{10}(\lambda)$')
        axes[2].set_ylabel('CV Loss')
        axes[2].set_title(f'Loss: {fold_loss[fold_best_idx]:.4f}')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(f'LOO Fold {fold_idx + 1}/{n_folds}: Grid Search (Best λ = {fold_lmdas[fold_best_idx]:.2e})', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(loo_folds_dir, f'fold_{fold_idx + 1:03d}_gridsearch.png'), dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    logger.info(f"  Saved {len(folds_to_save)} individual LOO fold gridsearch plots to {loo_folds_dir}")
    
    # Summary statistics
    logger.info(f"  LOO Gridsearch Summary:")
    logger.info(f"    Mean best λ: {np.mean(all_best_lambdas):.4e} (std: {np.std(all_best_lambdas):.4e})")
    logger.info(f"    Median best λ: {np.median(all_best_lambdas):.4e}")
    logger.info(f"    Mean accuracy at best λ: {np.mean([1.0 - r['test_error'][r['best_idx']] for r in loo_gridsearch_results]):.4f}")


def select_best_lambda(fit, optimize_metric: str = "accuracy") -> int:
    """
    Select the best lambda index based on the specified optimization metric.
    
    Args:
        fit: CVGrpnetResult object with metrics arrays
        optimize_metric: Metric to optimize. One of:
            - "accuracy": minimize test_error (Hamming loss) = maximize accuracy
            - "sensitivity": maximize sensitivity (true positive rate)
            - "specificity": maximize specificity (true negative rate)
            - "balanced_accuracy": maximize (sensitivity + specificity) / 2
            - "f1": maximize F1 score
            - "auc_roc": maximize ROC AUC
    
    Returns:
        Index of the best lambda in the path
    """
    if optimize_metric == "accuracy":
        # Minimize Hamming loss = maximize accuracy
        return int(np.argmin(fit.test_error))
    elif optimize_metric == "sensitivity":
        if fit.sensitivity is None:
            raise ValueError("Sensitivity not available (only supported for binary classification)")
        return int(np.argmax(fit.sensitivity))
    elif optimize_metric == "specificity":
        if fit.specificity is None:
            raise ValueError("Specificity not available (only supported for binary classification)")
        return int(np.argmax(fit.specificity))
    elif optimize_metric == "balanced_accuracy":
        if fit.balanced_accuracy is None:
            raise ValueError("Balanced accuracy not available (only supported for binary classification)")
        return int(np.argmax(fit.balanced_accuracy))
    elif optimize_metric == "f1":
        if fit.f1 is None:
            raise ValueError("F1 score not available (only supported for binary classification)")
        return int(np.argmax(fit.f1))
    elif optimize_metric == "auc_roc":
        if fit.roc_auc is None:
            raise ValueError("ROC AUC not available")
        return int(np.argmax(fit.roc_auc))
    else:
        raise ValueError(f"Unknown optimization metric: {optimize_metric}. "
                        f"Must be one of: accuracy, sensitivity, specificity, balanced_accuracy, f1, auc_roc")


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
    smote_random_state: int = 42,
    lmda_path_size: int = 100,
    optimize_metric: str = "accuracy",
    compute_ci: bool = True,
    bootstrap_method: str = "standard",
    bootstrap_n_rounds: int = 1000,
    ci_level: float = 0.95
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
        lmda_path_size: Number of lambda values in the regularization path (at least 50)
        optimize_metric: Metric to maximize during hyperparameter selection.
            Options: accuracy, sensitivity, specificity, balanced_accuracy, f1, auc_roc
        compute_ci: Whether to compute bootstrap confidence intervals
        bootstrap_method: Bootstrap method ('standard' or '632')
        bootstrap_n_rounds: Number of bootstrap iterations
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
    
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
    logger.info(f"Lambda path size: {lmda_path_size} (regularization parameters)")
    logger.info(f"Optimization metric: {optimize_metric}")
    logger.info(f"Threads: {n_threads}")
    logger.info(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    logger.info(f"Bootstrap CI: {'Enabled' if compute_ci else 'Disabled'}")
    if compute_ci:
        logger.info(f"  Method: {bootstrap_method}, Rounds: {bootstrap_n_rounds}, Level: {ci_level*100:.0f}%")
    
    # Use index as participant IDs if not provided
    if participant_ids is None:
        participant_ids = X.index
    
    # Convert penalty factors to importances (inverse relationship)
    penalty_scores_arr = np.array(penalty_scores)
    importances = 1.0 / penalty_scores_arr
    
    # Storage for predictions
    predictions = []
    
    # Storage for gridsearch results from each LOO fold
    loo_gridsearch_results = []
    
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
        
        # Ensure no NaN/Inf before SMOTE (e.g. from constant columns in scale_cols)
        X_train_scaled = pd.DataFrame(
            np.nan_to_num(X_train_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_train_scaled.columns,
            index=X_train_scaled.index
        )
        X_test_scaled = pd.DataFrame(
            np.nan_to_num(X_test_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_test_scaled.columns,
            index=X_test_scaled.index
        )
        
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
                lmda_path_size=lmda_path_size,  # At least 50 lambda values
                min_ratio=0.01,
                alpha=1.0,  # Pure L1
                penalty=pf,
                n_threads=n_threads,
                progress_bar=False
            )
            
            # Get best lambda index using the specified optimization metric
            best_lambda_idx = select_best_lambda(fit, optimize_metric)
            best_lambda = fit.lmdas[best_lambda_idx]
            best_accuracy = 1.0 - fit.test_error[best_lambda_idx]
            
            # Debug log for each fold
            sens_val = fit.sensitivity[best_lambda_idx] if fit.sensitivity is not None else 0.0
            spec_val = fit.specificity[best_lambda_idx] if fit.specificity is not None else 0.0
            logger.debug(f"  [LLM-LASSO] Fold {i+1}: Best λ={best_lambda:.2e}, "
                        f"Acc={best_accuracy:.4f}, Sens={sens_val:.4f}, Spec={spec_val:.4f}")
            
            # Store gridsearch results for this LOO fold (including extended metrics)
            loo_gridsearch_results.append({
                'fold': i,
                'lmdas': fit.lmdas.copy(),
                'test_error': fit.test_error.copy(),  # Hamming loss
                'sensitivity': fit.sensitivity.copy() if fit.sensitivity is not None else None,
                'specificity': fit.specificity.copy() if fit.specificity is not None else None,
                'balanced_accuracy': fit.balanced_accuracy.copy() if fit.balanced_accuracy is not None else None,
                'f1': fit.f1.copy() if fit.f1 is not None else None,
                'avg_losses': fit.avg_losses.copy(),
                'roc_auc': fit.roc_auc.copy() if fit.roc_auc is not None else None,
                'best_idx': best_lambda_idx,
                'best_lambda': best_lambda
            })
            
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
        
        # Extract coefficients at best lambda
        try:
            coef_raw = model.betas[best_lambda_idx, :]
            # Handle sparse matrix (csr_matrix) - convert to dense array
            if hasattr(coef_raw, 'toarray'):
                coef_at_best = np.asarray(coef_raw.toarray()).flatten()
            elif hasattr(coef_raw, 'todense'):
                coef_at_best = np.asarray(coef_raw.todense()).flatten()
            else:
                coef_at_best = np.asarray(coef_raw).flatten()
            intercept_at_best = float(model.intercepts[best_lambda_idx])
        except Exception as e:
            logger.debug(f"  LOO fold {i+1}: Could not extract coefficients: {e}")
            coef_at_best = np.zeros(X.shape[1])
            intercept_at_best = 0.0
        
        # Store prediction and coefficients
        predictions.append({
            'participant_id': participant_ids[i],
            'actual_label': int(y_test.iloc[0]),
            'predicted_probability': float(prob),
            'coefficients': coef_at_best,
            'intercept': float(intercept_at_best)
        })
        
        # Progress logging every 10%
        if (i + 1) % max(1, n_samples // 10) == 0 or i == n_samples - 1:
            logger.info(f"  LOO progress: {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
    
    loo_time = time.time() - start_time
    logger.info(f"LOO cross-validation completed in {loo_time:.2f}s")
    
    # Summary debug logging for LLM-Lasso
    logger.debug("")
    logger.debug("=" * 50)
    logger.debug("[LLM-LASSO] LOO SUMMARY")
    logger.debug("=" * 50)
    if loo_gridsearch_results:
        all_best_lambdas = [r['best_lambda'] for r in loo_gridsearch_results]
        logger.debug(f"  Best λ range: [{min(all_best_lambdas):.2e}, {max(all_best_lambdas):.2e}]")
        logger.debug(f"  Best λ median: {np.median(all_best_lambdas):.2e}")
        all_best_acc = [1.0 - r['test_error'][r['best_idx']] for r in loo_gridsearch_results]
        logger.debug(f"  Inner CV Accuracy: mean={np.mean(all_best_acc):.4f}, std={np.std(all_best_acc):.4f}")
    
    # Extract and aggregate coefficients from all LOO folds
    logger.info("")
    logger.info("Extracting and aggregating model coefficients...")
    feature_names = list(X.columns)
    all_coefficients = np.array([p['coefficients'] for p in predictions])
    all_intercepts = np.array([p['intercept'] for p in predictions])
    
    # Calculate mean and std of coefficients across folds
    mean_coefficients = np.mean(all_coefficients, axis=0)
    std_coefficients = np.std(all_coefficients, axis=0)
    mean_intercept = np.mean(all_intercepts)
    std_intercept = np.std(all_intercepts)
    
    # Count how many folds each feature was non-zero (selection frequency)
    non_zero_counts = np.sum(all_coefficients != 0, axis=0)
    selection_frequency = non_zero_counts / n_samples
    
    logger.info(f"  Features with non-zero mean coefficient: {np.sum(mean_coefficients != 0)}/{len(feature_names)}")
    logger.info(f"  Features selected in >50% of folds: {np.sum(selection_frequency > 0.5)}/{len(feature_names)}")
    
    # Train a final model on ALL data for stable coefficient estimates
    logger.info("Training final model on all data for stable coefficients...")
    try:
        X_all_scaled = scale_cols(X)
        
        # Ensure no NaN/Inf before SMOTE (e.g. from constant columns in scale_cols)
        X_all_scaled = pd.DataFrame(
            np.nan_to_num(X_all_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_all_scaled.columns,
            index=X_all_scaled.index
        )
        
        # Apply SMOTE if enabled for final model
        if use_smote:
            smote_final = SMOTE(random_state=smote_random_state, k_neighbors=min(5, y.value_counts().min() - 1))
            X_final_resampled, y_final_resampled = smote_final.fit_resample(X_all_scaled.to_numpy(), y.to_numpy())
        else:
            X_final_resampled = X_all_scaled.to_numpy()
            y_final_resampled = y.to_numpy()
        
        glm_final = ad.glm.binomial(y=y_final_resampled, dtype=np.float64)
        
        # Use median lambda selection across folds via CV on all data
        fit_final = cv_grpnet(
            X=X_final_resampled,
            glm=glm_final,
            seed=42,
            n_folds=inner_cv_folds,
            lmda_path_size=lmda_path_size,  # At least 50 lambda values
            min_ratio=0.01,
            alpha=1.0,
            penalty=pf,
            n_threads=n_threads,
            progress_bar=False
        )
        
        best_lambda_final = select_best_lambda(fit_final, optimize_metric)
        
        # Save gridsearch plots for the final model
        logger.info("Saving gridsearch parameter search plots...")
        save_gridsearch_plots(fit_final, save_dir, logger, fold_label="final_model", optimize_metric=optimize_metric)
        
        # Save gridsearch plots for each LOO fold and aggregate summary
        logger.info("Saving LOO fold gridsearch plots...")
        save_loo_gridsearch_plots(loo_gridsearch_results, save_dir, logger, optimize_metric=optimize_metric)
        
        model_final = grpnet(
            X=X_final_resampled,
            glm=glm_final,
            ddev_tol=0,
            early_exit=False,
            n_threads=n_threads,
            min_ratio=0.01,
            progress_bar=False,
            alpha=1.0,
            penalty=pf,
        )
        
        coef_raw_final = model_final.betas[best_lambda_final, :]
        # Handle sparse matrix (csr_matrix) - convert to dense array
        if hasattr(coef_raw_final, 'toarray'):
            final_coefficients = np.asarray(coef_raw_final.toarray()).flatten()
        elif hasattr(coef_raw_final, 'todense'):
            final_coefficients = np.asarray(coef_raw_final.todense()).flatten()
        else:
            final_coefficients = np.asarray(coef_raw_final).flatten()
        final_intercept = float(model_final.intercepts[best_lambda_final])
        logger.info(f"  Final model: {np.sum(final_coefficients != 0)} non-zero coefficients")
    except Exception as e:
        logger.warning(f"Could not train final model: {e}")
        final_coefficients = mean_coefficients
        final_intercept = mean_intercept
    
    # Create coefficients dictionary
    coefficients_data = {
        "feature_names": feature_names,
        "final_model": {
            "coefficients": {name: float(coef) for name, coef in zip(feature_names, final_coefficients)},
            "intercept": float(final_intercept),
            "n_nonzero": int(np.sum(final_coefficients != 0))
        },
        "loo_aggregated": {
            "mean_coefficients": {name: float(coef) for name, coef in zip(feature_names, mean_coefficients)},
            "std_coefficients": {name: float(std) for name, std in zip(feature_names, std_coefficients)},
            "selection_frequency": {name: float(freq) for name, freq in zip(feature_names, selection_frequency)},
            "mean_intercept": float(mean_intercept),
            "std_intercept": float(std_intercept)
        }
    }
    
    # Save coefficients to JSON
    coef_file = os.path.join(save_dir, "model_coefficients.json")
    with open(coef_file, 'w') as f:
        json.dump(coefficients_data, f, indent=2)
    logger.info(f"Coefficients saved to: {coef_file}")
    
    # Generate coefficient bar plot
    generate_coefficient_plot(
        feature_names=feature_names,
        coefficients=final_coefficients,
        mean_coefficients=mean_coefficients,
        std_coefficients=std_coefficients,
        selection_frequency=selection_frequency,
        save_dir=save_dir,
        logger=logger
    )
    
    # Create predictions DataFrame (without coefficients column for CSV)
    predictions_for_csv = [{
        'participant_id': p['participant_id'],
        'actual_label': p['actual_label'],
        'predicted_probability': p['predicted_probability']
    } for p in predictions]
    predictions_df = pd.DataFrame(predictions_for_csv)
    
    # Save predictions CSV
    predictions_file = os.path.join(save_dir, "loo_predictions.csv")
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Extract arrays for evaluation
    y_true = predictions_df['actual_label'].values
    y_prob = predictions_df['predicted_probability'].values
    y_pred = (y_prob > 0.5).astype(int)
    
    # Generate comprehensive evaluation plots and compute all metrics
    metrics = generate_evaluation_plots(y_true, y_prob, save_dir, logger)
    
    # Compute bootstrap confidence intervals
    ci_results = None
    if compute_ci:
        logger.info("")
        logger.info("Computing bootstrap confidence intervals...")
        try:
            from llm_lasso.utils.bootstrap_ci import compute_bootstrap_ci, plot_confidence_intervals
            
            # Create confidence_intervals subfolder
            ci_dir = os.path.join(save_dir, "confidence_intervals")
            os.makedirs(ci_dir, exist_ok=True)
            
            ci_results = compute_bootstrap_ci(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                method=bootstrap_method,
                n_rounds=bootstrap_n_rounds,
                ci_level=ci_level,
                random_seed=42,
                logger=logger
            )
            
            # Save confidence intervals to JSON in subfolder
            ci_file = os.path.join(ci_dir, "confidence_intervals.json")
            with open(ci_file, 'w') as f:
                json.dump({
                    "method": bootstrap_method,
                    "n_rounds": bootstrap_n_rounds,
                    "ci_level": ci_level,
                    "metrics": ci_results
                }, f, indent=2)
            logger.info(f"Confidence intervals saved to: {ci_file}")
            
            # Generate and save CI visualization plots
            logger.info("Generating confidence interval plots...")
            plot_confidence_intervals(
                ci_results=ci_results,
                save_dir=ci_dir,
                ci_level=ci_level,
                logger=logger
            )
            
        except ImportError as e:
            logger.warning(f"Could not import bootstrap module: {e}")
            logger.warning("Install mlxtend: pip install mlxtend>=0.23.0")
        except Exception as e:
            logger.warning(f"Error computing confidence intervals: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
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
        "bootstrap_ci": {
            "computed": ci_results is not None,
            "method": bootstrap_method if ci_results else None,
            "n_rounds": bootstrap_n_rounds if ci_results else None,
            "ci_level": ci_level if ci_results else None
        },
        **metrics  # Include all computed metrics
    }
    
    # Add CI bounds to summary for key metrics
    if ci_results:
        for metric_name in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1', 'auroc']:
            if metric_name in ci_results:
                ci_data = ci_results[metric_name]
                summary[f"{metric_name}_ci_lower"] = ci_data.get('ci_lower')
                summary[f"{metric_name}_ci_upper"] = ci_data.get('ci_upper')
    
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    # Log final results
    logger.info("")
    logger.info("=" * 60)
    logger.info("LOO CROSS-VALIDATION FINAL RESULTS")
    logger.info("=" * 60)
    
    # Helper function to format metric with optional CI
    def format_metric_log(name: str, value: float, ci_key: str = None) -> str:
        if ci_results and ci_key and ci_key in ci_results:
            ci = ci_results[ci_key]
            if ci.get('ci_lower') is not None and ci.get('ci_upper') is not None:
                return f"{name}: {value:.4f} ({int(ci_level*100)}% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f})"
        return f"{name}: {value:.4f}"
    
    logger.info(format_metric_log("AUROC", metrics['auroc'], 'auroc'))
    logger.info(format_metric_log("Accuracy", metrics['accuracy'], 'accuracy') + f" ({int(metrics['accuracy'] * n_samples)}/{n_samples} correct)")
    logger.info(format_metric_log("Balanced Accuracy", metrics['balanced_accuracy'], 'balanced_accuracy'))
    logger.info(format_metric_log("Sensitivity", metrics['sensitivity'], 'sensitivity'))
    logger.info(format_metric_log("Specificity", metrics['specificity'], 'specificity'))
    logger.info(format_metric_log("F1 Score", metrics['f1_score'], 'f1'))
    logger.info(format_metric_log("MCC", metrics['mcc'], 'mcc'))
    
    return {
        "predictions": predictions_df,
        "summary": summary
    }


def run_standard_lasso_with_loo(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: str,
    inner_cv_folds: int,
    n_threads: int,
    logger: logging.Logger,
    participant_ids: Optional[pd.Index] = None,
    use_smote: bool = False,
    smote_random_state: int = 42,
    lmda_path_size: int = 100,
    optimize_metric: str = "accuracy",
    compute_ci: bool = True,
    bootstrap_method: str = "standard",
    bootstrap_n_rounds: int = 1000,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """
    Run STANDARD Lasso (uniform penalties) with Leave-One-Out outer CV and inner k-fold CV.
    
    This is identical to run_lasso_with_loo() but uses uniform penalty factors
    instead of LLM-generated penalties. This serves as a baseline for comparison.
    
    Args:
        X: Feature matrix (all samples)
        y: Target labels (all samples)
        save_dir: Directory to save results (should be standard_lasso subfolder)
        inner_cv_folds: Number of inner CV folds for hyperparameter selection
        n_threads: Number of threads
        logger: Logger instance
        participant_ids: Optional participant IDs (uses index if not provided)
        use_smote: Whether to apply SMOTE to balance training data
        smote_random_state: Random state for SMOTE reproducibility
        lmda_path_size: Number of lambda values in the regularization path
        optimize_metric: Metric to maximize during hyperparameter selection
        compute_ci: Whether to compute bootstrap confidence intervals
        bootstrap_method: Bootstrap method ('standard' or '632')
        bootstrap_n_rounds: Number of bootstrap iterations
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        Dictionary with predictions and evaluation results
    """
    logger.info("=" * 60)
    logger.info("STANDARD LASSO WITH LEAVE-ONE-OUT CROSS-VALIDATION")
    logger.info("=" * 60)
    logger.info("Using UNIFORM penalty factors (baseline comparison)")
    
    if not ADELIE_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("ADELIE NOT INSTALLED - SKIPPING STANDARD LASSO")
        logger.warning("=" * 60)
        return {"predictions": None, "summary": {"lasso_skipped": True}}
    
    # Check SMOTE availability
    if use_smote and not SMOTE_AVAILABLE:
        logger.warning("SMOTE requested but imbalanced-learn not installed!")
        logger.warning("Proceeding without SMOTE...")
        use_smote = False
    
    n_samples = len(X)
    n_features = X.shape[1]
    class_counts = y.value_counts().sort_index()
    
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Features: {n_features}")
    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Class imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1")
    logger.info(f"Inner CV folds: {inner_cv_folds}")
    logger.info(f"Outer CV: Leave-One-Out ({n_samples} iterations)")
    logger.info(f"Lambda path size: {lmda_path_size} (regularization parameters)")
    logger.info(f"Optimization metric: {optimize_metric}")
    logger.info(f"Threads: {n_threads}")
    logger.info(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    logger.info(f"Bootstrap CI: {'Enabled' if compute_ci else 'Disabled'}")
    
    # Use index as participant IDs if not provided
    if participant_ids is None:
        participant_ids = X.index
    
    # Storage for predictions
    predictions = []
    
    # Storage for gridsearch results from each LOO fold
    loo_gridsearch_results = []
    
    # LOO outer loop
    logger.info("")
    logger.info("Starting LOO cross-validation (Standard Lasso)...")
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
        
        # Ensure no NaN/Inf before SMOTE (e.g. from constant columns in scale_cols)
        X_train_scaled = pd.DataFrame(
            np.nan_to_num(X_train_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_train_scaled.columns,
            index=X_train_scaled.index
        )
        X_test_scaled = pd.DataFrame(
            np.nan_to_num(X_test_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_test_scaled.columns,
            index=X_test_scaled.index
        )
        
        # Apply SMOTE to balance training data (if enabled)
        if use_smote:
            try:
                smote = SMOTE(random_state=smote_random_state + i, k_neighbors=min(5, y_train.value_counts().min() - 1))
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train_scaled.to_numpy(), 
                    y_train.to_numpy()
                )
            except Exception as e:
                logger.debug(f"  LOO fold {i+1}: SMOTE failed ({e}), using original data")
                X_train_resampled = X_train_scaled.to_numpy()
                y_train_resampled = y_train.to_numpy()
        else:
            X_train_resampled = X_train_scaled.to_numpy()
            y_train_resampled = y_train.to_numpy()
        
        # Initialize GLM for binomial classification
        glm_train = ad.glm.binomial(y=y_train_resampled, dtype=np.float64)
        
        # STANDARD LASSO: Use uniform penalty factors (all features penalized equally)
        pf = np.ones(X_train_resampled.shape[1])
        pf = pf / np.sum(pf) * X_train_resampled.shape[1]  # Normalize same as LLM-Lasso
        
        try:
            # Inner CV for lambda selection
            fit = cv_grpnet(
                X=X_train_resampled,
                glm=glm_train,
                seed=42 + i,
                n_folds=inner_cv_folds,
                lmda_path_size=lmda_path_size,
                min_ratio=0.01,
                alpha=1.0,  # Pure L1
                penalty=pf,
                n_threads=n_threads,
                progress_bar=False
            )
            
            # Get best lambda index using the specified optimization metric
            best_lambda_idx = select_best_lambda(fit, optimize_metric)
            best_lambda = fit.lmdas[best_lambda_idx]
            best_accuracy = 1.0 - fit.test_error[best_lambda_idx]
            
            # Debug log for each fold
            sens_val = fit.sensitivity[best_lambda_idx] if fit.sensitivity is not None else 0.0
            spec_val = fit.specificity[best_lambda_idx] if fit.specificity is not None else 0.0
            logger.debug(f"  [STD-LASSO] Fold {i+1}: Best λ={best_lambda:.2e}, "
                        f"Acc={best_accuracy:.4f}, Sens={sens_val:.4f}, Spec={spec_val:.4f}")
            
            # Store gridsearch results for this LOO fold
            loo_gridsearch_results.append({
                'fold': i,
                'lmdas': fit.lmdas.copy(),
                'test_error': fit.test_error.copy(),
                'sensitivity': fit.sensitivity.copy() if fit.sensitivity is not None else None,
                'specificity': fit.specificity.copy() if fit.specificity is not None else None,
                'balanced_accuracy': fit.balanced_accuracy.copy() if fit.balanced_accuracy is not None else None,
                'f1': fit.f1.copy() if fit.f1 is not None else None,
                'avg_losses': fit.avg_losses.copy(),
                'roc_auc': fit.roc_auc.copy() if fit.roc_auc is not None else None,
                'best_idx': best_lambda_idx,
                'best_lambda': best_lambda
            })
            
            # Train final model with best lambda on all training data
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
            
            # Predict on test sample
            etas = predict(
                X=X_test_scaled.to_numpy(),
                betas=model.betas,
                intercepts=model.intercepts,
                n_threads=n_threads,
            )
            
            # Convert eta to probability using sigmoid
            eta_best = etas[best_lambda_idx, 0]
            prob = 1.0 / (1.0 + np.exp(-eta_best))
            
        except Exception as e:
            logger.warning(f"  LOO fold {i+1}/{n_samples}: Error - {e}, using 0.5 probability")
            prob = 0.5
        
        # Extract coefficients at best lambda
        try:
            coef_raw = model.betas[best_lambda_idx, :]
            if hasattr(coef_raw, 'toarray'):
                coef_at_best = np.asarray(coef_raw.toarray()).flatten()
            elif hasattr(coef_raw, 'todense'):
                coef_at_best = np.asarray(coef_raw.todense()).flatten()
            else:
                coef_at_best = np.asarray(coef_raw).flatten()
            intercept_at_best = float(model.intercepts[best_lambda_idx])
        except Exception as e:
            logger.debug(f"  LOO fold {i+1}: Could not extract coefficients: {e}")
            coef_at_best = np.zeros(X.shape[1])
            intercept_at_best = 0.0
        
        # Store prediction and coefficients
        predictions.append({
            'participant_id': participant_ids[i],
            'actual_label': int(y_test.iloc[0]),
            'predicted_probability': float(prob),
            'coefficients': coef_at_best,
            'intercept': float(intercept_at_best)
        })
        
        # Progress logging every 10%
        if (i + 1) % max(1, n_samples // 10) == 0 or i == n_samples - 1:
            logger.info(f"  LOO progress (Standard Lasso): {i+1}/{n_samples} ({100*(i+1)/n_samples:.0f}%)")
    
    loo_time = time.time() - start_time
    logger.info(f"Standard Lasso LOO cross-validation completed in {loo_time:.2f}s")
    
    # Summary debug logging for Standard Lasso
    logger.debug("")
    logger.debug("=" * 50)
    logger.debug("[STD-LASSO] LOO SUMMARY")
    logger.debug("=" * 50)
    if loo_gridsearch_results:
        all_best_lambdas = [r['best_lambda'] for r in loo_gridsearch_results]
        logger.debug(f"  Best λ range: [{min(all_best_lambdas):.2e}, {max(all_best_lambdas):.2e}]")
        logger.debug(f"  Best λ median: {np.median(all_best_lambdas):.2e}")
        all_best_acc = [1.0 - r['test_error'][r['best_idx']] for r in loo_gridsearch_results]
        logger.debug(f"  Inner CV Accuracy: mean={np.mean(all_best_acc):.4f}, std={np.std(all_best_acc):.4f}")
    
    # Extract and aggregate coefficients from all LOO folds
    logger.info("")
    logger.info("Extracting and aggregating model coefficients...")
    feature_names = list(X.columns)
    all_coefficients = np.array([p['coefficients'] for p in predictions])
    all_intercepts = np.array([p['intercept'] for p in predictions])
    
    # Calculate mean and std of coefficients across folds
    mean_coefficients = np.mean(all_coefficients, axis=0)
    std_coefficients = np.std(all_coefficients, axis=0)
    mean_intercept = np.mean(all_intercepts)
    std_intercept = np.std(all_intercepts)
    
    # Count how many folds each feature was non-zero
    non_zero_counts = np.sum(all_coefficients != 0, axis=0)
    selection_frequency = non_zero_counts / n_samples
    
    logger.info(f"  Features with non-zero mean coefficient: {np.sum(mean_coefficients != 0)}/{len(feature_names)}")
    logger.info(f"  Features selected in >50% of folds: {np.sum(selection_frequency > 0.5)}/{len(feature_names)}")
    
    # Train a final model on ALL data for stable coefficient estimates
    logger.info("Training final Standard Lasso model on all data...")
    try:
        X_all_scaled = scale_cols(X)
        
        # Ensure no NaN/Inf before SMOTE (e.g. from constant columns in scale_cols)
        X_all_scaled = pd.DataFrame(
            np.nan_to_num(X_all_scaled.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
            columns=X_all_scaled.columns,
            index=X_all_scaled.index
        )
        
        # Apply SMOTE if enabled for final model
        if use_smote:
            smote_final = SMOTE(random_state=smote_random_state, k_neighbors=min(5, y.value_counts().min() - 1))
            X_final_resampled, y_final_resampled = smote_final.fit_resample(X_all_scaled.to_numpy(), y.to_numpy())
        else:
            X_final_resampled = X_all_scaled.to_numpy()
            y_final_resampled = y.to_numpy()
        
        glm_final = ad.glm.binomial(y=y_final_resampled, dtype=np.float64)
        
        # UNIFORM PENALTIES for final model
        pf_final = np.ones(X_final_resampled.shape[1])
        pf_final = pf_final / np.sum(pf_final) * X_final_resampled.shape[1]
        
        # CV on all data for lambda selection
        fit_final = cv_grpnet(
            X=X_final_resampled,
            glm=glm_final,
            seed=42,
            n_folds=inner_cv_folds,
            lmda_path_size=lmda_path_size,
            min_ratio=0.01,
            alpha=1.0,
            penalty=pf_final,
            n_threads=n_threads,
            progress_bar=False
        )
        
        best_lambda_final = select_best_lambda(fit_final, optimize_metric)
        
        # Save gridsearch plots for the final model
        logger.info("Saving gridsearch parameter search plots...")
        save_gridsearch_plots(fit_final, save_dir, logger, fold_label="final_model", optimize_metric=optimize_metric)
        
        # Save gridsearch plots for each LOO fold
        logger.info("Saving LOO fold gridsearch plots...")
        save_loo_gridsearch_plots(loo_gridsearch_results, save_dir, logger, optimize_metric=optimize_metric)
        
        model_final = grpnet(
            X=X_final_resampled,
            glm=glm_final,
            ddev_tol=0,
            early_exit=False,
            n_threads=n_threads,
            min_ratio=0.01,
            progress_bar=False,
            alpha=1.0,
            penalty=pf_final,
        )
        
        coef_raw_final = model_final.betas[best_lambda_final, :]
        if hasattr(coef_raw_final, 'toarray'):
            final_coefficients = np.asarray(coef_raw_final.toarray()).flatten()
        elif hasattr(coef_raw_final, 'todense'):
            final_coefficients = np.asarray(coef_raw_final.todense()).flatten()
        else:
            final_coefficients = np.asarray(coef_raw_final).flatten()
        final_intercept = float(model_final.intercepts[best_lambda_final])
        logger.info(f"  Final model: {np.sum(final_coefficients != 0)} non-zero coefficients")
    except Exception as e:
        logger.warning(f"Could not train final model: {e}")
        final_coefficients = mean_coefficients
        final_intercept = mean_intercept
    
    # Create coefficients dictionary
    coefficients_data = {
        "method": "standard_lasso",
        "feature_names": feature_names,
        "final_model": {
            "coefficients": {name: float(coef) for name, coef in zip(feature_names, final_coefficients)},
            "intercept": float(final_intercept),
            "n_nonzero": int(np.sum(final_coefficients != 0))
        },
        "loo_aggregated": {
            "mean_coefficients": {name: float(coef) for name, coef in zip(feature_names, mean_coefficients)},
            "std_coefficients": {name: float(std) for name, std in zip(feature_names, std_coefficients)},
            "selection_frequency": {name: float(freq) for name, freq in zip(feature_names, selection_frequency)},
            "mean_intercept": float(mean_intercept),
            "std_intercept": float(std_intercept)
        }
    }
    
    # Save coefficients to JSON
    coef_file = os.path.join(save_dir, "model_coefficients.json")
    with open(coef_file, 'w') as f:
        json.dump(coefficients_data, f, indent=2)
    logger.info(f"Coefficients saved to: {coef_file}")
    
    # Generate coefficient bar plot
    generate_coefficient_plot(
        feature_names=feature_names,
        coefficients=final_coefficients,
        mean_coefficients=mean_coefficients,
        std_coefficients=std_coefficients,
        selection_frequency=selection_frequency,
        save_dir=save_dir,
        logger=logger
    )
    
    # Create predictions DataFrame
    predictions_for_csv = [{
        'participant_id': p['participant_id'],
        'actual_label': p['actual_label'],
        'predicted_probability': p['predicted_probability']
    } for p in predictions]
    predictions_df = pd.DataFrame(predictions_for_csv)
    
    # Save predictions CSV
    predictions_file = os.path.join(save_dir, "loo_predictions.csv")
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Extract arrays for evaluation
    y_true = predictions_df['actual_label'].values
    y_prob = predictions_df['predicted_probability'].values
    y_pred = (y_prob > 0.5).astype(int)
    
    # Generate comprehensive evaluation plots and compute all metrics
    metrics = generate_evaluation_plots(y_true, y_prob, save_dir, logger)
    
    # Compute bootstrap confidence intervals
    ci_results = None
    if compute_ci:
        logger.info("")
        logger.info("Computing bootstrap confidence intervals...")
        try:
            from llm_lasso.utils.bootstrap_ci import compute_bootstrap_ci, plot_confidence_intervals
            
            ci_dir = os.path.join(save_dir, "confidence_intervals")
            os.makedirs(ci_dir, exist_ok=True)
            
            ci_results = compute_bootstrap_ci(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                method=bootstrap_method,
                n_rounds=bootstrap_n_rounds,
                ci_level=ci_level,
                random_seed=42,
                logger=logger
            )
            
            # Save confidence intervals to JSON
            ci_file = os.path.join(ci_dir, "confidence_intervals.json")
            with open(ci_file, 'w') as f:
                json.dump({
                    "method": bootstrap_method,
                    "n_rounds": bootstrap_n_rounds,
                    "ci_level": ci_level,
                    "metrics": ci_results
                }, f, indent=2)
            logger.info(f"Confidence intervals saved to: {ci_file}")
            
            # Generate and save CI visualization plots
            logger.info("Generating confidence interval plots...")
            plot_confidence_intervals(
                ci_results=ci_results,
                save_dir=ci_dir,
                ci_level=ci_level,
                logger=logger
            )
            
        except ImportError as e:
            logger.warning(f"Could not import bootstrap module: {e}")
        except Exception as e:
            logger.warning(f"Error computing confidence intervals: {e}")
    
    # Create summary with all metrics
    class_counts = y.value_counts().sort_index()
    summary = {
        "method": "standard_lasso",
        "cv_method": "Leave-One-Out",
        "inner_cv_folds": inner_cv_folds,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_imbalance_ratio": float(class_counts.max() / class_counts.min()),
        "smote_enabled": use_smote,
        "loo_time_seconds": float(loo_time),
        "bootstrap_ci": {
            "computed": ci_results is not None,
            "method": bootstrap_method if ci_results else None,
            "n_rounds": bootstrap_n_rounds if ci_results else None,
            "ci_level": ci_level if ci_results else None
        },
        **metrics
    }
    
    # Add CI bounds to summary for key metrics
    if ci_results:
        for metric_name in ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1', 'auroc']:
            if metric_name in ci_results:
                ci_data = ci_results[metric_name]
                summary[f"{metric_name}_ci_lower"] = ci_data.get('ci_lower')
                summary[f"{metric_name}_ci_upper"] = ci_data.get('ci_upper')
    
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    # Log final results
    logger.info("")
    logger.info("=" * 60)
    logger.info("STANDARD LASSO LOO CROSS-VALIDATION FINAL RESULTS")
    logger.info("=" * 60)
    
    def format_metric_log(name: str, value: float, ci_key: str = None) -> str:
        if ci_results and ci_key and ci_key in ci_results:
            ci = ci_results[ci_key]
            if ci.get('ci_lower') is not None and ci.get('ci_upper') is not None:
                return f"{name}: {value:.4f} ({int(ci_level*100)}% CI: {ci['ci_lower']:.4f} - {ci['ci_upper']:.4f})"
        return f"{name}: {value:.4f}"
    
    logger.info(format_metric_log("AUROC", metrics['auroc'], 'auroc'))
    logger.info(format_metric_log("Accuracy", metrics['accuracy'], 'accuracy') + f" ({int(metrics['accuracy'] * n_samples)}/{n_samples} correct)")
    logger.info(format_metric_log("Balanced Accuracy", metrics['balanced_accuracy'], 'balanced_accuracy'))
    logger.info(format_metric_log("Sensitivity", metrics['sensitivity'], 'sensitivity'))
    logger.info(format_metric_log("Specificity", metrics['specificity'], 'specificity'))
    logger.info(format_metric_log("F1 Score", metrics['f1_score'], 'f1'))
    logger.info(format_metric_log("MCC", metrics['mcc'], 'mcc'))
    
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
    logger: logging.Logger,
    lmda_path_size: int = 100
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
        lmda_path_size: Number of lambda values in the regularization path (at least 50)
    
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
    logger.info(f"Lambda path size: {lmda_path_size} (regularization parameters)")
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
        max_imp_pow=5,
        lmda_path_size=lmda_path_size
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


def run_standard_lasso_with_penalties(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str,
    n_threads: int,
    folds_cv: int,
    logger: logging.Logger,
    lmda_path_size: int = 100
) -> Dict[str, Any]:
    """
    Run STANDARD Lasso classification with uniform penalty factors (baseline).
    
    This is identical to run_lasso_with_penalties() but uses uniform penalty
    factors instead of LLM-generated penalties.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        save_dir: Directory to save results (should be standard_lasso subfolder)
        n_threads: Number of threads
        folds_cv: Number of CV folds
        logger: Logger instance
        lmda_path_size: Number of lambda values in the regularization path
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 60)
    logger.info("STANDARD LASSO CLASSIFICATION")
    logger.info("=" * 60)
    logger.info("Using UNIFORM penalty factors (baseline comparison)")
    
    if not ADELIE_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("ADELIE NOT INSTALLED - SKIPPING STANDARD LASSO")
        logger.warning("=" * 60)
        
        summary = {
            "lasso_skipped": True,
            "method": "standard_lasso",
            "reason": "adelie not installed",
            "total_features": X_train.shape[1],
            "training_samples": len(X_train),
            "test_samples": len(X_test)
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
    
    logger.info("Running Standard Lasso CV (uniform penalties)...")
    logger.info(f"Lambda path size: {lmda_path_size} (regularization parameters)")
    start_time = time.time()
    
    # STANDARD LASSO: Use uniform penalties (importance = 1 for all features)
    # Since llm_lasso_cv uses penalty = 1/score when score_type=PenaltyType.IMP,
    # we pass uniform scores of 1.0 to get uniform penalties
    uniform_scores = np.ones(X_train.shape[1])
    
    results = llm_lasso_cv(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        score=uniform_scores,
        regression=False,  # Classification
        score_type=PenaltyType.IMP,  # Importance scores (1/imp = 1/1 = uniform penalty)
        folds_cv=folds_cv,
        seed=42,
        n_threads=n_threads,
        alpha=1.0,  # Pure L1
        max_imp_pow=0,  # Only use 1/imp^0 = uniform penalties
        lmda_path_size=lmda_path_size
    )
    
    lasso_time = time.time() - start_time
    logger.info(f"Standard Lasso training completed in {lasso_time:.2f}s")
    
    # Log results
    logger.info("=" * 60)
    logger.info("STANDARD LASSO RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Results shape: {results.shape}")
    logger.debug(f"Results columns: {list(results.columns)}")
    
    # Find best result (filter to only Lasso results, not 1/imp variants)
    lasso_results = results[results['method'] == 'Lasso']
    if len(lasso_results) == 0:
        lasso_results = results
    
    best_idx = lasso_results['test_error'].idxmin()
    best_result = results.loc[best_idx]
    
    logger.info(f"Best Model:")
    logger.info(f"  Method: {best_result.get('method', 'N/A')}")
    logger.info(f"  Test error: {best_result['test_error']:.4f}")
    logger.info(f"  AUROC: {best_result.get('auroc', 'N/A')}")
    logger.info(f"  Number of features: {best_result['n_features']}")
    
    # Save results
    results_file = os.path.join(save_dir, "lasso_results.csv")
    results.to_csv(results_file, index=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Save summary
    summary = {
        "method": "standard_lasso",
        "best_method": str(best_result.get('method', 'N/A')),
        "best_method_model": str(best_result.get('best_method_model', 'N/A')),
        "test_error": float(best_result['test_error']),
        "accuracy": float(1 - best_result['test_error']),  # Accuracy = 1 - test_error
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
    pdf_collection_name: str = field(default="pdf_documents", metadata={
        "help": "Name of the PDF vectorstore collection (must match index_pdf_vectorstore.py --collection-name, default: pdf_documents)"
    })
    pdf_rag_num_docs: int = field(default=3, metadata={
        "help": "Number of PDF documents to retrieve for RAG"
    })
    
    # LLM options
    llm_backend: str = field(default="openai", metadata={
        "help": "LLM backend: openai (cloud API) or vllm (local open-source)",
        "choices": ["openai", "vllm"]
    })
    model_type: str = field(default="gpt-4o", metadata={
        "help": "LLM model type: gpt-4o, o1, o1-pro, openrouter (for openai backend), or vllm model name",
        "choices": ["gpt-4o", "o1", "o1-pro", "openrouter", "vllm"]
    })
    model_name: Optional[str] = field(default=None, metadata={
        "help": "Specific model name (optional). For vllm backend, defaults to VLLM_CHAT_MODEL env var."
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
    lmda_path_size: int = field(default=100, metadata={
        "help": "Number of lambda (regularization) parameters to search over. Must be at least 50 for adequate grid search."
    })
    optimize_metric: str = field(default="accuracy", metadata={
        "help": "Metric to maximize during hyperparameter selection. Default 'accuracy' minimizes Hamming loss.",
        "choices": ["accuracy", "sensitivity", "specificity", "balanced_accuracy", "f1", "auc_roc"]
    })
    
    # Bootstrap confidence intervals
    compute_ci: bool = field(default=True, metadata={
        "help": "Compute bootstrap confidence intervals for performance metrics"
    })
    bootstrap_method: str = field(default="standard", metadata={
        "help": "Bootstrap method: 'standard' (simple resampling) or '632' (.632 bootstrap for reduced bias)",
        "choices": ["standard", "632"]
    })
    bootstrap_n_rounds: int = field(default=1000, metadata={
        "help": "Number of bootstrap iterations for confidence intervals"
    })
    ci_level: float = field(default=0.95, metadata={
        "help": "Confidence level for intervals (e.g., 0.95 for 95% CI)"
    })
    
    # Class imbalance handling
    use_smote: bool = field(default=False, metadata={
        "help": "Use SMOTE to balance classes in training data (applied within each CV fold)"
    })
    smote_random_state: int = field(default=42, metadata={
        "help": "Random state for SMOTE reproducibility"
    })
    skip_standard_lasso: bool = field(default=False, metadata={
        "help": "Skip Standard Lasso and comparison plots; run only LLM-Lasso"
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
        
        # Step 2b: Validate data for LOO (target NaN/Inf, binary target) — exits with clear message if invalid
        validate_loo_data(X, y, args.target_column, args.dataset_path, logger)
        
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
                logger,
                llm_backend=args.llm_backend
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
            logger=logger,
            llm_backend=args.llm_backend
        )
        
        # Step 6b: MANDATORY - Save RAG retrieved documents when PDF RAG is enabled
        if args.pdf_rag and pdf_vectorstore is not None:
            save_rag_retrieved_documents(
                feature_names=validated_features,
                category=args.category,
                pdf_vectorstore=pdf_vectorstore,
                pdf_rag_num_docs=args.pdf_rag_num_docs,
                save_dir=args.save_dir,
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
                smote_random_state=args.smote_random_state,
                lmda_path_size=args.lmda_path_size,
                optimize_metric=args.optimize_metric,
                compute_ci=args.compute_ci,
                bootstrap_method=args.bootstrap_method,
                bootstrap_n_rounds=args.bootstrap_n_rounds,
                ci_level=args.ci_level
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
                logger=logger,
                lmda_path_size=args.lmda_path_size
            )
        
        # Step 7b: Run Standard Lasso for comparison (optional)
        if args.skip_standard_lasso:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 7b: SKIPPING STANDARD LASSO (--skip_standard_lasso)")
            logger.info("=" * 60)
            standard_lasso_result = None
            standard_lasso_dir = None
            comparison_dir = None
            comparison_summary_file = None
        else:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 7b: RUNNING STANDARD LASSO FOR COMPARISON")
            logger.info("=" * 60)
            
            standard_lasso_dir = os.path.join(args.save_dir, "standard_lasso")
            os.makedirs(standard_lasso_dir, exist_ok=True)
            
            if args.use_loo:
                # Use Leave-One-Out cross-validation for standard lasso
                standard_lasso_result = run_standard_lasso_with_loo(
                    X=X_imputed,
                    y=y,
                    save_dir=standard_lasso_dir,
                    inner_cv_folds=args.inner_cv_folds,
                    n_threads=args.n_threads,
                    logger=logger,
                    participant_ids=X_imputed.index,
                    use_smote=args.use_smote,
                    smote_random_state=args.smote_random_state,
                    lmda_path_size=args.lmda_path_size,
                    optimize_metric=args.optimize_metric,
                    compute_ci=args.compute_ci,
                    bootstrap_method=args.bootstrap_method,
                    bootstrap_n_rounds=args.bootstrap_n_rounds,
                    ci_level=args.ci_level
                )
            else:
                # Use standard train/test split for standard lasso
                standard_lasso_result = run_standard_lasso_with_penalties(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    save_dir=standard_lasso_dir,
                    n_threads=args.n_threads,
                    folds_cv=args.folds_cv,
                    logger=logger,
                    lmda_path_size=args.lmda_path_size
                )
            
            # Step 8: Generate comparison plots
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 8: GENERATING COMPARISON PLOTS")
            logger.info("=" * 60)
            
            comparison_dir = os.path.join(args.save_dir, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            
            comparison_stats = generate_comparison_plots(
                llm_lasso_results=lasso_result,
                standard_lasso_results=standard_lasso_result,
                save_dir=comparison_dir,
                logger=logger,
                use_loo=args.use_loo
            )
            
            # Save comparison summary JSON
            comparison_summary = {
                "llm_lasso": {
                    "method": "llm_lasso",
                    "summary": lasso_result.get('summary', {})
                },
                "standard_lasso": {
                    "method": "standard_lasso",
                    "summary": standard_lasso_result.get('summary', {})
                },
                "comparison_stats": comparison_stats
            }
            
            comparison_summary_file = os.path.join(comparison_dir, "comparison_summary.json")
            with open(comparison_summary_file, 'w') as f:
                # Convert numpy types to JSON-serializable Python types
                json.dump(convert_to_json_serializable(comparison_summary), f, indent=2)
            logger.info(f"Comparison summary saved to: {comparison_summary_file}")
        
        # Pipeline complete
        pipeline_time = time.time() - pipeline_start
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total pipeline time: {pipeline_time:.2f}s ({pipeline_time/60:.1f} minutes)")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("")
        logger.info("Output directories:")
        logger.info(f"  - {args.save_dir}/ (LLM-Lasso results)")
        if standard_lasso_dir is not None:
            logger.info(f"  - {standard_lasso_dir}/ (Standard Lasso results)")
        if comparison_dir is not None:
            logger.info(f"  - {comparison_dir}/ (Comparison plots)")
        logger.info("")
        logger.info("Key output files:")
        logger.info(f"  - {os.path.join(args.save_dir, 'penalty_scores.json')}")
        if args.use_loo:
            logger.info(f"  - {os.path.join(args.save_dir, 'loo_predictions.csv')}")
            if standard_lasso_dir is not None:
                logger.info(f"  - {os.path.join(standard_lasso_dir, 'loo_predictions.csv')}")
        elif lasso_result.get('results') is not None:
            logger.info(f"  - {os.path.join(args.save_dir, 'lasso_results.csv')}")
            if standard_lasso_dir is not None:
                logger.info(f"  - {os.path.join(standard_lasso_dir, 'lasso_results.csv')}")
        logger.info(f"  - {os.path.join(args.save_dir, 'summary.json')}")
        if standard_lasso_dir is not None:
            logger.info(f"  - {os.path.join(standard_lasso_dir, 'summary.json')}")
        if comparison_summary_file is not None:
            logger.info(f"  - {comparison_summary_file}")
            logger.info(f"  - {os.path.join(comparison_dir, 'comparison_dashboard.png')}")
        logger.info(f"  - {log_file}")
        logger.info("")
        
        if lasso_result['summary'].get('lasso_skipped'):
            logger.info("Penalty scores generated successfully!")
            logger.info("Lasso training was skipped (adelie not installed)")
            logger.info("To run Lasso, install adelie: cd adelie-fork && pip install -e .")
        elif not args.skip_standard_lasso and standard_lasso_result is not None and args.use_loo:
            llm_summary = lasso_result['summary']
            std_summary = standard_lasso_result['summary']
            
            logger.info("=" * 60)
            logger.info("COMPARISON: LLM-LASSO vs STANDARD LASSO")
            logger.info("=" * 60)
            logger.info("")
            logger.info(f"{'Metric':<25} {'LLM-Lasso':<15} {'Standard':<15} {'Difference':<15}")
            logger.info("-" * 70)
            
            metrics_to_compare = [
                ('AUROC', 'auroc'),
                ('Accuracy', 'accuracy'),
                ('Balanced Accuracy', 'balanced_accuracy'),
                ('Sensitivity', 'sensitivity'),
                ('Specificity', 'specificity'),
                ('F1 Score', 'f1_score'),
                ('MCC', 'mcc'),
            ]
            
            for name, key in metrics_to_compare:
                llm_val = llm_summary.get(key, 0)
                std_val = std_summary.get(key, 0)
                diff = llm_val - std_val
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                winner = "LLM" if diff > 0.001 else ("STD" if diff < -0.001 else "TIE")
                logger.info(f"{name:<25} {llm_val:.4f}         {std_val:.4f}         {diff_str} ({winner})")
            
            logger.info("-" * 70)
            logger.info("")
            
            # Summarize which method performed better
            llm_wins = sum(1 for _, key in metrics_to_compare 
                          if llm_summary.get(key, 0) > std_summary.get(key, 0) + 0.001)
            std_wins = sum(1 for _, key in metrics_to_compare 
                          if std_summary.get(key, 0) > llm_summary.get(key, 0) + 0.001)
            
            if llm_wins > std_wins:
                logger.info(f"WINNER: LLM-Lasso ({llm_wins}/{len(metrics_to_compare)} metrics)")
            elif std_wins > llm_wins:
                logger.info(f"WINNER: Standard Lasso ({std_wins}/{len(metrics_to_compare)} metrics)")
            else:
                logger.info(f"RESULT: TIE (both methods perform similarly)")
            
            logger.info("")
            logger.info("Comparison plots saved to:")
            logger.info(f"  - {os.path.join(comparison_dir, 'metrics_comparison_bar.png')}")
            logger.info(f"  - {os.path.join(comparison_dir, 'roc_curves_comparison.png')}")
            logger.info(f"  - {os.path.join(comparison_dir, 'confusion_matrices_comparison.png')}")
            logger.info(f"  - {os.path.join(comparison_dir, 'comparison_dashboard.png')}")
        else:
            summ = lasso_result["summary"]
            test_err = summ.get("test_error")
            auroc_val = summ.get("auroc")
            n_feat = summ.get("n_features")
            total_feat = summ.get("total_features")
            logger.info(
                f"LLM-Lasso - Best model test error: {test_err:.4f}" if test_err is not None else "LLM-Lasso - Best model test error: N/A (lasso skipped)"
            )
            logger.info(
                f"LLM-Lasso - Best model AUROC: {auroc_val}" if auroc_val is not None else "LLM-Lasso - Best model AUROC: N/A (lasso skipped)"
            )
            logger.info(
                f"LLM-Lasso - Features selected: {n_feat}/{total_feat}" if n_feat is not None and total_feat is not None else "LLM-Lasso - Features selected: N/A (lasso skipped)"
            )
            if standard_lasso_result is not None:
                logger.info("")
                logger.info(f"Standard Lasso - Best model test error: {standard_lasso_result['summary'].get('test_error', 'N/A')}")
                logger.info(f"Standard Lasso - Best model AUROC: {standard_lasso_result['summary'].get('auroc', 'N/A')}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
