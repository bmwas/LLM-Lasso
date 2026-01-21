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
    from llm_lasso.task_specific_lasso.llm_lasso import llm_lasso_cv, PenaltyType
    ADELIE_AVAILABLE = True
except ImportError:
    pass


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
        "help": "Number of cross-validation folds"
    })
    n_threads: int = field(default=4, metadata={
        "help": "Number of threads for Lasso training"
    })
    wipe: bool = field(default=False, metadata={
        "help": "Wipe existing results before starting"
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
        
        # Step 4: Split data
        logger.info("")
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
        
        # Step 6: Generate LLM penalties
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
        if lasso_result['results'] is not None:
            logger.info(f"  - {os.path.join(args.save_dir, 'lasso_results.csv')}")
        logger.info(f"  - {os.path.join(args.save_dir, 'summary.json')}")
        logger.info(f"  - {log_file}")
        logger.info("")
        
        if lasso_result['summary'].get('lasso_skipped'):
            logger.info("Penalty scores generated successfully!")
            logger.info("Lasso training was skipped (adelie not installed)")
            logger.info("To run Lasso, install adelie: cd adelie-fork && pip install -e .")
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
