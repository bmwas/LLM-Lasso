# LLM-Lasso

LLM-Lasso is a novel framework that leverages large language models (LLMs) to guide feature selection in Lasso $\ell_1$ regression.

Unlike traditional feature selection methods that rely solely on numerical data, LLM-Lasso incorporates domain-specific knowledge extracted from natural language, enhanced through an optional retrieval-augmented generation (RAG) pipeline, to seamlessly integrate data-driven modeling with contextual insights. Specifically, the LLM generates penalty factors for each feature, which are converted into weights for the Lasso penalty using a simple, tunable model. Features identified as more relevant by the LLM receive lower penalties, increasing their likelihood of being retained in the final model, while less relevant features are assigned higher penalties, reducing their influence. Importantly, LLM-Lasso has an internal validation step that determines how much to trust the contextual knowledge in our prediction pipeline.

🔗 Paper link: [LLM-Lasso: A Robust Framework for Domain-Informed Feature Selection and Regularization](https://arxiv.org/abs/2502.10648)

![LLM-Lasso pipeline](documentation/rag-image.png)

📖 **[Detailed RAG Pipeline Documentation](documentation/RAG_PIPELINE.md)** - Understand how retrieval-augmented generation enhances feature selection.

---

## Table of Contents

- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [API Keys Configuration](#api-keys-configuration)
  - [Common Setup Issues](#common-setup-issues)
- [Quick Start](#quick-start)
- [Running LLM-Lasso Pipeline](#running-llm-lasso-pipeline)
  - [Preparing Your Data](#preparing-your-data)
  - [Creating a Prompt File](#creating-a-prompt-file)
  - [Running the Pipeline](#running-the-pipeline)
  - [Understanding Output Files](#understanding-output-files)
- [PDF RAG Pipeline](#pdf-rag-pipeline)
- [Tutorials](#tutorials)
- [Repo Structure](#repo-structure)

---

## Setup Instructions

### Prerequisites

#### System Dependencies

Before installing LLM-Lasso, ensure you have the required system libraries:

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install -y eigen3-devel llvm libomp-devel
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y libeigen3-dev llvm libomp-dev
```

**macOS (via Homebrew):**
```bash
brew install eigen llvm libomp
```

#### Python Requirements

- Python 3.9+ recommended
- pip or conda package manager

### Installation

Follow these steps in order:

#### Step 1: Install System Dependencies

**Complete the [Prerequisites](#prerequisites) section above first.** The adelie library requires C++ compilation with Eigen headers, so system dependencies must be installed before proceeding.

#### Step 2: Clone Repository and Create Virtual Environment

```bash
git clone https://github.com/your-repo/LLM-Lasso.git
cd LLM-Lasso

# Create and activate virtual environment
python -m venv virtualenv
source virtualenv/bin/activate  # Linux/macOS
# or: virtualenv\Scripts\activate  # Windows
```

#### Step 3: Install LLM-Lasso Package

```bash
pip install -e .
```

Or for conda:
```bash
conda develop .  # Requires: conda install conda-build
```

#### Step 4: Clone and Install Adelie (Required for Lasso Training)

The `adelie` library is required for solving Lasso with custom penalty factors. It's included as a git submodule that must be cloned and installed.

```bash
# Step 4a: Clone the adelie submodule
git submodule update --init --recursive

# Step 4b: Install adelie from the cloned fork
cd adelie-fork
pip install -e .
cd ..
```

> **Troubleshooting Submodule Clone:**
> 
> If Step 4a fails with SSH permission errors like:
> ```
> git@github.com: Permission denied (publickey)
> ```
> 
> Switch to HTTPS and retry:
> ```bash
> git config submodule.adelie-fork.url https://github.com/NSagan271/adelie-fork.git
> git submodule update --init --recursive
> ```

> **Troubleshooting Adelie Build:**
> 
> If Step 4b fails with `Eigen/Core: No such file or directory`, ensure you completed Step 1 (system dependencies). You may also need to set include paths:
> ```bash
> export C_INCLUDE_PATH="/usr/include/eigen3:$C_INCLUDE_PATH"
> export CPLUS_INCLUDE_PATH="/usr/include/eigen3:$CPLUS_INCLUDE_PATH"
> cd adelie-fork && pip install -e .
> ```

#### Step 5: Install Additional Python Dependencies

```bash
pip install python-dotenv langchain langchain-openai chromadb pymupdf4llm
```

### API Keys Configuration

LLM-Lasso requires an OpenAI API key for LLM queries and embeddings.

**Option A: Using a `.env` file (Recommended)**

Create a `.env` file in the project root:
```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Option B: Using `_my_constants.py`**

Copy the sample constants file and add your API key:
```bash
cp sample_constants.py _my_constants.py
```

Edit `_my_constants.py`:
```python
OPENAI_API = "sk-your-openai-api-key-here"
```

**Option C: Environment variable**
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
```

### Common Setup Issues

#### macOS-Specific Issues

If you installed LLVM via Homebrew on macOS, you may need to set these environment variables before installing adelie:

```bash
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
cd adelie-fork && pip install -e .
```

#### Alternative: Install Adelie from PyPI (Limited Functionality)

If building the forked adelie continues to fail, you can install the standard version from PyPI:

```bash
pip install adelie
```

**Note:** The PyPI version lacks custom diagnostic functions (`auc_roc`, `test_error_hamming`, `test_error_mse`) that are used for model evaluation. The penalty score generation will still work, but Lasso training with full diagnostics requires the forked version.

---

## Quick Start

After installation, run a quick test:

```bash
# Activate virtual environment
source virtualenv/bin/activate

# Test imports
python -c "from llm_lasso.llm_penalty.penalty_collection import collect_penalties; print('LLM-Lasso ready!')"

# Test adelie (if installed from fork)
python -c "from adelie.diagnostic import auc_roc; print('Adelie ready!')"
```

---

## Running LLM-Lasso Pipeline

### Preparing Your Data

LLM-Lasso requires three inputs:

1. **Dataset CSV file** - Your data with features and target variable
2. **Feature names file** - Text file with one feature name per line
3. **Prompt file** - Template for LLM queries

#### Dataset Format

Your CSV file should contain:
- Feature columns (predictor variables)
- Target column (binary 0/1 for classification, continuous for regression)

Example (`my_dataset.csv`):
```csv
Age,Gender,BloodPressure,Cholesterol,target_var
45,1,120,200,0
62,0,140,250,1
38,1,110,180,0
...
```

#### Feature Names File

Create a text file with one feature name per line (`features.txt`):
```
Age
Gender
BloodPressure
Cholesterol
```

> **Important:** Feature names must match column names in your CSV exactly.

### Creating a Prompt File

Create a prompt template that instructs the LLM how to evaluate features. Use `{genes}` as a placeholder for feature names.

Example (`prompts/my_prompt.txt`):
```
You are a medical expert analyzing risk factors for heart disease.

For each of the following variables, rate their relevance for predicting 
heart disease on a scale of 1-5, where:
- 1 = Highly relevant (strong predictor)
- 2 = Relevant
- 3 = Moderately relevant
- 4 = Slightly relevant
- 5 = Not relevant (weak/no predictive value)

Variables to evaluate:
{genes}

For each variable, provide a score and brief justification.
```

### Running the Pipeline

Use the provided pipeline script:

```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/your/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/my_prompt.txt \
    --target_column target_var \
    --category "Heart Disease" \
    --pdf_rag \
    --pdf_rag_num_docs 3 \
    --model-type gpt-4o \
    --temp 0 \
    --n-trials 1 \
    --test_size 0.3 \
    --imputation_strategy median \
    --log_level INFO
```

#### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_path` | Path to your CSV dataset | **Required** |
| `--feature_names_path` | Path to feature names file | **Required** |
| `--prompt-filename` | Path to prompt template | **Required** |
| `--target_column` | Name of target variable column | **Required** |
| `--category` | Description for LLM context | **Required** |
| `--save_dir` | Output directory | Same as dataset directory |
| `--pdf_rag` | Enable PDF RAG | `False` |
| `--pdf_rag_num_docs` | Documents to retrieve per feature | `3` |
| `--model-type` | LLM model (`gpt-4o`, `gpt-4`, etc.) | `gpt-4o` |
| `--temp` | LLM temperature (0 = deterministic) | `0` |
| `--n-trials` | Number of scoring trials | `1` |
| `--test_size` | Train/test split ratio | `0.2` |
| `--imputation_strategy` | Handle missing values (`median`, `mean`, `most_frequent`) | `median` |
| `--log_level` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `--wipe` | Clear previous results before running | `False` |

### Understanding Output Files

All outputs are saved to the same directory as your input dataset (or `--save_dir` if specified):

| File | Description |
|------|-------------|
| `penalty_scores.json` | LLM-generated penalty scores for each feature |
| `summary.json` | Pipeline execution summary and statistics |
| `pipeline.log` | Detailed execution log |
| `results_RAG.txt` | Raw LLM responses |
| `trial_scores_RAG.json` | Per-trial scoring data |
| `final_scores_RAG.pkl` | Serialized final scores |
| `lasso_results.csv` | Lasso model coefficients (if adelie installed) |

#### Example Output (`penalty_scores.json`):
```json
{
    "Age": 2.0,
    "Gender": 4.0,
    "BloodPressure": 1.0,
    "Cholesterol": 2.0
}
```

Lower scores (1-2) indicate more relevant features that receive lower Lasso penalties.

### Complete Example

Here's a full example running LLM-Lasso with PDF RAG on a clinical dataset:

```bash
# Activate virtual environment
source virtualenv/bin/activate

# Run the full pipeline
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/my_clinical_data.csv \
    --feature_names_path /path/to/clinical_features.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Suicidal Ideation" \
    --pdf_rag \
    --pdf_rag_num_docs 3 \
    --model-type gpt-4o \
    --temp 0 \
    --n-trials 1 \
    --test_size 0.3 \
    --imputation_strategy median \
    --log_level INFO \
    --wipe
```

**Expected Output:**
```
================================================================================
PBD LLM-LASSO PIPELINE WITH PDF RAG
================================================================================
STEP 1: LOAD FEATURE NAMES
  Loaded 31 feature names
STEP 2: LOAD AND VALIDATE DATASET
  Dataset shape: 137 rows x 32 columns
  Target distribution: {1: 100, 0: 37}
STEP 3: MISSING VALUE IMPUTATION
  Total missing values: 89
  Imputation strategy: median
STEP 4: TRAIN/TEST SPLIT
  Training set: 95 samples
  Test set: 42 samples
STEP 5: PDF VECTORSTORE SETUP
  Loaded vectorstore with 287 documents
STEP 6: GENERATE LLM PENALTIES
  Penalty collection completed in 64.17s
  Collected 31 penalty scores
STEP 7: LASSO CLASSIFICATION
  Lasso training completed in 8.62s
  
RESULTS:
  Method: Lasso
  Test error: 0.1667
  AUROC: 0.868
  Number of features selected: 15
  
PIPELINE COMPLETE
  Total time: 88.82s
  Results saved to: /path/to/my_clinical_data/
```

**Output Files Generated:**
- `penalty_scores.json` - LLM penalty scores for each feature
- `lasso_results.csv` - Full model results with coefficients
- `summary.json` - Performance metrics (AUROC, test error, etc.)
- `pipeline.log` - Detailed execution log

### Leave-One-Out Cross-Validation (LOOCV)

For more rigorous evaluation, LLM-Lasso supports nested cross-validation with:
- **Outer loop**: Leave-One-Out (each sample tested exactly once)
- **Inner loop**: k-fold CV for hyperparameter (lambda) selection

**Key advantages:**
- LLM RAG runs only ONCE (not repeated for each fold)
- Each sample gets a probability prediction
- Unbiased performance estimates
- SMOTE support for handling class imbalance

**Command:**

```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Your Category" \
    --pdf_rag \
    --use_loo \
    --inner_cv_folds 10 \
    --model-type gpt-4o
```

**LOOCV-Specific Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_loo` | Enable Leave-One-Out cross-validation | `False` |
| `--inner_cv_folds` | Inner CV folds for hyperparameter tuning | `10` |

### Handling Class Imbalance with SMOTE

For imbalanced datasets (e.g., 100 positive vs 37 negative samples), LLM-Lasso supports SMOTE (Synthetic Minority Over-sampling Technique) to balance classes during training.

**How it works:**
- SMOTE is applied **only to training data** within each LOO fold
- Test data is **never** resampled, ensuring unbiased evaluation
- Synthetic samples are generated for the minority class
- The imbalanced-learn library is used for SMOTE implementation

**Command with SMOTE:**

```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Your Category" \
    --pdf_rag \
    --use_loo \
    --inner_cv_folds 10 \
    --use_smote \
    --model-type gpt-4o
```

**SMOTE Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_smote` | Enable SMOTE for class balancing | `False` |
| `--smote_random_state` | Random state for SMOTE reproducibility | `42` |

**When to use SMOTE:**
- Low specificity with high sensitivity (model biased toward majority class)
- Significant class imbalance (ratio > 2:1)
- Balanced accuracy much lower than regular accuracy

**Output Files (LOOCV mode):**

| File | Description |
|------|-------------|
| `loo_predictions.csv` | Per-sample predictions with participant_id, actual_label, predicted_probability |
| `roc_curve.png` | Publication-quality ROC curve with AUC |
| `precision_recall_curve.png` | Precision-Recall curve with Average Precision |
| `confusion_matrix.png` | Confusion matrix (counts and normalized) |
| `probability_distribution.png` | Probability histograms by true class |
| `calibration_curve.png` | Model calibration (reliability diagram) |
| `metrics_summary.png` | Bar chart of all performance metrics |
| `performance_dashboard.png` | Combined dashboard with all visualizations |
| `detailed_metrics.json` | Comprehensive metrics (AUROC, accuracy, F1, MCC, etc.) |
| `classification_report.txt` | Sklearn classification report |

**Example LOOCV Output:**

```
LOO CROSS-VALIDATION FINAL RESULTS
============================================================
AUROC: 0.7892
Accuracy: 0.7591 (104/137 correct)
Balanced Accuracy: 0.7204
Sensitivity: 0.8200
Specificity: 0.6216
F1 Score: 0.8317
MCC: 0.4261

Generated 7 publication-quality figures
```

**Sample `loo_predictions.csv`:**

```csv
participant_id,actual_label,predicted_probability
0,1,0.647
1,1,0.644
2,0,0.608
3,1,0.954
...
```

---

## PDF RAG Pipeline

LLM-Lasso supports retrieval-augmented generation using local PDF documents (e.g., scientific papers) to provide domain-specific context.

📖 **[Full RAG Documentation](documentation/RAG_PIPELINE.md)** - Detailed explanation with diagrams.

### Quick Setup

1. **Prepare PDF Documents:**
   
   Place your PDF files in a directory:
   ```bash
   mkdir sample_pdfs
   cp /path/to/your/papers/*.pdf sample_pdfs/
   ```

2. **Create the Vector Store:**
   ```bash
   python playground/interactive_pdf_RAG.py
   ```
   
   Follow the prompts to index your PDFs into ChromaDB.

3. **Verify Indexed Documents:**
   ```bash
   # View all indexed documents
   python playground/view_all_documents.py
   
   # With longer content preview
   python playground/view_all_documents.py --max-length 1000
   
   # Save to file
   python playground/view_all_documents.py --save indexed_documents.txt
   ```

4. **Run Pipeline with PDF RAG:**
   ```bash
   python scripts/run_pbd_llm_lasso.py \
       --dataset_path /path/to/dataset.csv \
       --feature_names_path /path/to/features.txt \
       --prompt-filename prompts/my_prompt.txt \
       --target_column target_var \
       --category "My Research Domain" \
       --pdf_rag \
       --pdf_rag_num_docs 3
   ```

### Programmatic Usage

```python
from llm_lasso.llm_penalty.rag import create_pdf_vectorstore, load_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings

# Create vectorstore from PDFs
embeddings = OpenAIEmbeddings()
vectorstore = create_pdf_vectorstore(
    pdf_directory="sample_pdfs",
    persist_directory="pdf_vectorstore",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model=embeddings
)

# Load existing vectorstore
vectorstore = load_pdf_vectorstore(
    persist_directory="pdf_vectorstore",
    embedding_model=embeddings
)

# Check statistics
collection = vectorstore._collection
print(f"Total document chunks: {collection.count()}")
```

---

## Tutorials

For detailed walkthroughs:

- **`examples/llm_lasso_tutorial.ipynb`** - Full LLM-Lasso pipeline tutorial with classification and regression examples

- **`examples/omim_rag_tutorial.ipynb`** - OMIM database scraping and RAG tutorial

---

## Repo Structure

```
LLM-Lasso/
├── adelie-fork/           # Adelie submodule for Lasso with penalty factors
├── documentation/         # Additional documentation
│   ├── RAG_PIPELINE.md    # Detailed RAG explanation with diagrams
│   └── rag-image.png      # Pipeline visualization
├── examples/              # Jupyter notebook tutorials
├── omim_scrape/           # OMIM RAG helper functions
├── playground/            # Interactive scripts
│   ├── interactive_pdf_RAG.py    # PDF vectorstore creation/querying
│   └── view_all_documents.py     # Vectorstore inspection utility
├── prompts/               # LLM prompt templates
├── sample_pdfs/           # Directory for PDF documents
├── scripts/               # Pipeline execution scripts
│   ├── run_pbd_llm_lasso.py      # Main pipeline script
│   ├── llm_lasso_scores.py       # Penalty score generation
│   ├── run_baselines.py          # Baseline methods
│   └── small_scale_splits.py     # Data split generation
├── src/llm_lasso/         # Core package
│   ├── llm_penalty/       # LLM penalty generation
│   │   ├── rag/           # RAG modules (OMIM, PubMed, PDF)
│   │   └── ...
│   ├── task_specific_lasso/  # Lasso with LLM penalties
│   └── utils/             # Helper functions
├── constants.py           # Configuration loader
├── sample_constants.py    # Template for API keys
├── _my_constants.py       # Your API keys (create this)
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

---

## Citation

If you use LLM-Lasso in your research, please cite:

```bibtex
@article{llm-lasso-2025,
    title={LLM-Lasso: A Robust Framework for Domain-Informed Feature Selection and Regularization},
    author={...},
    journal={arXiv preprint arXiv:2502.10648},
    year={2025}
}
```

---

## License

[Add your license information here]
