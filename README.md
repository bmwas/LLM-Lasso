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
- [Open-Source LLM Deployment (Docker)](#open-source-llm-deployment-docker)
  - [Docker Prerequisites](#docker-prerequisites)
  - [Environment Setup](#environment-setup)
  - [Building and Launching](#building-and-launching)
  - [Verifying the Services](#verifying-the-services)
  - [Viewing Logs](#viewing-logs)
  - [Using with LLM-Lasso](#using-with-llm-lasso)
  - [Stopping the Services](#stopping-the-services)
  - [Troubleshooting Docker](#troubleshooting-docker)
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

## Open-Source LLM Deployment (Docker)

LLM-Lasso can use open-source LLMs instead of OpenAI's API by running [vLLM](https://docs.vllm.ai/) locally. vLLM provides OpenAI-compatible API endpoints, allowing seamless switching between cloud and local models.

The Docker setup includes two services:
- **vllm_chat**: Qwen3-30B thinking model for penalty score generation (port 8000)
- **vllm_embed**: Qwen3-Embedding-8B for RAG embeddings (port 8001)

### Docker Prerequisites

Before deploying the Docker services, ensure you have:

1. **NVIDIA GPU** with CUDA 12.1+ and sufficient VRAM
   - Chat model (Qwen3-30B-FP8): ~24GB VRAM recommended
   - Embedding model (Qwen3-8B): ~8GB VRAM recommended
   - Both services share GPU 0 by default

2. **NVIDIA Container Toolkit** installed:

   **Ubuntu/Debian:**
   ```bash
   # Add NVIDIA repository
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   
   # Configure Docker runtime
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

   **Fedora/RHEL/CentOS:**
   ```bash
   # Add NVIDIA repository
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
     sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
   sudo dnf install -y nvidia-container-toolkit
   
   # Configure Docker runtime
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Docker Engine 24.0+** and **Docker Compose v2.20+**:
   ```bash
   docker --version    # Should be 24.0+
   docker compose version  # Should be v2.20+
   ```

4. **HuggingFace Token** (required for gated Qwen3 models):
   - Create an account at https://huggingface.co/
   - Generate a token at https://huggingface.co/settings/tokens
   - Accept the model license agreements:
     - https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
     - https://huggingface.co/Qwen/Qwen3-Embedding-8B

### Environment Setup

1. **Copy the example environment file** to the project root:
   ```bash
   cp opensource_llms/.env.example .env
   ```

2. **Edit `.env`** and add your credentials:
   ```bash
   # Required - your HuggingFace token
   HUGGINGFACE_TOKEN=hf_your_actual_token_here
   
   # Optional but recommended - API key for vLLM endpoints
   # Generate with: openssl rand -hex 32
   VLLM_API_KEY=your_secure_api_key_here
   ```

> **Important:** The `HUGGINGFACE_TOKEN` is **required** for downloading the Qwen3 models. Without it, the containers will fail to start.

### Building and Launching

Navigate to the project root and start the services:

```bash
# Start both services (run from project root)
docker compose --env-file .env -f opensource_llms/docker-compose.yml up

# Or start with build (if you've made changes)
docker compose --env-file .env -f opensource_llms/docker-compose.yml up --build
```

The first launch will download the models from HuggingFace (~30GB+ total), which may take considerable time depending on your internet connection. Model files are cached in a Docker volume for subsequent starts.

### Verifying the Services

1. **Check container status:**
   ```bash
   docker compose --env-file .env -f opensource_llms/docker-compose.yml ps
   ```
   
   Expected output (after models load):
   ```
   NAME         IMAGE                       STATUS                   PORTS
   vllm_chat    vllm/vllm-openai:v0.12.0   Up 5 minutes (healthy)   0.0.0.0:8000->8000/tcp
   vllm_embed   vllm/vllm-openai:v0.12.0   Up 5 minutes (healthy)   0.0.0.0:8001->8000/tcp
   ```

2. **Test the chat endpoint:**
   ```bash
   curl http://localhost:8000/v1/models
   ```
   
   Expected response:
   ```json
   {"object":"list","data":[{"id":"qwen3-thinking","object":"model",...}]}
   ```

3. **Test the embeddings endpoint:**
   ```bash
   curl http://localhost:8001/v1/models
   ```
   
   Expected response:
   ```json
   {"object":"list","data":[{"id":"qwen3-embed","object":"model",...}]}
   ```

4. **Test a chat completion:**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${VLLM_API_KEY}" \
     -d '{
       "model": "qwen3-thinking",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'
   ```

5. **Test embeddings:**
   ```bash
   curl http://localhost:8001/v1/embeddings \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${VLLM_API_KEY}" \
     -d '{
       "model": "qwen3-embed",
       "input": "Test embedding"
     }'
   ```

### Viewing Logs

The Docker Compose configuration includes extensive debug logging for troubleshooting:

```bash
# View all logs (follow mode)
docker compose --env-file .env -f opensource_llms/docker-compose.yml logs -f

# View only chat service logs
docker compose --env-file .env -f opensource_llms/docker-compose.yml logs -f vllm_chat

# View only embedding service logs
docker compose --env-file .env -f opensource_llms/docker-compose.yml logs -f vllm_embed

# View last 100 lines of logs
docker compose --env-file .env -f opensource_llms/docker-compose.yml logs --tail=100

# Save logs to file for analysis
docker compose --env-file .env -f opensource_llms/docker-compose.yml logs > vllm_debug.log 2>&1
```

**Debug logging includes:**
- Model loading progress and memory allocation
- Request/response timing and queue states
- Cache hit rates and batch statistics
- GPU memory utilization

### Using with LLM-Lasso

To use the local vLLM services with LLM-Lasso, configure your environment to point to the local endpoints:

```python
# In your Python code or .env file
import os

# For chat/completion models
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "your_vllm_api_key"  # Same as VLLM_API_KEY

# For embedding models (if using separate endpoint)
os.environ["OPENAI_EMBEDDING_API_BASE"] = "http://localhost:8001/v1"
```

Or pass directly to the LLM-Lasso pipeline:
```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/my_prompt.txt \
    --target_column target_var \
    --category "My Category" \
    --model-type qwen3-thinking \
    --pdf_rag
```

**OpenAI-Compatible Endpoints:**

| Service | Endpoint | Description |
|---------|----------|-------------|
| Chat | `http://localhost:8000/v1/chat/completions` | Text generation |
| Chat Models | `http://localhost:8000/v1/models` | List available models |
| Embeddings | `http://localhost:8001/v1/embeddings` | Vector embeddings |
| Embed Models | `http://localhost:8001/v1/models` | List available models |
| Health | `http://localhost:8000/health` | Service health check |

### Stopping the Services

```bash
# Stop services (keeps volumes)
docker compose --env-file .env -f opensource_llms/docker-compose.yml down

# Stop services and remove volumes (deletes cached models!)
docker compose --env-file .env -f opensource_llms/docker-compose.yml down -v

# Stop a specific service
docker compose --env-file .env -f opensource_llms/docker-compose.yml stop vllm_chat
```

### Troubleshooting Docker

**Container fails to start with "HF_TOKEN is required":**
- Ensure you've created `.env` in the project root with a valid HuggingFace token
- Verify the token has read access and you've accepted the model licenses

**Container starts but health check fails:**
- Large models take 3-5 minutes to load; check logs for progress
- Ensure sufficient GPU memory (use `nvidia-smi` to monitor)
- Reduce `--max-model-len` or `--gpu-memory-utilization` if OOM errors occur

**CUDA out of memory errors:**
- Edit `opensource_llms/docker-compose.yml` to reduce memory usage:
  ```yaml
  "--gpu-memory-utilization", "0.70"  # Reduce from 0.78
  "--max-model-len", "16384"          # Reduce from 32768
  ```

**Models download slowly or fail:**
- HuggingFace may have rate limits; wait and retry
- Check your internet connection and firewall settings
- Verify HUGGINGFACE_TOKEN is correct and has appropriate permissions

**Permission denied errors:**
- Ensure your user is in the `docker` group: `sudo usermod -aG docker $USER`
- Log out and back in for group changes to take effect

**GPU not detected:**
- Verify NVIDIA Container Toolkit is installed: `nvidia-ctk --version`
- Check Docker can access GPU: `docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi`
- Restart Docker after installing toolkit: `sudo systemctl restart docker`

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
| `--pdf_rag_num_docs` | Number of PDF documents to retrieve per feature query. Each feature gets up to this many documents. With deduplication, total unique documents will be fewer. Higher values provide more context but increase processing time. | `3` |
| `--model-type` | LLM model (`gpt-4o`, `gpt-4`, etc.) | `gpt-4o` |
| `--temp` | LLM temperature (0-2). Use `0` for deterministic, reproducible results. Higher values (0.7-1.0) add creativity but reduce reproducibility. **Recommended: `0` for research reproducibility.** | `0` |
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
| `model_coefficients.json` | Lasso coefficients (final model + LOO-aggregated with mean/std/selection frequency) |
| `coefficients_nonzero.png` | Bar plot of non-zero coefficients sorted by signed value (positive=green, negative=red) |
| `coefficients_stability.png` | Coefficient stability plot showing mean±std across LOO folds + selection frequency |
| `coefficients_top_features.png` | Top 20 features by coefficient magnitude, sorted by signed value |
| `coefficients_tornado.png` | Tornado plot showing ALL coefficients sorted by signed value |
| `roc_curve.png` | Publication-quality ROC curve with AUC |
| `precision_recall_curve.png` | Precision-Recall curve with Average Precision |
| `confusion_matrix.png` | Confusion matrix (counts and normalized) |
| `probability_distribution.png` | Probability histograms by true class |
| `calibration_curve.png` | Model calibration (reliability diagram) |
| `metrics_summary.png` | Bar chart of all performance metrics |
| `performance_dashboard.png` | Combined dashboard with all visualizations |
| `detailed_metrics.json` | Comprehensive metrics (AUROC, accuracy, F1, MCC, etc.) |
| `classification_report.txt` | Sklearn classification report |

**Coefficient Outputs:**

The `model_coefficients.json` file contains:
- **`final_model`**: Coefficients from a model trained on all data
- **`loo_aggregated`**: Mean, std, and selection frequency of coefficients across all LOO folds

Selection frequency indicates how often each feature was selected (non-zero) across LOO folds - features with >50% are considered stable predictors.

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

Features with non-zero mean coefficient: 12/31
Features selected in >50% of folds: 8/31
Training final model on all data for stable coefficients...
  Final model: 10 non-zero coefficients
Coefficients saved to: model_coefficients.json
Generated 4 coefficient plots
Generated 7 evaluation figures
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

### Document Retrieval Details

**Number of Documents Returned:**

- **Per feature query**: Controlled by `--pdf_rag_num_docs` (default: `3`)
  - Each feature gets up to `pdf_rag_num_docs` documents from the vectorstore
  - Example: With 31 features and `--pdf_rag_num_docs 3`, up to 93 document retrievals occur
  - After deduplication (same document retrieved for multiple features), the total unique documents is typically lower
  - Example: 31 features × 3 docs = 93 retrievals → ~29 unique documents (after deduplication)

- **Total unique documents**: Saved in `rag_retrieved_documents.json` under `summary.total_unique_documents`

**Choosing `--pdf_rag_num_docs`:**
- **Lower values (1-3)**: Faster, focused context, good for well-indexed literature
- **Higher values (5-10)**: More comprehensive context, better coverage, slower processing
- **Recommendation**: Start with `3`, increase if features lack sufficient context

### Reproducibility and Determinism

**Ensuring Reproducible Results:**

The RAG retrieval process is **mostly deterministic** but requires specific settings for full reproducibility:

1. **Use `--temp 0`** (default): 
   - Sets LLM temperature to 0 for deterministic responses
   - Without this, LLM responses may vary between runs
   - **Always use `--temp 0` for research reproducibility**

2. **Keep vectorstore unchanged**:
   - Don't rebuild or modify the PDF vectorstore between runs
   - Same documents → same embeddings → same retrieval results

3. **Fixed parameters**:
   - Use the same `--pdf_rag_num_docs` value
   - Use the same `--random_state` for data splitting (if applicable)

**What's Deterministic:**
- ✅ Document embeddings (OpenAI embeddings are deterministic)
- ✅ Vectorstore similarity search (ChromaDB retrieval is deterministic)
- ✅ Document deduplication (based on content hash)

**What May Vary:**
- ⚠️ LLM responses if `--temp > 0` (use `--temp 0` to fix)
- ⚠️ Results if vectorstore is modified between runs
- ⚠️ Order of documents with identical similarity scores (rare)

**Example for Reproducible Research:**
```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/my_prompt.txt \
    --target_column target_var \
    --category "My Research Domain" \
    --pdf_rag \
    --pdf_rag_num_docs 3 \
    --temp 0 \
    --random_state 42
```

### Reference Section Filtering

> **Why filter references?** Bibliography and reference lists in scientific papers contain only citation metadata (author names, journal titles, years, DOIs) with **little to no scientific context**. When indexed, these reference entries introduce noise into RAG retrieval—the LLM may retrieve a reference title instead of actual scientific content about your features, degrading the quality of penalty score generation.

LLM-Lasso **automatically filters out reference/bibliography sections** during PDF indexing to ensure only substantive scientific content is retrieved.

#### The Problem: Reference Lists as Noise

Consider a typical reference entry:
```
[23] Smith J, Jones K, Brown M. Neural correlates of bipolar disorder 
     in pediatric populations. J Psychiatry. 2020;45(3):123-145. 
     doi:10.1234/jp.2020.045
```

This entry contains:
- ❌ Author names (not useful for feature context)
- ❌ Journal metadata (volume, issue, pages)
- ❌ DOI identifiers
- ❌ Only a brief title with minimal context

**Without filtering**, this reference might be retrieved when querying for "bipolar disorder" features, providing the LLM with citation metadata instead of actual scientific findings.

#### Two-Stage Filtering Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    PDF DOCUMENT                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: PAGE-LEVEL FILTERING                                  │
│  ─────────────────────────────                                  │
│  • Detects "References", "Bibliography", "Works Cited" headers  │
│  • Analyzes page content for citation patterns                  │
│  • Position-aware: pages near document end get extra scrutiny   │
│  • Entire reference pages are SKIPPED                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: CHUNK-LEVEL FILTERING                                 │
│  ──────────────────────────────                                 │
│  • Catches reference content spanning page boundaries           │
│  • Filters chunks with high DOI/citation density                │
│  • Uses stricter thresholds than page-level                     │
│  • Individual reference chunks are REMOVED                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLEAN VECTORSTORE (substantive content only)                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Detection Patterns

| Pattern Type | What It Detects | Examples |
|--------------|-----------------|----------|
| **Section Headers** | Reference section starts | `References`, `Bibliography`, `Works Cited`, `Literature Cited` |
| **Numbered Entries** | Citation list items | `[1] Author...`, `1. Author...`, `(1) Author...` |
| **DOI Patterns** | Digital Object Identifiers | `doi:10.1234/...`, `https://doi.org/10.1234/...` |
| **Journal Formats** | Citation metadata | `2020;45(3):123-145`, `Vol. 12, pp. 100-110` |
| **Author Patterns** | Name formatting | `Smith J, Jones K.`, `et al.` |

#### Filtering Statistics

When creating a vectorstore, you'll see filtering statistics:

```
Reference filtering: ENABLED
Found 15 PDF file(s) in /path/to/pdfs
Page-level reference filtering: 12/127 pages filtered (9.4%)
Chunk-level reference filtering: 23/450 chunks filtered (5.1%)
Created 427 chunks from 115 document segments
```

#### Configuration

Reference filtering is **enabled by default**. In most cases, you don't need to change anything.

To disable filtering (not recommended):

```python
from llm_lasso.llm_penalty.rag import create_pdf_vectorstore

vectorstore = create_pdf_vectorstore(
    pdf_directory="sample_pdfs",
    persist_directory="pdf_vectorstore",
    filter_references=False  # ⚠️ Not recommended
)
```

> **Warning:** Disabling reference filtering may result in:
> - Retrieval of citation metadata instead of scientific content
> - Reduced quality of RAG context for LLM penalty generation
> - Noisy feature penalties based on reference titles rather than findings

### Programmatic Usage

```python
from llm_lasso.llm_penalty.rag import create_pdf_vectorstore, load_pdf_vectorstore
from langchain_openai import OpenAIEmbeddings

# Create vectorstore from PDFs (with reference filtering enabled by default)
embeddings = OpenAIEmbeddings()
vectorstore = create_pdf_vectorstore(
    pdf_directory="sample_pdfs",
    persist_directory="pdf_vectorstore",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model=embeddings,
    filter_references=True  # Enabled by default - filters out bibliography sections
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

## RAG Document Embedding Visualization

After running the LLM-Lasso pipeline with PDF RAG, you can analyze and visualize the retrieved documents using the embedding visualization tool. This helps understand which scientific literature is being used to inform the LLM penalty scores and how documents relate to different feature queries.

### Pipeline Overview

```
rag_retrieved_documents.json
         │
         ▼
┌─────────────────────────────────────┐
│  1. EMBEDDING (OpenAI text-embedding-ada-002)
│     - Same embedding model as RAG pipeline
│     - 1536-dimensional vectors
│     - Cosine similarity metric
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  2. DIMENSIONALITY REDUCTION
│     ├── UMAP: Preserves global structure
│     │   - n_neighbors=15, min_dist=0.1
│     │   - Faster, good for large datasets
│     └── t-SNE: Preserves local structure
│         - perplexity=auto, n_iter=1000
│         - Better for small clusters
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  3. CLUSTERING
│     ├── K-Means: Fixed number of clusters
│     │   - Automatic k selection (silhouette)
│     │   - Range: k=2 to k=15
│     └── HDBSCAN: Density-based clustering
│         - Automatic cluster detection
│         - Handles noise points
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  4. VISUALIZATION
│     - Publication-quality scatter plots
│     - Feature query labels per document
│     - Cluster statistics and legends
└─────────────────────────────────────┘
```

### Technical Details

**Embedding Method:**
- Uses `langchain_openai.OpenAIEmbeddings` (same as RAG retrieval)
- Model: `text-embedding-ada-002` (1536 dimensions)
- Documents are embedded using their full content

**Dimensionality Reduction:**

| Method | Strengths | Parameters |
|--------|-----------|------------|
| **UMAP** | Preserves global structure, faster | n_neighbors=15, min_dist=0.1, metric=cosine |
| **t-SNE** | Better local cluster separation | perplexity=auto, learning_rate=auto, n_iter=1000 |

**Clustering Algorithms:**

| Method | Strengths | Selection Criteria |
|--------|-----------|-------------------|
| **K-Means** | Well-defined clusters, reproducible | Optimal k via silhouette score (k=2 to 15) |
| **HDBSCAN** | No k required, handles noise/outliers | min_cluster_size=auto, min_samples=2 |

### Usage

```bash
# After running the main pipeline (which generates rag_retrieved_documents.json)
python scripts/visualize_rag_embeddings.py \
    --input /path/to/rag_retrieved_documents.json \
    --output /path/to/output_directory
```

**Example:**

```bash
python scripts/visualize_rag_embeddings.py \
    --input /home/benson/Downloads/pbd_dataset/rag_retrieved_documents.json \
    --output /home/benson/Downloads/pbd_dataset/rag_visualization
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Path to `rag_retrieved_documents.json` | Required |
| `--output`, `-o` | Output directory for plots and data | Required |
| `--log_level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `--random_state` | Random seed for reproducibility | `42` |

### Output Files

| File | Description |
|------|-------------|
| `rag_embeddings_umap_kmeans.png` | UMAP projection with K-Means clusters |
| `rag_embeddings_umap_hdbscan.png` | UMAP projection with HDBSCAN clusters |
| `rag_embeddings_tsne_kmeans.png` | t-SNE projection with K-Means clusters |
| `rag_embeddings_tsne_hdbscan.png` | t-SNE projection with HDBSCAN clusters |
| `rag_embeddings_dashboard.png` | Combined dashboard with all 4 visualizations |
| `rag_embeddings_data.csv` | Document embeddings with 2D coordinates and cluster labels |
| `rag_embeddings_summary.json` | Clustering statistics, silhouette scores, and metadata |

### Example Output

```
============================================================
RAG DOCUMENT EMBEDDING VISUALIZER
============================================================
STEP 1: Loading RAG documents
  Category: Suicidal Ideation
  Features queried: 31
  Unique documents: 29
  
STEP 2: Computing embeddings
  Embedding shape: (29, 1536)
  
STEP 3: Dimensionality reduction
  UMAP output: (29, 2)
  t-SNE output: (29, 2)
  
STEP 4: Clustering
  UMAP + K-Means: 2 clusters (silhouette: 0.505)
  UMAP + HDBSCAN: 5 clusters + 3 noise points
  t-SNE + K-Means: 5 clusters (silhouette: 0.500)
  t-SNE + HDBSCAN: 5 clusters + 2 noise points

Generated 7 output files
```

### Interpretation Guide

| Element | Meaning |
|---------|---------|
| **Cluster colors** | Documents with similar content are grouped together |
| **Feature labels** | Which feature queries (e.g., "Ed_years", "CDRS.CDRSTscore") retrieved each document |
| **Multiple labels** | Document is relevant to multiple features - indicates shared scientific context |
| **Noise points (gray, HDBSCAN)** | Documents that don't fit well into any cluster - may be unique or outliers |
| **Silhouette score** | Cluster quality metric (higher is better, >0.5 is good) |

### Use Cases

1. **Understand RAG Context**: See what scientific literature informs each feature's penalty score
2. **Identify Knowledge Gaps**: Features with few/no documents may need more literature
3. **Detect Redundancy**: Highly overlapping clusters may indicate redundant retrieval
4. **Validate Relevance**: Check if retrieved documents semantically group by research topic

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
├── opensource_llms/       # Docker setup for local open-source LLMs
│   ├── docker-compose.yml # vLLM services configuration
│   └── .env.example       # Template for required environment variables
├── playground/            # Interactive scripts
│   ├── interactive_pdf_RAG.py    # PDF vectorstore creation/querying
│   └── view_all_documents.py     # Vectorstore inspection utility
├── prompts/               # LLM prompt templates
├── sample_pdfs/           # Directory for PDF documents
├── scripts/               # Pipeline execution scripts
│   ├── run_pbd_llm_lasso.py      # Main pipeline script
│   ├── visualize_rag_embeddings.py  # RAG document embedding visualization
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
