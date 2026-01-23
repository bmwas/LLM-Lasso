# LLM-Lasso

LLM-Lasso is a novel framework that leverages large language models (LLMs) to guide feature selection in Lasso $\ell_1$ regression.

Unlike traditional feature selection methods that rely solely on numerical data, LLM-Lasso incorporates domain-specific knowledge extracted from natural language, enhanced through an optional retrieval-augmented generation (RAG) pipeline, to seamlessly integrate data-driven modeling with contextual insights. Specifically, the LLM generates penalty factors for each feature, which are converted into weights for the Lasso penalty using a simple, tunable model. Features identified as more relevant by the LLM receive lower penalties, increasing their likelihood of being retained in the final model, while less relevant features are assigned higher penalties, reducing their influence. Importantly, LLM-Lasso has an internal validation step that determines how much to trust the contextual knowledge in our prediction pipeline.

🔗 Paper link: [LLM-Lasso: A Robust Framework for Domain-Informed Feature Selection and Regularization](https://arxiv.org/abs/2502.10648)

![LLM-Lasso pipeline](documentation/rag-image.png)

📖 **[Detailed RAG Pipeline Documentation](documentation/RAG_PIPELINE.md)** - Understand how retrieval-augmented generation enhances feature selection.

## Key Features

✨ **Open-Source LLM Support**: Run LLM-Lasso locally with Qwen3 models via vLLM - no cloud API required  
🔍 **Comprehensive Grid Search**: Automatic hyperparameter search with visualization (100+ λ values tested)  
📊 **Grid Search Visualization**: Publication-ready plots showing regularization path analysis  
🎯 **Flexible Metric Optimization**: Choose which metric to optimize (accuracy, sensitivity, specificity, balanced accuracy, F1, AUC-ROC)  
📈 **Bootstrap Confidence Intervals**: Automatic computation of CIs for all performance metrics  
🔐 **Privacy & Cost Control**: Process sensitive data locally without API costs  

See [Open-Source LLM Integration](#open-source-llm-integration-with-qwen-models), [Hyperparameter Search](#hyperparameter-search-and-grid-search-visualization), [Optimization Metric Selection](#optimization-metric-selection), and [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals) for details.

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
  - [Open-Source LLM Integration with Qwen Models](#open-source-llm-integration-with-qwen-models)
  - [Stopping the Services](#stopping-the-services)
  - [Troubleshooting Docker](#troubleshooting-docker)
- [Quick Start](#quick-start)
- [Running LLM-Lasso Pipeline](#running-llm-lasso-pipeline)
  - [Preparing Your Data](#preparing-your-data)
  - [Creating a Prompt File](#creating-a-prompt-file)
  - [Running the Pipeline](#running-the-pipeline)
  - [Understanding Output Files](#understanding-output-files)
  - [Hyperparameter Search and Grid Search Visualization](#hyperparameter-search-and-grid-search-visualization)
  - [Optimization Metric Selection](#optimization-metric-selection)
  - [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
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

6. **Run the comprehensive test script:**
   
   A Python test script is provided to verify both endpoints with detailed output. The script automatically loads environment variables from `.env` in the project root:
   ```bash
   # From project root (no need to source .env - script loads it automatically)
   python opensource_llms/test_vllm_endpoints.py
   ```
   
   The script performs comprehensive testing:
   - **Health checks** for both services (verifies services are responding)
   - **Model listing endpoints** (checks available models and their configurations)
   - **Chat completion** with a sample prompt (tests text generation)
   - **Embeddings** with sample texts (tests vector generation with dimension verification)
   
   The script provides detailed output including:
   - Response times for each endpoint
   - Model details (ID, root model, max length)
   - Token usage statistics
   - Embedding dimensions and sample values
   - Comprehensive pass/fail summary
   
   **Example successful output:**
   ```
   Loading environment from: /path/to/.env
   
   ======================================================================
    vLLM ENDPOINT TEST SUITE
   ======================================================================
   
   Chat Service URL: http://localhost:8000/v1
   Embed Service URL: http://localhost:8001/v1
   API Key: ********...XXXX
   
   [Testing Chat Service Health]
     Status: 200 OK
     Response Time: 18.0ms
   
   [Testing Chat Completion]
     Status: 200 OK
     Response Time: 43179.1ms
     Response Details:
       Role: assistant
       Content: [model response]
     Token Usage:
       Prompt Tokens: 37
       Completion Tokens: 50
       Total Tokens: 87
   
   [Testing Embeddings]
     Status: 200 OK
     Response Time: 41624.2ms
     Response Details:
       Number of Embeddings: 2
       Dimensions: 4096
       Magnitude: 1.0000
   
   ======================================================================
    TEST SUMMARY
   ======================================================================
   
     Total Tests: 6
     Passed: 6
     Failed: 0
   
     All tests passed! vLLM services are working correctly.
   ```
   
   **Note:** The script can also use environment variables `VLLM_CHAT_BASE_URL`, `VLLM_EMBED_BASE_URL`, `VLLM_CHAT_MODEL`, and `VLLM_EMBED_MODEL` to override defaults if your services run on different ports or use different model names.

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

LLM-Lasso now supports both **OpenAI** (cloud) and **vLLM** (local open-source) backends. Use the `--llm-backend` argument to switch between them.

**Quick Start with vLLM (Qwen Models):**

```bash
# 1. Ensure vLLM services are running (see Prerequisites above)
# 2. Run LLM-Lasso with vLLM backend

python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/my_prompt.txt \
    --target_column target_var \
    --category "My Category" \
    --llm-backend vllm \
    --pdf_rag
```

**Running with OpenAI (Cloud - Default):**

```bash
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/dataset.csv \
    --feature_names_path /path/to/features.txt \
    --prompt-filename prompts/my_prompt.txt \
    --target_column target_var \
    --category "My Category" \
    --llm-backend openai \
    --model-type gpt-4o \
    --pdf_rag
```

**Backend-Specific Arguments:**

| Argument | OpenAI Backend | vLLM Backend |
|----------|---------------|--------------|
| `--llm-backend` | `openai` (default) | `vllm` |
| `--model-type` | `gpt-4o`, `o1`, `o1-pro`, `openrouter` | `vllm` (uses env vars) |
| `--model-name` | Optional custom model name | Uses `VLLM_CHAT_MODEL` env var |

> **📖 For detailed information about the Qwen models, code integration, and advanced usage, see the [Open-Source LLM Integration with Qwen Models](#open-source-llm-integration-with-qwen-models) section below.**

**OpenAI-Compatible Endpoints:**

| Service | Endpoint | Description |
|---------|----------|-------------|
| Chat | `http://localhost:8000/v1/chat/completions` | Text generation |
| Chat Models | `http://localhost:8000/v1/models` | List available models |
| Embeddings | `http://localhost:8001/v1/embeddings` | Vector embeddings |
| Embed Models | `http://localhost:8001/v1/models` | List available models |
| Health | `http://localhost:8000/health` | Service health check |

### Open-Source LLM Integration with Qwen Models

LLM-Lasso has been extended to support open-source LLMs through vLLM, with **Qwen3 models** as the default configuration. This integration allows you to run the entire LLM-Lasso pipeline locally without relying on cloud APIs.

#### Models Used

The Docker setup deploys two Qwen3 models:

1. **Qwen3-30B-A3B-Thinking-2507-FP8** (`qwen3-thinking`)
   - **Purpose**: Chat completions for penalty score generation
   - **Model Card**: [Qwen/Qwen3-30B-A3B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8)
   - **Architecture**: 30B parameter thinking model with FP8 quantization
   - **VRAM**: ~24GB recommended
   - **Max Context**: 32,768 tokens
   - **Endpoint**: `http://localhost:8000/v1/chat/completions`
   - **Served as**: `qwen3-thinking`

2. **Qwen3-Embedding-8B** (`qwen3-embed`)
   - **Purpose**: Text embeddings for RAG vectorstore creation and retrieval
   - **Model Card**: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
   - **Architecture**: 8B parameter embedding model
   - **VRAM**: ~8GB recommended
   - **Max Context**: 8,192 tokens
   - **Embedding Dimensions**: 4,096
   - **Endpoint**: `http://localhost:8001/v1/embeddings`
   - **Served as**: `qwen3-embed`

#### Code Integration

The integration adds support for vLLM across the entire LLM-Lasso pipeline:

**1. LLM Module (`src/llm_lasso/llm_penalty/llm.py`)**
   - Added `LLMType.VLLM` enum value
   - Created `VLLMLLM` class implementing LangChain's LLM interface
   - Reads configuration from environment variables:
     - `VLLM_CHAT_BASE_URL` (default: `http://localhost:8000/v1`)
     - `VLLM_API_KEY` (for authentication)
     - `VLLM_CHAT_MODEL` (default: `qwen3-thinking`)

**2. Embeddings Module (`src/llm_lasso/llm_penalty/embeddings.py`)**
   - New module with `VLLMEmbeddings` class
   - Implements LangChain's `Embeddings` interface
   - Factory function `get_embeddings(backend="vllm")` for easy initialization
   - Reads configuration from:
     - `VLLM_EMBED_BASE_URL` (default: `http://localhost:8001/v1`)
     - `VLLM_API_KEY` (for authentication)
     - `VLLM_EMBED_MODEL` (default: `qwen3-embed`)

**3. RAG Pipeline Updates**
   - `pdf_vectorstore.py`: Updated to accept any LangChain-compatible embeddings
   - `score_collection.py`: MultiQueryRetriever accepts optional LLM parameter
   - `omim_RAG_process.py`: Updated to support vLLM for query expansion

**4. Main Script (`scripts/run_pbd_llm_lasso.py`)**
   - Added `--llm-backend` argument with choices: `openai`, `vllm`
   - Automatically initializes appropriate embeddings and LLM based on backend
   - Backward compatible: defaults to `openai` if not specified
   - Fully compatible with hyperparameter grid search (works with both backends)

#### Usage Examples

**Example 1: Run LLM-Lasso with Qwen models (local)**

```bash
# 1. Start vLLM services (in one terminal)
docker compose --env-file .env -f opensource_llms/docker-compose.yml up

# 2. Wait for models to load (check logs or health endpoints)

# 3. Run LLM-Lasso with vLLM backend (in another terminal)
python scripts/run_pbd_llm_lasso.py \
    --dataset_path examples/example_data/pbd_focal.csv \
    --feature_names_path examples/example_data/pbd_focal_variables.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Suicidal Ideation" \
    --llm-backend vllm \
    --pdf_rag \
    --save_dir results/qwen_experiment
```

**Example 1b: Using Qwen models with comprehensive grid search**

```bash
# Run with vLLM backend AND comprehensive hyperparameter search
python scripts/run_pbd_llm_lasso.py \
    --dataset_path examples/example_data/pbd_focal.csv \
    --feature_names_path examples/example_data/pbd_focal_variables.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Suicidal Ideation" \
    --llm-backend vllm \
    --pdf_rag \
    --use_loo \
    --lmda_path_size 100 \
    --save_dir results/qwen_with_gridsearch
```

This will:
- Use local Qwen models (no cloud API calls)
- Perform comprehensive grid search (100 λ values)
- Generate grid search plots in `results/qwen_with_gridsearch/gridsearch/`

**Example 2: Using custom vLLM endpoints**

If your vLLM services run on different ports or hosts, configure via `.env`:

```bash
# In .env file
VLLM_CHAT_BASE_URL=http://192.168.1.100:8000/v1
VLLM_EMBED_BASE_URL=http://192.168.1.100:8001/v1
VLLM_CHAT_MODEL=qwen3-thinking
VLLM_EMBED_MODEL=qwen3-embed
VLLM_API_KEY=your_secure_api_key
```

Then run normally:
```bash
python scripts/run_pbd_llm_lasso.py \
    --llm-backend vllm \
    # ... other arguments
```

**Example 3: Switching between OpenAI and vLLM**

```bash
# Use OpenAI (cloud)
python scripts/run_pbd_llm_lasso.py \
    --llm-backend openai \
    --model-type gpt-4o \
    # ... other arguments

# Use vLLM (local Qwen models)
python scripts/run_pbd_llm_lasso.py \
    --llm-backend vllm \
    # ... other arguments (model names come from env vars)
```

#### Benefits of Using Qwen Models

1. **Privacy**: All processing happens locally; no data sent to cloud APIs
2. **Cost**: No API usage fees; only GPU electricity costs
3. **Control**: Full control over model versions, parameters, and infrastructure
4. **Performance**: Lower latency for local deployments
5. **Reproducibility**: Consistent model versions without API changes
6. **Grid Search Compatible**: Works seamlessly with hyperparameter grid search visualization

#### Integration with Hyperparameter Search

The open-source vLLM backend is fully compatible with LLM-Lasso's hyperparameter grid search:

- **Grid Search**: Works identically with both `openai` and `vllm` backends
- **Visualization**: Grid search plots are generated regardless of backend choice
- **Performance**: Local models can be faster for repeated grid search iterations
- **Cost Savings**: No API costs when testing many hyperparameter combinations

Example combining both features:
```bash
# Use local Qwen models + comprehensive grid search
python scripts/run_pbd_llm_lasso.py \
    --llm-backend vllm \
    --use_loo \
    --lmda_path_size 100 \
    # ... other arguments
```

#### Model Characteristics

**Qwen3-30B-A3B-Thinking-2507-FP8:**
- **Thinking Model**: Uses internal reasoning chains for complex problem-solving
- **FP8 Quantization**: Reduced precision for memory efficiency while maintaining quality
- **Context Length**: 32K tokens (suitable for long documents in RAG)
- **Use Case**: Penalty score generation, where reasoning about feature relevance is important

**Qwen3-Embedding-8B:**
- **High Dimensionality**: 4,096 dimensions (vs. OpenAI's 1,536)
- **Long Context**: 8K tokens per document
- **Use Case**: Dense vector embeddings for semantic search in RAG pipeline

#### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VLLM_CHAT_BASE_URL` | Base URL for chat completions | `http://localhost:8000/v1` | No |
| `VLLM_EMBED_BASE_URL` | Base URL for embeddings | `http://localhost:8001/v1` | No |
| `VLLM_CHAT_MODEL` | Model name for chat | `qwen3-thinking` | No |
| `VLLM_EMBED_MODEL` | Model name for embeddings | `qwen3-embed` | No |
| `VLLM_API_KEY` | API key for authentication | (empty) | No* |
| `HUGGINGFACE_TOKEN` | Token for downloading models | - | Yes** |

\* Required if vLLM is configured with `--api-key`  
\** Required for downloading gated Qwen3 models from HuggingFace

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
| `--llm-backend` | LLM backend: `openai` (cloud) or `vllm` (local open-source). See [Using with LLM-Lasso](#using-with-llm-lasso) for vLLM setup. | `openai` |
| `--model-type` | LLM model (`gpt-4o`, `o1`, `o1-pro`, `openrouter` for OpenAI backend) | `gpt-4o` |
| `--temp` | LLM temperature (0-2). Use `0` for deterministic, reproducible results. Higher values (0.7-1.0) add creativity but reduce reproducibility. **Recommended: `0` for research reproducibility.** | `0` |
| `--n-trials` | Number of scoring trials | `1` |
| `--test_size` | Train/test split ratio | `0.2` |
| `--imputation_strategy` | Handle missing values (`median`, `mean`, `most_frequent`) | `median` |
| `--use_loo` | Use Leave-One-Out cross-validation for outer testing loop | `False` |
| `--inner_cv_folds` | Number of inner CV folds for hyperparameter selection | `10` |
| `--lmda_path_size` | Number of lambda values in regularization path (min 50) | `100` |
| `--optimize_metric` | Metric to maximize during hyperparameter selection: `accuracy`, `sensitivity`, `specificity`, `balanced_accuracy`, `f1`, `auc_roc`. See [Optimization Metric Selection](#optimization-metric-selection). | `accuracy` |
| `--compute_ci` | Compute bootstrap confidence intervals for metrics. See [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals). | `True` |
| `--bootstrap_method` | Bootstrap method: `standard` or `632` (.632 bootstrap) | `standard` |
| `--bootstrap_n_rounds` | Number of bootstrap iterations | `1000` |
| `--ci_level` | Confidence level (e.g., 0.95 for 95% CI) | `0.95` |
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
| `confidence_intervals/` | **Bootstrap CI results and plots** (if `--compute_ci`) - see below |
| `gridsearch/` | **Grid search plots and data** (see below) |

#### Confidence Intervals Output (`confidence_intervals/` subfolder)

When `--compute_ci` is enabled (default), this subfolder contains:

| File | Description |
|------|-------------|
| `confidence_intervals.json` | Full CI data for all metrics (point estimate, lower/upper bounds) |
| `metrics_with_ci_bars.png` | Horizontal bar chart showing all metrics with error bars |
| `forest_plot_ci.png` | Forest plot style visualization (common in medical literature) |
| `ci_width_comparison.png` | Bar chart comparing CI widths (narrower = more precise) |
| `ci_summary_dashboard.png` | 4-panel summary with key metrics, sensitivity vs specificity, and more |

#### Grid Search Output (`gridsearch/` subfolder)

LLM-Lasso performs comprehensive hyperparameter search over regularization parameters (λ) to find the optimal model. When using LOO cross-validation (`--use_loo`), the pipeline automatically saves detailed grid search results showing how different λ values affect model performance.

**What is Grid Search?**

Grid search systematically tests multiple regularization parameter values to find the optimal λ that minimizes cross-validation loss. This ensures the Lasso model generalizes well to unseen data.

**Generated Files:**

The gridsearch folder contains three types of outputs:

**1. Final Model Grid Search** (trained on all data after LOO):

| File | Description |
|------|-------------|
| `lambda_vs_accuracy_final_model.png` | λ vs accuracy for the final model |
| `lambda_vs_auc_final_model.png` | λ vs AUC-ROC for the final model |
| `lambda_vs_loss_final_model.png` | λ vs CV loss for the final model |
| `gridsearch_summary_final_model.png` | Combined 3-panel summary |
| `extended_metrics_final_model.png` | Extended metrics (sensitivity, specificity, balanced accuracy, F1) |
| `gridsearch_results_final_model.csv` | Raw data (λ, accuracy, AUC, loss, extended metrics) |

**2. LOO Aggregate Grid Search** (aggregated across all LOO folds):

| File | Description |
|------|-------------|
| `lambda_vs_accuracy_loo_aggregate.png` | Mean accuracy ± std across all LOO folds (with individual fold traces in gray) |
| `lambda_vs_auc_loo_aggregate.png` | Mean AUC-ROC ± std across all LOO folds |
| `lambda_vs_loss_loo_aggregate.png` | Mean CV loss ± std across all LOO folds |
| `gridsearch_summary_loo_aggregate.png` | Combined 3-panel LOO aggregate summary |
| `extended_metrics_loo_aggregate.png` | Extended metrics (sensitivity, specificity, balanced accuracy, F1) aggregated across folds |
| `gridsearch_results_loo_aggregate.csv` | Aggregate statistics (mean, std for each metric including extended metrics) |
| `best_lambda_distribution.png` | Histogram showing distribution of selected λ values across folds |
| `best_lambda_per_fold.csv` | Best λ and accuracy for each LOO fold |

**3. Individual LOO Fold Grid Search** (in `loo_folds/` subfolder):

| File | Description |
|------|-------------|
| `loo_folds/fold_001_gridsearch.png` | Grid search plot for LOO fold 1 |
| `loo_folds/fold_002_gridsearch.png` | Grid search plot for LOO fold 2 |
| `loo_folds/fold_XXX_gridsearch.png` | **ALL** LOO fold grid search plots (one per fold) |

**Note:** All LOO fold gridsearch plots are saved. If you have N samples, you'll get N individual fold plots showing how lambda selection varied across each LOO iteration.

**Understanding the Plots:**

- **X-axis**: Uses `-log₁₀(λ)` (common practice in regularization visualization)
  - Left side = stronger regularization (more features removed)
  - Right side = weaker regularization (more features retained)
- **Y-axis**: Performance metric (accuracy, AUC, or loss)
- **Red vertical line**: Marks the optimal λ that minimizes CV loss
- **Red point**: Shows the best metric value at optimal λ

**Configuration:**

- **Minimum**: At least 50 λ values are tested (ensures adequate search coverage)
- **Default**: 100 λ values (provides fine-grained search)
- **Custom**: Set via `--lmda_path_size` argument

**Example Interpretation:**

```
Optimal λ found at -log₁₀(λ) = 2.5
Best accuracy: 0.85
Best AUC-ROC: 0.91
Minimum CV loss: 0.12
```

This means the model performs best with moderate regularization, balancing feature selection and model complexity.

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

### Hyperparameter Search and Grid Search Visualization

LLM-Lasso includes comprehensive hyperparameter search capabilities with automatic visualization of the search process.

#### How It Works

1. **Regularization Path**: The pipeline tests multiple λ (lambda) values along a regularization path
2. **Cross-Validation**: For each λ, performance is evaluated using k-fold cross-validation
3. **Optimal Selection**: The λ that minimizes CV loss is selected
4. **Visualization**: All results are automatically plotted and saved

#### Key Features

- **Comprehensive Search**: Tests at least 50 λ values by default (configurable up to 100+)
- **Multiple Metrics**: Tracks accuracy, AUC-ROC, and CV loss simultaneously
- **Visual Analysis**: Generates publication-ready plots showing the entire search space
- **Reproducibility**: Saves raw data (CSV) for further analysis

#### Configuration Options

```bash
# Use default (100 lambda values)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    # ... other arguments

# Customize search space (minimum 50)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --lmda_path_size 150 \
    # ... other arguments
```

#### Optimization Metric Selection

By default, LLM-Lasso selects the optimal λ by minimizing the Hamming loss (which is equivalent to maximizing accuracy). However, for imbalanced datasets or specific use cases, you may want to optimize for a different metric.

**Available Metrics:**

| Metric | Description | When to Use |
|--------|-------------|-------------|
| `accuracy` | 1 - Hamming loss (default) | Balanced datasets |
| `sensitivity` | True Positive Rate (Recall) | Medical screening - minimize missed positives |
| `specificity` | True Negative Rate | Minimize false positives |
| `balanced_accuracy` | (Sensitivity + Specificity) / 2 | Imbalanced datasets |
| `f1` | Harmonic mean of precision and recall | Balance precision and recall |
| `auc_roc` | Area under ROC curve | Overall discrimination ability |

**Usage Examples:**

```bash
# Default: maximize accuracy
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --optimize_metric accuracy \
    # ... other arguments

# Maximize sensitivity (important for medical screening - don't miss positives)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --optimize_metric sensitivity \
    # ... other arguments

# Maximize specificity (reduce false positives)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --optimize_metric specificity \
    # ... other arguments

# Use balanced accuracy (recommended for imbalanced datasets)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --optimize_metric balanced_accuracy \
    # ... other arguments
```

**Grid Search Visualization:**

When using an optimization metric other than accuracy, the grid search plots will:
1. Highlight the optimized metric with a "★ OPTIMIZED" label
2. Show all extended metrics (sensitivity, specificity, balanced accuracy, F1) in an additional plot
3. Include the optimization metric in the saved CSV data

The extended metrics plots are saved as:
- `extended_metrics_final_model.png` - For the final model
- `extended_metrics_loo_aggregate.png` - Aggregated across LOO folds

#### Bootstrap Confidence Intervals

LLM-Lasso can compute bootstrap confidence intervals for all performance metrics, providing a measure of statistical uncertainty in your results.

**What are Bootstrap Confidence Intervals?**

Bootstrap confidence intervals estimate how much your reported metrics might vary if you collected multiple independent test sets. This is particularly important for:
- Small datasets where single-point estimates can be unreliable
- Imbalanced datasets where minority class performance drives uncertainty
- Comparing models - overlapping CIs suggest no significant difference

**Available Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| `standard` | Simple resampling with replacement (default) | General purpose, widely accepted |
| `632` | .632 bootstrap with bias correction | When concerned about optimistic bias |

**Usage:**

```bash
# Default: 95% CI with standard bootstrap (1000 rounds)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --compute_ci \
    # ... other arguments

# Customize bootstrap settings
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --compute_ci \
    --bootstrap_method 632 \
    --bootstrap_n_rounds 2000 \
    --ci_level 0.99 \
    # ... other arguments

# Disable confidence intervals (faster)
python scripts/run_pbd_llm_lasso.py \
    --use_loo \
    --no-compute_ci \
    # ... other arguments
```

**Output:**

When `--compute_ci` is enabled (default), the pipeline outputs:
- **Console logs** with CIs: `Accuracy: 0.8523 (95% CI: 0.7821 - 0.9104)`
- **`confidence_intervals/` subfolder** containing:
  - `confidence_intervals.json` - Full CI data for all metrics
  - `metrics_with_ci_bars.png` - Horizontal bar chart with error bars
  - `forest_plot_ci.png` - Forest plot style visualization
  - `ci_width_comparison.png` - Comparison of CI widths (precision)
  - `ci_summary_dashboard.png` - 4-panel summary dashboard
- **`summary.json`**: Includes CI bounds for key metrics

**Metrics with Confidence Intervals:**

- Accuracy, Balanced Accuracy
- Sensitivity (Recall), Specificity
- Precision, F1 Score
- AUROC, Average Precision
- Matthews Correlation Coefficient (MCC)

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--compute_ci` | Enable bootstrap CI computation | `True` |
| `--bootstrap_method` | `standard` or `632` | `standard` |
| `--bootstrap_n_rounds` | Number of bootstrap iterations | `1000` |
| `--ci_level` | Confidence level (0.90, 0.95, 0.99) | `0.95` |

#### Output Location

All grid search plots and data are saved in:
```
{save_dir}/gridsearch/
├── *_final_model.*          # Final model grid search
├── *_loo_aggregate.*        # Aggregated LOO results (mean ± std)
├── best_lambda_*.png/csv    # Lambda distribution across folds
└── loo_folds/               # Individual LOO fold plots
    ├── fold_001_gridsearch.png
    ├── fold_002_gridsearch.png
    └── ...
```

This includes:
- **Final model plots**: Grid search for model trained on all data
- **LOO aggregate plots**: Mean ± std across all LOO folds (with individual traces)
- **Lambda distribution**: How the selected λ varies across folds
- **Individual fold plots**: **ALL** LOO fold grid searches (one plot per LOO iteration)

See [Grid Search Output](#grid-search-output-gridsearch-subfolder) section above for detailed file descriptions.

### Complete Example

Here's a full example running LLM-Lasso with PDF RAG on a clinical dataset:

**Example A: Using OpenAI with Grid Search**

```bash
# Activate virtual environment
source virtualenv/bin/activate

# Run the full pipeline with OpenAI backend
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/my_clinical_data.csv \
    --feature_names_path /path/to/clinical_features.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Suicidal Ideation" \
    --llm-backend openai \
    --model-type gpt-4o \
    --pdf_rag \
    --pdf_rag_num_docs 3 \
    --use_loo \
    --lmda_path_size 100 \
    --optimize_metric accuracy \
    --temp 0 \
    --n-trials 1 \
    --test_size 0.3 \
    --imputation_strategy median \
    --log_level INFO \
    --wipe
```

**Example B: Using Open-Source Qwen Models with Grid Search**

```bash
# 1. Start vLLM services (in one terminal)
docker compose --env-file .env -f opensource_llms/docker-compose.yml up

# 2. Run LLM-Lasso with vLLM backend (in another terminal)
python scripts/run_pbd_llm_lasso.py \
    --dataset_path /path/to/my_clinical_data.csv \
    --feature_names_path /path/to/clinical_features.txt \
    --prompt-filename prompts/pbd_normal.txt \
    --target_column target_var \
    --category "Suicidal Ideation" \
    --llm-backend vllm \
    --pdf_rag \
    --pdf_rag_num_docs 3 \
    --use_loo \
    --lmda_path_size 100 \
    --optimize_metric balanced_accuracy \
    --temp 0 \
    --n-trials 1 \
    --test_size 0.3 \
    --imputation_strategy median \
    --log_level INFO \
    --wipe
```

**Outputs Generated:**

Both examples will create:
- `penalty_scores.json` - LLM-generated feature penalties
- `gridsearch/` folder - Comprehensive hyperparameter search results:
  - **Final model plots**: `*_final_model.png` and `.csv`
  - **LOO aggregate plots**: `*_loo_aggregate.png` (mean ± std across all folds)
  - **Best λ distribution**: `best_lambda_distribution.png` and `best_lambda_per_fold.csv`
  - **Individual fold plots**: `loo_folds/fold_XXX_gridsearch.png` (up to 20 folds)
- `loo_predictions.csv` - Cross-validation predictions
- `model_coefficients.json` - Final model coefficients
- Evaluation plots (ROC, PR curves, etc.)

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
| `--lmda_path_size` | Number of lambda (regularization) parameters to search. Gridsearch plots are saved in `gridsearch/` subfolder. | `100` |

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

   **Option A: Using OpenAI Embeddings (default)**
   ```bash
   # Requires OPENAI_API_KEY in .env
   python scripts/index_pdf_vectorstore.py \
       --pdf-directory ./sample_pdfs \
       --persist-directory ./pdf_vectorstore \
       --embedding-backend openai
   ```

   **Option B: Using vLLM Embeddings (open-source)**
   ```bash
   # First, start the vLLM embedding service
   docker compose --env-file .env -f opensource_llms/docker-compose.yml up
   
   # Then index with vLLM (uses Qwen3-Embedding-8B, 4096-dim embeddings)
   python scripts/index_pdf_vectorstore.py \
       --pdf-directory ./sample_pdfs \
       --persist-directory ./pdf_vectorstore \
       --embedding-backend vllm
   ```

   **Key Options:**
   - `--chunk-size`: Text chunk size (default: 1000 chars)
   - `--chunk-overlap`: Overlap between chunks (default: 200 chars)
   - `--collection-name`: ChromaDB collection name (default: pdf_documents)
   - `--no-filter-references`: Include reference sections (filtered by default)
   - `--log-level DEBUG`: Verbose output

   **IMPORTANT:** The indexer cleans (deletes) any existing vectorstore before creating a fresh index. Use `--no-clean` to append to an existing index instead.

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

   **IMPORTANT:** Use the **same embedding backend** for indexing and pipeline execution!

   ```bash
   # If indexed with OpenAI embeddings:
   python scripts/run_pbd_llm_lasso.py \
       --dataset_path /path/to/dataset.csv \
       --feature_names_path /path/to/features.txt \
       --prompt-filename prompts/my_prompt.txt \
       --target_column target_var \
       --category "My Research Domain" \
       --llm-backend openai \
       --pdf_rag \
       --pdf_rag_num_docs 3

   # If indexed with vLLM embeddings:
   python scripts/run_pbd_llm_lasso.py \
       --dataset_path /path/to/dataset.csv \
       --feature_names_path /path/to/features.txt \
       --prompt-filename prompts/my_prompt.txt \
       --target_column target_var \
       --category "My Research Domain" \
       --llm-backend vllm \
       --pdf_rag \
       --pdf_rag_num_docs 3
   ```

### Vectorstore Indexing Improvements

**Recent Updates (2025):**

The PDF vectorstore indexing system has been enhanced with the following improvements:

#### 1. **Standalone Indexing Script**

A new dedicated indexing script (`scripts/index_pdf_vectorstore.py`) provides a clean CLI interface for creating vectorstores:

- **Unified interface** for both OpenAI and vLLM embedding backends
- **Automatic cleanup** of existing indexes before recreating (ensures fresh, consistent indexes)
- **Comprehensive logging** and progress reporting
- **Connection testing** for vLLM endpoints before indexing begins
- **Flexible configuration** via command-line arguments

**Key Features:**
- Supports both `openai` and `vllm` embedding backends
- Automatically cleans existing vectorstore before indexing (use `--no-clean` to append)
- Tests vLLM connection before starting (prevents wasted time on failed indexing)
- Detailed logging with progress indicators
- Validates PDF directory and file counts before processing

#### 2. **Automatic Index Cleanup**

**Before:** Re-indexing would append to existing collections, leading to duplicate documents and inconsistent indexes.

**After:** By default, the indexer **automatically deletes** any existing vectorstore at the target directory before creating a fresh index. This ensures:
- ✅ Clean, consistent indexes every time
- ✅ No duplicate documents from previous runs
- ✅ Predictable behavior for reproducibility
- ✅ Option to append (`--no-clean`) when needed

**Usage:**
```bash
# Default: Cleans existing index, creates fresh one
python scripts/index_pdf_vectorstore.py \
    --pdf-directory ./sample_pdfs \
    --persist-directory ./pdf_vectorstore \
    --embedding-backend vllm

# Append mode: Adds to existing index (if it exists)
python scripts/index_pdf_vectorstore.py \
    --pdf-directory ./sample_pdfs \
    --persist-directory ./pdf_vectorstore \
    --embedding-backend vllm \
    --no-clean
```

#### 3. **Embedding Backend Compatibility**

The indexing system now fully supports both embedding backends with proper dimension handling:

| Backend | Model | Embedding Dimension | Use Case |
|---------|-------|---------------------|----------|
| **OpenAI** | `text-embedding-ada-002` | 1536 | Cloud-based, fast, reliable |
| **vLLM** | `Qwen3-Embedding-8B` | 4096 | Open-source, local, high-quality |

**Important Notes:**
- **ChromaDB handles both dimensions** without issues (no code changes needed)
- **You MUST use the same backend** for indexing and querying
- **vLLM embeddings are higher-dimensional** (4096 vs 1536), potentially capturing more semantic nuance
- **Dimension mismatch detection**: The pipeline will fail if you try to query an OpenAI-indexed vectorstore with vLLM embeddings (or vice versa)

#### 4. **Improved Error Handling**

- **Connection validation**: Tests vLLM endpoint before starting indexing
- **Clear error messages**: Provides actionable guidance when services are unavailable
- **Graceful failures**: Stops early with helpful error messages rather than failing mid-process

**Example Error Message:**
```
Failed to connect to vLLM embedding service at http://localhost:8001/v1: ...
Make sure the vLLM embedding service is running:
  docker compose --env-file .env -f opensource_llms/docker-compose.yml up
```

#### 5. **Migration from Old Workflow**

**Old Method (deprecated):**
```bash
python playground/interactive_pdf_RAG.py  # Interactive, OpenAI-only
```

**New Method (recommended):**
```bash
# OpenAI
python scripts/index_pdf_vectorstore.py \
    --pdf-directory ./sample_pdfs \
    --embedding-backend openai

# vLLM
python scripts/index_pdf_vectorstore.py \
    --pdf-directory ./sample_pdfs \
    --embedding-backend vllm
```

**Benefits:**
- ✅ Scriptable (can be automated/CI)
- ✅ Supports both backends
- ✅ Better logging and progress reporting
- ✅ Automatic cleanup prevents stale data
- ✅ Non-interactive (better for batch processing)

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
from llm_lasso.llm_penalty.embeddings import get_embeddings

# Option 1: Using OpenAI embeddings
embeddings = get_embeddings(backend="openai")

# Option 2: Using vLLM embeddings (requires vLLM service running)
embeddings = get_embeddings(backend="vllm")

# Create vectorstore from PDFs
# Note: clean_existing=True (default) deletes any existing vectorstore first
vectorstore = create_pdf_vectorstore(
    pdf_directory="sample_pdfs",
    persist_directory="pdf_vectorstore",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model=embeddings,
    filter_references=True,  # Enabled by default - filters out bibliography sections
    clean_existing=True      # Default: deletes existing index before creating fresh one
)

# To append to existing index instead of cleaning:
vectorstore = create_pdf_vectorstore(
    pdf_directory="sample_pdfs",
    persist_directory="pdf_vectorstore",
    embedding_model=embeddings,
    clean_existing=False  # Append mode - adds to existing collection
)

# Load existing vectorstore (must use same embedding backend as when indexing)
vectorstore = load_pdf_vectorstore(
    persist_directory="pdf_vectorstore",
    embedding_model=embeddings  # Must match the backend used during indexing
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
│   ├── .env.example       # Template for required environment variables
│   └── test_vllm_endpoints.py  # Test script for verifying vLLM services
├── playground/            # Interactive scripts
│   ├── interactive_pdf_RAG.py    # Interactive PDF RAG (legacy, use scripts/index_pdf_vectorstore.py)
│   └── view_all_documents.py     # Vectorstore inspection utility
├── prompts/               # LLM prompt templates
├── sample_pdfs/           # Directory for PDF documents
├── scripts/               # Pipeline execution scripts
│   ├── run_pbd_llm_lasso.py      # Main pipeline script
│   ├── index_pdf_vectorstore.py  # PDF vectorstore indexing (OpenAI/vLLM support)
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
