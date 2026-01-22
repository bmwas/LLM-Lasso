"""
Embedding provider abstraction for LLM-Lasso.

This module provides support for multiple embedding backends:
- OpenAI: Uses langchain_openai.OpenAIEmbeddings
- vLLM: Uses local vLLM server with OpenAI-compatible API

Configuration for vLLM is done via environment variables:
    - VLLM_EMBED_BASE_URL: Base URL for vLLM embeddings endpoint (default: http://localhost:8001/v1)
    - VLLM_API_KEY: API key for authentication (if configured in vLLM)
    - VLLM_EMBED_MODEL: Model name served by vLLM (default: qwen3-embed)

Usage:
    from llm_lasso.llm_penalty.embeddings import get_embeddings, EmbeddingBackend
    
    # Get OpenAI embeddings (default)
    embeddings = get_embeddings(backend="openai")
    
    # Get vLLM embeddings
    embeddings = get_embeddings(backend="vllm")
"""

import os
import json
import logging
from enum import Enum
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingBackend(Enum):
    """Supported embedding backends."""
    OPENAI = "openai"
    VLLM = "vllm"


class VLLMEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings class for vLLM's OpenAI-compatible API.
    
    vLLM provides a drop-in replacement for OpenAI's embeddings API, allowing
    the use of open-source embedding models like Qwen3-Embedding.
    
    Configuration is done via environment variables or constructor arguments:
        - VLLM_EMBED_BASE_URL: Base URL for vLLM embeddings endpoint
        - VLLM_API_KEY: API key for authentication (optional)
        - VLLM_EMBED_MODEL: Model name served by vLLM
    
    Example:
        embeddings = VLLMEmbeddings()
        vectors = embeddings.embed_documents(["Hello world", "Goodbye world"])
        query_vector = embeddings.embed_query("Hello")
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize VLLMEmbeddings.
        
        Args:
            base_url: Base URL for vLLM API. Defaults to VLLM_EMBED_BASE_URL env var
                      or http://localhost:8001/v1
            api_key: API key for authentication. Defaults to VLLM_API_KEY env var.
            model: Model name. Defaults to VLLM_EMBED_MODEL env var or qwen3-embed.
            timeout: Request timeout in seconds (default: 300 for large batches).
        """
        self.base_url = base_url or os.environ.get(
            "VLLM_EMBED_BASE_URL", "http://localhost:8001/v1"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.model = model or os.environ.get("VLLM_EMBED_MODEL", "qwen3-embed")
        self.timeout = timeout
        
        logger.debug(f"VLLMEmbeddings initialized: base_url={self.base_url}, model={self.model}")
    
    def _get_headers(self) -> dict:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if not texts:
            return []
        
        url = f"{self.base_url.rstrip('/')}/embeddings"
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        logger.debug(f"Requesting embeddings for {len(texts)} texts from {url}")
        
        try:
            response = requests.post(
                url=url,
                headers=self._get_headers(),
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = []
                
                # Sort by index to maintain order
                data = sorted(result.get("data", []), key=lambda x: x.get("index", 0))
                
                for item in data:
                    embedding = item.get("embedding", [])
                    embeddings.append(embedding)
                
                logger.debug(f"Received {len(embeddings)} embeddings, dimension={len(embeddings[0]) if embeddings else 0}")
                return embeddings
            else:
                error_msg = f"vLLM Embeddings Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"vLLM Embeddings request failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of documents to embed.
            
        Returns:
            List of embeddings, one for each document.
        """
        return self._embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector for the query.
        """
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []


def get_embeddings(
    backend: str = "openai",
    **kwargs
) -> Embeddings:
    """
    Factory function to get the appropriate embeddings instance.
    
    Args:
        backend: Embedding backend to use. One of "openai" or "vllm".
        **kwargs: Additional arguments passed to the embeddings constructor.
    
    Returns:
        LangChain Embeddings instance.
    
    Raises:
        ValueError: If an unsupported backend is specified.
    
    Example:
        # OpenAI embeddings (default)
        embeddings = get_embeddings(backend="openai")
        
        # vLLM embeddings with custom base URL
        embeddings = get_embeddings(
            backend="vllm",
            base_url="http://myserver:8001/v1",
            model="custom-embed-model"
        )
    """
    backend_lower = backend.lower()
    
    if backend_lower == EmbeddingBackend.OPENAI.value:
        from langchain_openai import OpenAIEmbeddings
        logger.info("Using OpenAI embeddings backend")
        return OpenAIEmbeddings(**kwargs)
    
    elif backend_lower == EmbeddingBackend.VLLM.value:
        logger.info("Using vLLM embeddings backend")
        return VLLMEmbeddings(**kwargs)
    
    else:
        supported = [b.value for b in EmbeddingBackend]
        raise ValueError(
            f"Unsupported embedding backend: '{backend}'. "
            f"Supported backends: {supported}"
        )
