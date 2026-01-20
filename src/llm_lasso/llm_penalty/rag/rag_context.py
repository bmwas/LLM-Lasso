"""
This script offers the main function to process retrieved RAG documents using features provided in 
omim_RAG_process.py, pubMed_RAG_process.py, and pdf_RAG_process.py.
"""

from typing import Optional
from llm_lasso.llm_penalty.rag.omim_RAG_process import *
from llm_lasso.llm_penalty.rag.pubMed_RAG_process import pubmed_retrieval
from llm_lasso.llm_penalty.rag.pdf_RAG_process import get_pdf_retrieval_context
from llm_lasso.utils.score_collection import retrieval_docs, get_unique_docs
from langchain_community.vectorstores import Chroma


def get_rag_context(
    batch_genes: list[str],
    category: str,
    vectorstore: Chroma,
    model: LLMQueryWrapperWithMemory,
    omim_api_key: str,
    pubmed_docs: bool = False,
    filtered_cancer_docs: bool = False,
    summarized_gene_docs: bool = False,
    original_docs: bool = True,
    pdf_docs: bool = False,
    pdf_vectorstore: Optional[Chroma] = None,
    pdf_num_docs: int = 3,
    default_num_docs: int = 3,
    small: bool = False,
    prompt_constr: bool = False
):
    """
    Retrieve RAG context for gene data, combining multiple RAG processes:
    - OMIM original docs
    - OMIM filtered cancer docs
    - OMIM summarized gene docs  
    - PubMed docs
    - PDF docs (local scientific papers)
    
    Args:
        batch_genes: List of gene names to retrieve context for.
        category: The category/domain context (e.g., cancer type).
        vectorstore: OMIM Chroma vectorstore.
        model: LLM query wrapper for summarization tasks.
        omim_api_key: API key for OMIM access.
        pubmed_docs: Whether to retrieve from PubMed.
        filtered_cancer_docs: Whether to retrieve filtered cancer docs from OMIM.
        summarized_gene_docs: Whether to retrieve summarized gene docs from OMIM.
        original_docs: Whether to retrieve original OMIM docs.
        pdf_docs: Whether to retrieve from PDF vectorstore.
        pdf_vectorstore: Chroma vectorstore containing PDF documents.
        pdf_num_docs: Number of PDF documents to retrieve per query.
        default_num_docs: Number of OMIM documents to retrieve.
        small: Whether to use smaller context for limited context LLMs.
        prompt_constr: Whether to use LangChain's prompt construction framework.
    
    Returns:
        Concatenated context string from all enabled RAG sources.
    """
    context = ""
    skip_genes = set()
    
    if pubmed_docs:
        print("Retrieving pubmed")
        context += pubmed_retrieval(batch_genes, category, model) + "\n"

    if filtered_cancer_docs:
        print("Retrieving cancer docs")
        (add_ctx, skip_genes) = get_filtered_cancer_docs_and_genes_found(
            batch_genes, vectorstore.as_retriever(search_kwargs={"k": 100}),
            model, category
        )
        context += add_ctx + "\n"

    if summarized_gene_docs:
        print("Retrieving gene docs")
        preamble = "\nAdditional gene information: \n" if context.strip() != "" else ""
        context += preamble + get_summarized_gene_docs(
            [gene for gene in batch_genes if gene not in skip_genes],
            model, omim_api_key
        ) + "\n"

    if original_docs:
        print("Retrieving original docs")
        docs = retrieval_docs(
            batch_genes, category,
            vectorstore.as_retriever(search_kwargs={"k": default_num_docs}),
            small=small, prompt_constr=prompt_constr
        )
        unique_docs = get_unique_docs(docs)
        context = "\n".join([doc.page_content for doc in unique_docs])

    if pdf_docs and pdf_vectorstore is not None:
        print("Retrieving PDF docs")
        pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": pdf_num_docs})
        pdf_context = get_pdf_retrieval_context(
            batch_genes, category, pdf_retriever, num_docs=pdf_num_docs
        )
        if pdf_context:
            preamble = "\n\nRelevant information from scientific papers:\n" if context.strip() != "" else "Relevant information from scientific papers:\n"
            context += preamble + pdf_context

    return context.strip()