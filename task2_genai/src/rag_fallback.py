"""
rag_fallback.py
----------------
Bonus - RAG Fallback Layer (up to 5 points).

When the fine-tuned model's self-rated confidence is "low" (a proxy for
perplexity-based uncertainty that avoids needing raw logits, which are not
always exposed by inference servers), retrieve relevant context from a
ChromaDB vector store built from the training-domain policy documents and
re-query the model with the retrieved excerpt appended.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from . import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LOW_CONFIDENCE_VALUES = {"low"}


def build_policy_vector_store(policy_documents: List[Dict[str, str]], persist_dir: str = "./chroma_policy_db"):
    """
    Build (or load) a local Chroma collection from full policy documents.

    `policy_documents` is a list of {"id": ..., "topic": ..., "text": ...} dicts -
    typically the full-length source policies your synthetic excerpts were drawn
    from (or your firm's real policy manual, if adapting this for actual use).
    """
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(name="compliance_policies", embedding_function=embed_fn)

    if collection.count() == 0:
        collection.add(
            ids=[doc["id"] for doc in policy_documents],
            documents=[doc["text"] for doc in policy_documents],
            metadatas=[{"topic": doc.get("topic", "unknown")} for doc in policy_documents],
        )
        logger.info("Indexed %d policy documents into Chroma.", len(policy_documents))
    else:
        logger.info("Loaded existing Chroma collection with %d documents.", collection.count())

    return collection


def retrieve_context(collection, query: str, n_results: int = 2) -> List[str]:
    results = collection.query(query_texts=[query], n_results=n_results)
    return results.get("documents", [[]])[0]


def needs_rag_fallback(model_response: Dict[str, Any]) -> bool:
    """Trigger condition: the model reported low confidence in its own structured output."""
    return str(model_response.get("confidence", "")).lower() in LOW_CONFIDENCE_VALUES


def rag_fallback_query(
    collection,
    employee_question: str,
    original_excerpt: str,
    original_response: Dict[str, Any],
    query_model_fn,
) -> Dict[str, Any]:
    """
    Re-query the model with retrieved context appended, if the original response
    signalled low confidence. `query_model_fn(policy_excerpt, employee_question)`
    should call your fine-tuned model and return a parsed response dict, in the
    same shape used throughout evaluate.py.
    """
    if not needs_rag_fallback(original_response):
        return original_response

    logger.info("Low confidence detected - retrieving additional context via RAG.")
    retrieved_docs = retrieve_context(collection, employee_question, n_results=2)
    augmented_excerpt = original_excerpt + "\n\nAdditional retrieved context:\n" + "\n---\n".join(retrieved_docs)

    new_response = query_model_fn(augmented_excerpt, employee_question)
    logger.info(
        "RAG fallback result - before: %s, after: %s",
        json.dumps(original_response), json.dumps(new_response),
    )
    return new_response
