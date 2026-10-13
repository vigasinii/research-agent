"""MCP-style tool definitions as plain Python functions."""
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llm import call_llm

logger = logging.getLogger(__name__)

_model = None
_qdrant = None
COLLECTION = "research_docs"
TOP_K = 5


def _log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] [TOOL] {msg}")


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host="localhost", port=6333)
    return _qdrant


def retrieve(query: str) -> list[dict]:
    """
    Search the Qdrant knowledge base by semantic similarity.

    Args:
        query: Natural language search query.

    Returns:
        Top 5 matching chunks, each with keys: source, chunk_index, text, score.
    """
    _log(f"retrieve | query={query!r}")
    model = _get_model()
    qdrant = _get_qdrant()
    vector = model.encode(query).tolist()
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=TOP_K,
        with_payload=True,
    )
    results = [
        {
            "source": h.payload["source"],
            "chunk_index": h.payload["chunk_index"],
            "text": h.payload["text"],
            "score": round(h.score, 4),
        }
        for h in hits
    ]
    _log(f"retrieve | returned {len(results)} chunks, top score={results[0]['score'] if results else 'n/a'}")
    return results


def check_sufficiency(context: list[dict], question: str) -> bool:
    """
    Use the LLM to judge whether the retrieved context is sufficient to answer the question.

    Args:
        context: List of retrieved chunks (dicts with 'text' key).
        question: The user's question.

    Returns:
        True if the context is sufficient, False otherwise.
    """
    _log(f"check_sufficiency | question={question!r} | context_chunks={len(context)}")
    if not context:
        _log("check_sufficiency | no context → insufficient")
        return False

    combined = "\n\n".join(c["text"] for c in context)
    prompt = (
        "You are an expert evaluator. Given the following context passages and a question, "
        "decide whether the context contains enough specific information to give a complete and accurate answer.\n\n"
        f"CONTEXT:\n{combined}\n\n"
        f"QUESTION: {question}\n\n"
        "Reply with exactly one word: YES if the context is sufficient, NO if it is not. "
        "Do not explain."
    )
    response = call_llm(prompt).strip().upper()
    sufficient = response.startswith("YES")
    _log(f"check_sufficiency | llm_response={response!r} → sufficient={sufficient}")
    return sufficient


def generate_answer(context: list[dict], question: str) -> dict:
    """
    Generate a grounded answer using only the retrieved context.

    Args:
        context: List of retrieved chunks (dicts with 'text' and 'source' keys).
        question: The user's question.

    Returns:
        Dict with keys: answer (str), sources (list[str]), grounded (bool).
        If context is empty or insufficient, returns a refusal with grounded=False.
    """
    _log(f"generate_answer | question={question!r} | context_chunks={len(context)}")
    if not context:
        _log("generate_answer | empty context → refusing")
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
            "grounded": False,
        }

    combined = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}" for c in context
    )
    sources = sorted({c["source"] for c in context})

    prompt = (
        "You are a research assistant. Answer the question using ONLY the provided context passages. "
        "Do not use any external knowledge. If the context does not contain enough information to answer "
        "the question, reply with exactly: I don't have enough information to answer this.\n\n"
        f"CONTEXT:\n{combined}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    answer = call_llm(prompt).strip()
    grounded = "I don't have enough information" not in answer
    _log(f"generate_answer | grounded={grounded} | sources={sources}")
    return {"answer": answer, "sources": sources, "grounded": grounded}
