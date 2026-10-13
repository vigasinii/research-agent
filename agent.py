"""LangGraph state machine for the research agent."""
import logging
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END
from llm import call_llm
from tools import retrieve, check_sufficiency, generate_answer

logger = logging.getLogger(__name__)


def _log(node: str, state: dict) -> None:
    summary = {
        "question": state.get("question"),
        "query": state.get("query"),
        "context_chunks": len(state.get("context", [])),
        "expanded": state.get("expanded"),
        "grounded": state.get("grounded"),
    }
    print(f"[{datetime.now().isoformat(timespec='seconds')}] [AGENT:{node}] {summary}")


class AgentState(TypedDict):
    question: str
    query: str
    context: list[dict]
    expanded: bool
    answer: str
    sources: list[str]
    grounded: bool


# ── Nodes ────────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    _log("retrieve_node", state)
    chunks = retrieve(state["query"])
    return {**state, "context": chunks}


def check_node(state: AgentState) -> AgentState:
    _log("check_node", state)
    # sufficiency is read by the router; store it as a transient flag
    sufficient = check_sufficiency(state["context"], state["question"])
    # Store the result so the router can read it without re-calling the LLM
    return {**state, "_sufficient": sufficient}  # type: ignore[misc]


def expand_node(state: AgentState) -> AgentState:
    _log("expand_node", state)
    prompt = (
        f"Rephrase the following question to improve semantic search retrieval. "
        f"Return only the rephrased question, nothing else.\n\nQuestion: {state['question']}"
    )
    new_query = call_llm(prompt).strip()
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [AGENT:expand_node] "
        f"rephrased query: {new_query!r}"
    )
    extra_chunks = retrieve(new_query)
    # Deduplicate by (source, chunk_index), keeping original order
    seen = {(c["source"], c["chunk_index"]) for c in state["context"]}
    merged = list(state["context"])
    for chunk in extra_chunks:
        key = (chunk["source"], chunk["chunk_index"])
        if key not in seen:
            merged.append(chunk)
            seen.add(key)
    return {**state, "context": merged, "expanded": True, "query": new_query}


def answer_node(state: AgentState) -> AgentState:
    _log("answer_node", state)
    result = generate_answer(state["context"], state["question"])
    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"],
        "grounded": result["grounded"],
    }


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_check(state: AgentState) -> str:
    """Deterministic edge: go to expand if insufficient and not yet expanded."""
    sufficient = state.get("_sufficient", False)  # type: ignore[arg-type]
    expanded = state.get("expanded", False)
    if sufficient or expanded:
        return "answer_node"
    return "expand_node"


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("check_node", check_node)
    graph.add_node("expand_node", expand_node)
    graph.add_node("answer_node", answer_node)

    graph.set_entry_point("retrieve_node")
    graph.add_edge("retrieve_node", "check_node")
    graph.add_conditional_edges("check_node", route_after_check)
    graph.add_edge("expand_node", "answer_node")
    graph.add_edge("answer_node", END)
    return graph


_compiled = None


def run_agent(question: str) -> dict:
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()

    initial: AgentState = {
        "question": question,
        "query": question,
        "context": [],
        "expanded": False,
        "answer": "",
        "sources": [],
        "grounded": False,
    }
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [AGENT] starting | question={question!r}"
    )
    final = _compiled.invoke(initial)
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [AGENT] done | "
        f"grounded={final['grounded']} | sources={final['sources']}"
    )
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "grounded": final["grounded"],
    }
