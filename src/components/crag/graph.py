from langgraph.graph import END, START, StateGraph

from src.components.crag.nodes import (
    evaluate_retrieval_node,
    generate_node,
    knowledge_refinement_node,
    retrieve_resume_node,
    web_search_node,
    evaluate_support_node,
    revise_answer_node,
    evaluate_usefulness_node,
    rewrite_query_node
)
from src.components.crag.state import CRAGState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def route_retrieval(state: CRAGState) -> str | list[str]:

    status = state["retrieval_status"]

    if status == "correct":
        return "knowledge_refinement"

    if status == "ambiguous":
        return [
            "knowledge_refinement",
            "web_search"
        ]

    return "web_search"


def route_support(state: CRAGState) -> str:

    if state["support_status"] == "supported":
        return "evaluate_usefulness"

    if state.get("revision_count", 0) >= 2:
        return "evaluate_usefulness"

    return "revise_answer"


def route_usefulness(state: CRAGState) -> str:

    if state["usefulness_status"] == "useful":
        return "end"

    if state.get("rewrite_count", 0) >= 2:
        return "end"

    return "rewrite_query"


builder = StateGraph(CRAGState)

builder.add_node(
    "retrieve_resume",
    retrieve_resume_node
)

builder.add_node(
    "evaluate_retrieval",
    evaluate_retrieval_node
)

builder.add_node(
    "knowledge_refinement",
    knowledge_refinement_node
)

builder.add_node(
    "web_search",
    web_search_node
)

builder.add_node(
    "generate",
    generate_node
)

builder.add_node(
    "evaluate_support",
    evaluate_support_node
)

builder.add_node(
    "revise_answer",
    revise_answer_node
)

builder.add_node(
    "evaluate_usefulness",
    evaluate_usefulness_node
)

builder.add_node(
    "rewrite_query",
    rewrite_query_node
)

builder.add_edge(START, "retrieve_resume")

builder.add_edge("retrieve_resume", "evaluate_retrieval")

builder.add_conditional_edges(
    "evaluate_retrieval", 
    route_retrieval,
    {
        "knowledge_refinement": "knowledge_refinement",
        "web_search": "web_search"
    }
)

builder.add_edge("knowledge_refinement", "generate")

builder.add_edge("web_search", "generate")

builder.add_edge("generate", "evaluate_support")

builder.add_conditional_edges(
    "evaluate_support",
    route_support,
    {
        "revise_answer": "revise_answer",
        "evaluate_usefulness": "evaluate_usefulness"
    }
)

builder.add_edge("revise_answer", "evaluate_support")

builder.add_conditional_edges(
    "evaluate_usefulness",
    route_usefulness,
    {
        "rewrite_query": "rewrite_query",
        "end": END
    }
)

builder.add_edge(
    "rewrite_query",
    "retrieve_resume"
)

crag_graph = builder.compile()