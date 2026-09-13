from langgraph.graph import END, START, StateGraph

from src.components.crag.nodes import (
    evaluate_retrieval_node,
    generate_node,
    knowledge_refinement_node,
    retrieve_resume_node,
    web_search_node,
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

builder.add_edge("generate", END)

crag_graph = builder.compile()