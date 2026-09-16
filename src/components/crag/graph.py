from langgraph.graph import END, START, StateGraph

from src.components.crag.nodes import (
    evaluate_retrieval_node,
    knowledge_refinement_node,
    retrieve_resume_node
)
from src.components.crag.state import CRAGState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def route_retrieval(state: CRAGState) -> str:

    status = state["retrieval_status"]

    if status in {
        "correct",
        "ambiguous",
    }:
        return "knowledge_refinement"

    return "end"


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

builder.add_edge(START, "retrieve_resume")

builder.add_edge("retrieve_resume", "evaluate_retrieval")

builder.add_conditional_edges(
    "evaluate_retrieval", 
    route_retrieval,
    {
        "knowledge_refinement": "knowledge_refinement",
        "end": END
    }
)

builder.add_edge("knowledge_refinement", END)

crag_graph = builder.compile()

logger.info("Compiled CRAG subgraph")