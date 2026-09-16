from langgraph.graph import END, START, StateGraph

from src.components.graphrag.retriever import retrieve_resume_graph
from src.components.graphrag.state import GraphRAGState

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def retrieve_graph_node(
    state: GraphRAGState,
) -> dict:

    graph_facts = await retrieve_resume_graph(
        user_id=state["user_id"],
        resume_id=state["resume_id"],
    )

    return {
        "graph_facts": graph_facts
    }


async def format_graph_context_node(
    state: GraphRAGState,
) -> dict:

    graph_facts = state.get("graph_facts", [])

    graph_context = "\n".join(
        f"{fact.source_name} "
        f"-[{fact.relationship}]-> "
        f"{fact.target_name}"
        for fact in graph_facts
    )

    return {
        "graph_context": graph_context
    }


builder = StateGraph(GraphRAGState)

builder.add_node(
    "retrieve_graph",
    retrieve_graph_node,
)

builder.add_node(
    "format_graph_context",
    format_graph_context_node,
)

builder.add_edge(
    START,
    "retrieve_graph",
)

builder.add_edge(
    "retrieve_graph",
    "format_graph_context",
)


builder.add_edge(
    "format_graph_context",
    END,
)

graphrag_graph = builder.compile()

logger.info("Compiled graphRAG.")