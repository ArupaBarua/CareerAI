from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from src.components.graphrag.extractor import (
    graph_extraction_chain,
)
from src.components.graphrag.state import (
    GraphIngestionState,
)
from src.components.graphrag.writer import (
    save_resume_graph,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


async def extract_graph_node(state: GraphIngestionState) -> dict:

    extracted_graph = await graph_extraction_chain.ainvoke(
        {
            "content": state["content"]
        }
    )

    logger.info(
        "Extracted %d entities and %d relationships from resume %d.",
        len(extracted_graph.entities),
        len(extracted_graph.relationships),
        state["resume_id"],
    )

    return {
        "extracted_graph": extracted_graph
    }

async def persist_graph_node(state: GraphIngestionState) -> dict:

    await save_resume_graph(
        user_id=state["user_id"],
        resume_id=state["resume_id"],
        filename=state["filename"],
        graph=state["extracted_graph"]
    )

    logger.info(
        "Resume %d knowledge graph saved to Neo4j.",
        state["resume_id"]
    )

    return {}


builder = StateGraph(GraphIngestionState)

builder.add_node("extract_graph", extract_graph_node)

builder.add_node("persist_graph", persist_graph_node)

builder.add_edge(START, "extract_graph")

builder.add_edge("extract_graph", "persist_graph")

builder.add_edge("persist_graph", END)

resume_graph_ingestion = builder.compile()