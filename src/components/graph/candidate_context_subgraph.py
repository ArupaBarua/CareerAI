from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.components.crag.graph import crag_graph
from src.components.graphrag.query_graph import graphrag_graph

class CandidateContextState(TypedDict):
    query: str
    user_id: int
    resume_id: int

    text_context: NotRequired[str]
    web_context: NotRequired[str]
    graph_context: NotRequired[str]


async def crag_context_node(
    state: CandidateContextState,
) -> dict:

    result = await crag_graph.ainvoke(
        {
            "query": state["query"],
            "user_id": state["user_id"],
            "resume_id": state["resume_id"],
        }
    )

    return {
        "text_context": result.get(
            "refined_context",
            "",
        ),
        "web_context": result.get(
            "web_context",
            "",
        ),
    }


async def graphrag_context_node(
    state: CandidateContextState,
) -> dict:

    result = await graphrag_graph.ainvoke(
        {
            "query": state["query"],
            "user_id": state["user_id"],
            "resume_id": state["resume_id"],
        }
    )

    return {
        "graph_context": result.get(
            "graph_context",
            "",
        )
    }

builder = StateGraph(CandidateContextState)

builder.add_node("crag_context", crag_context_node)

builder.add_node("graphrag_context", graphrag_context_node)

builder.add_edge(START, "crag_context")

builder.add_edge(START, "graphrag_context")

builder.add_edge(
    [
        "crag_evidence",
        "graphrag_evidence",
    ],
    END,
)

candidate_context_graph = builder.compile()