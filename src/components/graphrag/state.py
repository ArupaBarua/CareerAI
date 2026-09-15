from typing import NotRequired, TypedDict

from src.components.graphrag.extractor import ExtractedKnowledgeGraph
from src.components.graphrag.retriever import GraphFact

class GraphIngestionState(TypedDict):
    user_id: int
    resume_id: int
    filename: str
    content: str
    extracted_graph: NotRequired[ExtractedKnowledgeGraph]


class GraphRAGState(TypedDict):
    query: str
    user_id: int
    resume_id: int

    graph_facts: NotRequired[list[GraphFact]]
    graph_context: NotRequired[str]
    