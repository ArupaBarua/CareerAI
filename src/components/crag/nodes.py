from src.components.crag.knowledge_refiner import knowledge_refinement_graph
from src.components.crag.retrieval_evaluator import evaluate_retrieval
from src.components.crag.state import CRAGState
from src.components.retrieval.resume_vector_store import retrieve_resume_chunks


async def retrieve_resume_node(state: CRAGState) -> dict:

    documents = await retrieve_resume_chunks(
        user_id=state["user_id"],
        resume_id=state["resume_id"],
        query=state["query"]
    )

    return {
        "retrieved_documents": documents
    }


async def evaluate_retrieval_node(state: CRAGState) -> dict:

    status = await evaluate_retrieval(
        query=state["query"],
        documents=state["retrieved_documents"]
    )

    return {
        "retrieval_status": status
    }


async def knowledge_refinement_node(state: CRAGState) -> dict:

    result = await knowledge_refinement_graph.ainvoke(
        {
            "query": state["query"],
            "documents": state["retrieved_documents"],
            "strips": [],
            "kept_strips": [],
            "refined_context": ""
        }
    )

    return {
        "refined_context": result["refined_context"]
    }