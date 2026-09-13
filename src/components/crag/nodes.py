from src.components.crag.knowledge_refiner import knowledge_refinement_graph
from src.components.crag.retrieval_evaluator import evaluate_retrieval
from src.components.crag.state import CRAGState
from src.components.crag.generator import generation_chain
from src.components.crag.response_evaluator import support_chain, usefulness_chain
from src.components.crag.response_revision import revision_chain
from src.components.crag.query_rewriter import query_rewrite_chain
from src.components.retrieval.resume_vector_store import retrieve_resume_chunks
from src.components.mcp.client import get_exa_search_tools
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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


async def web_search_node(state: CRAGState) -> dict:

    search_tool = await get_exa_search_tools()

    if search_tool is None:
        logger.warning("Web search unavailable. Continuing without external context.")
        return {
            "web_context": ""
        }

    result = await search_tool.ainvoke(
        {
            "query": state["query"],
            "numResults": 5
        }
    )

    return {
        "web_context": str(result)
    }


async def generate_node(state: CRAGState) -> dict:

    response = await generation_chain.ainvoke(
        {
            "query": state["query"],
            "refined_context": state.get(
                "refined_context",
                ""
            ),
            "web_context": state.get(
                "web_context",
                ""
            )
        }
    )

    return {"final_response": response.content}


async def evaluate_support_node(state: CRAGState) -> dict:

    evaluation = await support_chain.ainvoke(
        {
            "query": state["query"],
            "refined_context": state.get(
                "refined_context",
                ""
            ),
            "web_context": state.get(
                "web_context",
                ""
            ),
            "answer": state["final_response"]
        }
    )

    return {
        "support_status": evaluation.status
    }

async def evaluate_usefulness_node(state: CRAGState) -> dict:

    evaluation = await usefulness_chain.ainvoke(
        {
            "query": state["query"],
            "answer": state["final_response"]
        }
    )

    return {
        "usefulness_status": evaluation.status
    }

async def revise_answer_node(state: CRAGState) -> dict:

    response = await revision_chain.ainvoke(
        {
            "query": state["query"],
            "refined_context": state.get(
                "refined_context",
                ""
            ),
            "web_context": state.get(
                "web_context",
                ""
            ),
            "answer": state["final_response"]
        }
    )

    return {
        "final_response": response.content,
        "revision_count": (
            state.get("revision_count", 0) + 1
        )
    }


async def rewrite_query_node(state: CRAGState) -> dict:

    response = await query_rewrite_chain.ainvoke(
        {
            "query": state["query"]
        }
    )

    return {
        "query": response.content.strip(),
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "retrieved_documents": [],
        "retrieval_status": None,
        "refined_context": "",
        "web_context": "",
        "final_response": "",
        "support_status": None,
        "usefulness_status": None,
        "revision_count": 0,
    }