from typing import Literal, NotRequired, TypedDict
from langchain_core.prompts import ChatPromptTemplate

from src.components.graph.candidate_context_graph import candidate_context_graph
from src.components.evaluation.response_evaluation import (
    query_rewrite_chain,
    revision_chain,
    support_evaluator_chain,
    usefulness_evaluator_chain,
)
from src.components.llm.model import llm
from langgraph.graph import END, START, StateGraph


#GraphState
class ResumeAnalysisState(TypedDict):
    query: str
    user_id: int
    resume_id: int

    retrieval_query: NotRequired[str]

    text_context: NotRequired[str]
    graph_context: NotRequired[str]
    web_context: NotRequired[str]

    final_response: NotRequired[str]

    support_status: NotRequired[
        Literal["supported", "unsupported"] | None
    ]

    usefulness_status: NotRequired[
        Literal["useful", "not_useful"] | None
    ]

    revision_count: NotRequired[int]
    rewrite_count: NotRequired[int]


#Nodes
resume_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Resume Analysis Agent of CareerAI.

CareerAI is an AI-powered career assistant chatbot designed to help users
with job searching, resume analysis, job-role recommendations, skill-gap
analysis, and general career-related questions.

Answer the user's question by analyzing the supplied candidate context.

Use only the supplied context for factual claims about the candidate.

The context may include:
- relevant resume text,
- knowledge-graph relationships,
- web context when retrieval correction required it.

Do not invent skills, experience, projects, education,
achievements, roles, technologies, organizations, or dates.

If the available context is insufficient, say so clearly.
"""
        ),
        (
            "human",
            """
User query:
{query}

Resume text context:
{text_context}

Knowledge graph context:
{graph_context}

Web context:
{web_context}
""",
        ),
    ]
)


resume_analysis_chain = resume_analysis_prompt | llm


async def retrieve_candidate_context_node(
    state: ResumeAnalysisState,
) -> dict:

    retrieval_query = state.get(
        "retrieval_query",
        state["query"],
    )

    result = await candidate_context_graph.ainvoke(
        {
            "query": retrieval_query,
            "user_id": state["user_id"],
            "resume_id": state["resume_id"],
        }
    )

    return {
        "text_context": result.get(
            "text_context",
            "",
        ),
        "graph_context": result.get(
            "graph_context",
            "",
        ),
        "web_context": result.get(
            "web_context",
            "",
        ),
    }


async def generate_response_node(
    state: ResumeAnalysisState,
) -> dict:

    response = await resume_analysis_chain.ainvoke(
        {
            "query": state["query"],
            "text_context": state.get(
                "text_context",
                "",
            ),
            "graph_context": state.get(
                "graph_context",
                "",
            ),
            "web_context": state.get(
                "web_context",
                "",
            ),
        }
    )

    return {
        "final_response": response.content
    }


async def evaluate_support_node(
    state: ResumeAnalysisState,
) -> dict:

    evaluation = await support_evaluator_chain.ainvoke(
        {
            "query": state["query"],
            "text_context": state.get(
                "text_context",
                "",
            ),
            "graph_context": state.get(
                "graph_context",
                "",
            ),
            "web_context": state.get(
                "web_context",
                "",
            ),
            "response": state["final_response"],
        }
    )

    return {
        "support_status": evaluation.status
    }


async def revise_response_node(
    state: ResumeAnalysisState,
) -> dict:

    response = await revision_chain.ainvoke(
        {
            "query": state["query"],
            "text_context": state.get(
                "text_context",
                "",
            ),
            "graph_context": state.get(
                "graph_context",
                "",
            ),
            "web_context": state.get(
                "web_context",
                "",
            ),
            "response": state["final_response"],
        }
    )

    return {
        "final_response": response.content,
        "revision_count": (
            state.get("revision_count", 0) + 1
        ),
    }


async def evaluate_usefulness_node(
    state: ResumeAnalysisState,
) -> dict:

    evaluation = await usefulness_evaluator_chain.ainvoke(
        {
            "query": state["query"],
            "response": state["final_response"],
        }
    )

    return {
        "usefulness_status": evaluation.status
    }


async def rewrite_query_node(
    state: ResumeAnalysisState,
) -> dict:

    response = await query_rewrite_chain.ainvoke(
        {
            "query": state["query"],
            "response": state["final_response"],
        }
    )

    return {
        "retrieval_query": response.content.strip(),
        "rewrite_count": (
            state.get("rewrite_count", 0) + 1
        ),

        # New retrieval cycle
        "text_context": "",
        "graph_context": "",
        "web_context": "",
        "final_response": "",
        "support_status": None,
        "usefulness_status": None,
        "revision_count": 0,
    }


#graph
def route_support(state: ResumeAnalysisState) -> str:

    if state["support_status"] == "supported":
        return "evaluate_usefulness"

    if state.get("revision_count", 0) >= 2:
        return "evaluate_usefulness"

    return "revise_response"


def route_usefulness(
    state: ResumeAnalysisState,
) -> str:

    if state["usefulness_status"] == "useful":
        return "end"

    if state.get("rewrite_count", 0) >= 2:
        return "end"

    return "rewrite_query"


builder = StateGraph(ResumeAnalysisState)

builder.add_node(
    "retrieve_candidate_context",
    retrieve_candidate_context_node,
)

builder.add_node(
    "generate_response",
    generate_response_node,
)

builder.add_node(
    "evaluate_support",
    evaluate_support_node,
)

builder.add_node(
    "revise_response",
    revise_response_node,
)

builder.add_node(
    "evaluate_usefulness",
    evaluate_usefulness_node,
)

builder.add_node(
    "rewrite_query",
    rewrite_query_node,
)


builder.add_edge(
    START,
    "retrieve_candidate_context",
)

builder.add_edge(
    "retrieve_candidate_context",
    "generate_response",
)

builder.add_edge(
    "generate_response",
    "evaluate_support",
)

builder.add_conditional_edges(
    "evaluate_support",
    route_support,
    {
        "revise_response": "revise_response",
        "evaluate_usefulness": "evaluate_usefulness",
    },
)

builder.add_edge(
    "revise_response",
    "evaluate_support",
)

builder.add_conditional_edges(
    "evaluate_usefulness",
    route_usefulness,
    {
        "rewrite_query": "rewrite_query",
        "end": END,
    },
)

builder.add_edge(
    "rewrite_query",
    "retrieve_candidate_context",
)


resume_analysis_agent = builder.compile()
