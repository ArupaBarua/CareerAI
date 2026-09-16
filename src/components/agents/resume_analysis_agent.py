from typing import Literal, NotRequired, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.components.evaluation.response_evaluation import (
    query_rewrite_chain,
    revision_chain,
    support_evaluator_chain,
    usefulness_evaluator_chain,
)
from src.components.graph.candidate_context_graph import candidate_context_graph
from src.components.llm.model import llm
from src.components.mcp.client import get_exa_search_tools
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


MAX_TOOL_ROUNDS = 2


# State
class ResumeAnalysisState(TypedDict):

    query: str

    intent: Literal[
        "resume_analysis",
        "skill_gap",
    ]

    user_id: int
    resume_id: int

    retrieval_query: NotRequired[str]

    # Candidate evidence
    text_context: NotRequired[str]
    graph_context: NotRequired[str]

    job_context: NotRequired[str]

    final_response: NotRequired[str]

    support_status: NotRequired[
        Literal[
            "supported",
            "unsupported",
        ]
        | None
    ]

    usefulness_status: NotRequired[
        Literal[
            "useful",
            "not_useful",
        ]
        | None
    ]

    revision_count: NotRequired[int]
    rewrite_count: NotRequired[int]


# Prompt
resume_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Resume Analysis Agent of CareerAI.

CareerAI is an AI-powered career assistant that helps users with:

- resume and CV analysis
- skill-gap analysis
- job recommendations based on a user's background
- job searching
- career-related questions and guidance

You handle two intents:

1. resume_analysis

Analyze the candidate's supplied context and answer questions about
their skills, experience, projects, education, research, achievements,
roles, and other resume-related information.

2. skill_gap

Compare the candidate's supplied context with a target job or role.

For skill-gap analysis:

- determine the relevant skills and qualifications required for the
  requested job or role using the user's query, supplied job context,
  or external tools when necessary,
- identify which of those skills are clearly demonstrated by the
  candidate's context,
- identify skills that are only partially demonstrated or supported
  by related experience,
- identify required skills for which the supplied candidate context
  provides no support,
- clearly distinguish demonstrated skills from skill gaps.

Candidate context consists only of:

- resume text context,
- candidate knowledge-graph context.

Never use external search results to infer facts about the candidate.

External tools may be used when current, specific, or otherwise
unavailable information about a target job, role, company, or job
market is necessary.

Use an external tool only when the information is not already
sufficiently available in the user's query or supplied job context.

Never use external tools to fill missing candidate information.

Important rules:

- Follow the supplied intent.
- Candidate-related factual claims must be supported by candidate context.
- Job- or role-related factual claims must be supported by the user's
  query, supplied job context, or information obtained through an external tool.
- Do not invent candidate skills, experience, qualifications,
  education, achievements, projects, organizations, roles or technologies.
- Do not invent job requirements.
- If the available evidence is insufficient, say so clearly.
""",
        ),
        (
            "human",
            """
Intent:
{intent}

User query:
{query}

Candidate resume context:
{text_context}

Candidate knowledge-graph context:
{graph_context}

Existing target job or role context:
{job_context}
""",
        ),
    ]
)

# Nodes
async def retrieve_candidate_context_node(state: ResumeAnalysisState) -> dict:

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
    }


async def generate_response_node(
    state: ResumeAnalysisState,
) -> dict:

    search_tool = await get_exa_search_tools()

    if search_tool is not None:
        agent_llm = llm.bind_tools(
            [search_tool]
        )
    else:
        logger.warning(
            "Exa MCP search tool unavailable. "
            "Continuing without external search."
        )
        agent_llm = llm

    job_context = state.get(
        "job_context",
        "",
    )

    messages = resume_analysis_prompt.format_messages(
        intent=state["intent"],
        query=state["query"],
        text_context=state.get(
            "text_context",
            "",
        ),
        graph_context=state.get(
            "graph_context",
            "",
        ),
        job_context=job_context,
    )

    for _ in range(MAX_TOOL_ROUNDS):

        response = await agent_llm.ainvoke(messages)

        tool_calls = response.tool_calls

        # LLM generated the final answer.
        if not tool_calls:
            return {
                "final_response": response.content,
                "job_context": job_context
            }

        # Add the LLM's tool-call message.
        messages.append(response)

        for tool_call in tool_calls:

            if tool_call["name"] != search_tool.name:
                logger.warning(
                    "Unsupported tool requested: %s",
                    tool_call["name"],
                )
                continue

            tool_result = await search_tool.ainvoke(tool_call["args"])

            tool_result_text = str(tool_result)

            if job_context:
                job_context += (
                    "\n\n"
                    + tool_result_text
                )
            else:
                job_context = tool_result_text

            # Give the tool result back to the LLM.
            messages.append(
                ToolMessage(
                    content=tool_result_text,
                    tool_call_id=tool_call["id"],
                )
            )

    # Maximum tool rounds reached.
    # Calling the LLM without tools so it must now produce a final response using the accumulated tool results.
    final_response = await llm.ainvoke(messages)

    return {
        "final_response": final_response.content,
        "job_context": job_context
    }


async def evaluate_support_node(state: ResumeAnalysisState) -> dict:

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
            "job_context": state.get(
                "job_context",
                "",
            ),
            "response": state["final_response"]
        }
    )

    return {
        "support_status": evaluation.status
    }


async def revise_response_node(state: ResumeAnalysisState) -> dict:

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
            "job_context": state.get(
                "job_context",
                "",
            ),
            "response": state["final_response"]
        }
    )

    return {
        "final_response": response.content,
        "revision_count": (
            state.get(
                "revision_count",
                0,
            )
            + 1
        ),
    }


async def evaluate_usefulness_node(state: ResumeAnalysisState) -> dict:

    evaluation = await usefulness_evaluator_chain.ainvoke(
        {
            "query": state["query"],
            "response": state["final_response"],
        }
    )

    return {
        "usefulness_status": evaluation.status
    }


async def rewrite_query_node(state: ResumeAnalysisState) -> dict:

    response = await query_rewrite_chain.ainvoke(
        {
            "query": state["query"],
            "response": state["final_response"],
        }
    )

    return {
        "retrieval_query": response.content.strip(),

        "rewrite_count": (
            state.get(
                "rewrite_count",
                0,
            )
            + 1
        ),

        "text_context": "",
        "graph_context": "",
        "final_response": "",
        "support_status": None,
        "usefulness_status": None,
        "revision_count": 0,
    }

# Routing
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


# Graph
builder = StateGraph(ResumeAnalysisState)

builder.add_node(
    "retrieve_candidate_context",
    retrieve_candidate_context_node
)

builder.add_node(
    "generate_response",
    generate_response_node
)

builder.add_node(
    "evaluate_support",
    evaluate_support_node
)

builder.add_node(
    "revise_response",
    revise_response_node
)

builder.add_node(
    "evaluate_usefulness",
    evaluate_usefulness_node
)

builder.add_node(
    "rewrite_query",
    rewrite_query_node
)


builder.add_edge(
    START,
    "retrieve_candidate_context"
)

builder.add_edge(
    "retrieve_candidate_context",
    "generate_response"
)

builder.add_edge(
    "generate_response",
    "evaluate_support"
)

builder.add_conditional_edges(
    "evaluate_support",
    route_support,
    {
        "revise_response": "revise_response",
        "evaluate_usefulness": "evaluate_usefulness",
    }
)

builder.add_edge(
    "revise_response",
    "evaluate_support"
)

builder.add_conditional_edges(
    "evaluate_usefulness",
    route_usefulness,
    {
        "rewrite_query": "rewrite_query",
        "end": END,
    }
)

builder.add_edge(
    "rewrite_query",
    "retrieve_candidate_context"
)

resume_analysis_agent = builder.compile()