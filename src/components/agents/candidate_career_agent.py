import json
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.components.evaluation.response_evaluation import (
    query_rewrite_chain,
    revision_chain,
    support_evaluator_chain,
    usefulness_evaluator_chain,
)
from src.components.graph.candidate_context_graph import (
    candidate_context_graph,
)
from src.components.llm.model import llm
from src.components.mcp.client import get_exa_search_tools
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


MAX_TOOL_ROUNDS = 2
MAX_REVISION_COUNT = 2
MAX_REWRITE_COUNT = 2


# State
class CandidateCareerState(TypedDict):

    # Current user request
    query: str

    intent: Literal[
        "resume_analysis",
        "skill_gap",
        "job_recommendation",
    ]

    user_id: int
    resume_id: int

    # Active recent LangGraph messages.
    messages: NotRequired[list[BaseMessage]]

    # Rolling summary of older messages removed from active state.
    conversation_summary: NotRequired[str]

    # Semantically retrieved user-level long-term memories.
    long_term_memories: NotRequired[list[dict]]

    # May differ from the original query after a retrieval rewrite.
    retrieval_query: NotRequired[str]

    # Candidate context from CRAG.
    text_context: NotRequired[str]

    # Candidate context from GraphRAG.
    graph_context: NotRequired[str]


    # Context returned by Exa MCP when external information is needed.
    external_context: NotRequired[str]

    # Response evaluation
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
candidate_career_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Candidate Career Agent of CareerAI.

CareerAI is an AI-powered career assistant that helps users with
resume analysis, skill-gap analysis, job recommendations, job
searching, and general career guidance.

You handle three intents:

1. resume_analysis

Analyze the supplied candidate context and answer questions about
the candidate's:

- skills,
- experience,
- projects,
- education,
- research,
- publications,
- achievements,
- roles,
- technologies,
- and other resume-related information.

2. skill_gap

Compare the candidate context with a target job or career role.

For skill-gap analysis:

- determine the relevant skills and qualifications for the requested
  job or role,
- identify skills clearly demonstrated by candidate context,
- identify skills that are only partially demonstrated or supported
  through related experience,
- identify important requirements for which the available candidate
  context provides no support,
- clearly distinguish demonstrated skills from unsupported or
  partially demonstrated requirements.

Do not claim that the candidate definitely lacks a skill merely
because it does not appear in the supplied context.

Instead, state that the skill is not demonstrated by the available
candidate context.

3. job_recommendation

Recommend TYPES OF JOBS, CAREER ROLES, or JOB TITLES that fit the
candidate's demonstrated background.

Job recommendation is different from job search. 
You are recommending suitable kinds of roles.

For job recommendation:

- analyze the candidate's demonstrated skills, experience, projects,
  research, education, and technical background
- identify job roles that reasonably align with user's skills,
- explain why each recommended role is relevant,
- distinguish strong matches from roles that may require additional
  development,
- do not invent candidate qualifications.


--------------------------------------------------
EXTERNAL SEARCH TOOL
--------------------------------------------------

You have access to an external web-search tool.

Use the external search tool whenever external information is needed
to answer the user's request accurately or when external information
would materially improve the answer.

Examples include:

- current or specific role requirements,
- skills commonly associated with a career,
- identifying jobs or career fields that commonly use a particular
  combination of technologies,
- unfamiliar or niche roles,
- company-specific information,
- industry information,
- current career or job-market information,
- information needed to make a better-supported job recommendation.

You do NOT need to use the tool when the supplied context and your
existing knowledge are sufficient to answer accurately.

Do not search merely because candidate information is missing.

External search must NEVER be used to fill missing candidate facts.

When you receive external search results:

- inspect them carefully,
- use only information actually supported by those results,
- determine whether the context is sufficient,
- if another search would materially help, request another tool call
  with improved search arguments,
- do not perform unnecessary searches.


GROUNDING RULES

- Follow the supplied intent.
- Do not treat memory as resume context.
- If the latest user request conflicts with older conversation context
  or long-term memory, follow the latest user request.
- Candidate factual claims must be grounded in candidate context.
- Current, specific, or externally researched claims must be grounded
  in external context when external search is used.
- Do not invent skills, experience, qualifications, projects,
  education, publications, achievements, employers, organizations,
  dates, technologies, job requirements, or market facts.
- If the available context is insufficient, say so clearly.
""",
        ),
        (
            "human",
            """
Intent:
{intent}

Current user query:
{query}

Conversation summary:
{conversation_summary}

Recent conversation:
{recent_messages}

Relevant long-term memories:
{long_term_memories}

Candidate resume context:
{text_context}

Candidate knowledge-graph context:
{graph_context}

External context gathered so far:
{external_context}
""",
        ),
    ]
)


def format_messages(
    messages: list[BaseMessage],
) -> str:

    if not messages:
        return ""

    formatted_messages: list[str] = []

    for message in messages:

        formatted_messages.append(
            f"{message.type}: {message.content}"
        )

    return "\n".join(
        formatted_messages
    )


def format_memories(
    memories: list[dict],
) -> str:

    if not memories:
        return ""

    return json.dumps(
        memories,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def serialize_tool_result(
    result: Any,
) -> str:

    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def append_external_context(
    existing_context: str,
    new_context: str,
) -> str:

    if not existing_context:
        return new_context

    return (
        existing_context
        + "\n\n"
        + new_context
    )


# Candidate-context retrieval

async def retrieve_candidate_context_node(
    state: CandidateCareerState,
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
    }


# Response generation + Exa tool loop

async def generate_response_node(
    state: CandidateCareerState,
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

    external_context = state.get(
        "external_context",
        "",
    )

    messages = candidate_career_prompt.format_messages(
        intent=state["intent"],

        query=state["query"],

        conversation_summary=state.get(
            "conversation_summary",
            "",
        ),

        recent_messages=format_messages(
            state.get(
                "messages",
                [],
            )
        ),

        long_term_memories=format_memories(
            state.get(
                "long_term_memories",
                [],
            )
        ),

        text_context=state.get(
            "text_context",
            "",
        ),

        graph_context=state.get(
            "graph_context",
            "",
        ),

        external_context=external_context,
    )

    # Tool loop

    for _ in range(MAX_TOOL_ROUNDS):

        response = await agent_llm.ainvoke(messages)

        tool_calls = response.tool_calls

        # No tool call:
        # the model has produced its final answer.
        
        if not tool_calls:

            return {
                "final_response": response.content,
                "external_context": external_context,
            }

        messages.append(response)

        for tool_call in tool_calls:

            if (tool_call["name"] != search_tool.name):

                logger.warning(
                    "Unsupported tool requested: %s",
                    tool_call["name"],
                )

                messages.append(
                    ToolMessage(
                        content=(
                            "The requested external tool is unavailable."
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )

                continue

            try:

                tool_result = await search_tool.ainvoke(
                    tool_call["args"]
                )

            except Exception:

                logger.exception(
                    "Exa MCP search failed."
                )

                messages.append(
                    ToolMessage(
                        content=(
                            "The external search failed and returned no usable context."
                        ),
                        tool_call_id=tool_call["id"],
                        name=search_tool.name,
                    )
                )

                continue

            tool_result_text = serialize_tool_result(tool_result)

            external_context = append_external_context(external_context, tool_result_text)

            messages.append(
                ToolMessage(
                    content=tool_result_text,
                    tool_call_id=tool_call["id"],
                    name=search_tool.name,
                )
            )

    # Maximum tool rounds reached.
    # Calling the plain LLM without tools. 

    final_response = await llm.ainvoke(messages)

    return {
        "final_response": final_response.content,
        "external_context": external_context,
    }

# Evaluation

async def evaluate_support_node(
    state: CandidateCareerState,
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

            "job_context": state.get(
                "external_context",
                "",
            ),

            "response": state[
                "final_response"
            ],
        }
    )

    return {
        "support_status": evaluation.status
    }


async def revise_response_node(
    state: CandidateCareerState,
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

            "job_context": state.get(
                "external_context",
                "",
            ),

            "response": state[
                "final_response"
            ],
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

async def evaluate_usefulness_node(
    state: CandidateCareerState,
) -> dict:

    evaluation = (
        await usefulness_evaluator_chain.ainvoke(
            {
                "query": state["query"],

                "response": state[
                    "final_response"
                ],
            }
        )
    )

    return {
        "usefulness_status": evaluation.status
    }


async def rewrite_candidate_query_node(
    state: CandidateCareerState,
) -> dict:

    response = await query_rewrite_chain.ainvoke(
        {
            "query": state["query"],

            "response": state[
                "final_response"
            ],
        }
    )

    return {
        "retrieval_query": (
            response.content.strip()
        ),

        "rewrite_count": (
            state.get(
                "rewrite_count",
                0,
            )
            + 1
        ),

        # Candidate context will be retrieved again.
        "text_context": "",
        "graph_context": "",

        "final_response": "",

        "support_status": None,
        "usefulness_status": None,

        "revision_count": 0,
    }

# Routing

def route_support(
    state: CandidateCareerState,
) -> str:

    if (state["support_status"]== "supported"):
        return "evaluate_usefulness"

    if (state.get("revision_count", 0) >= MAX_REVISION_COUNT):
        return "evaluate_usefulness"

    return "revise_response"


def route_usefulness(
    state: CandidateCareerState,
) -> str:

    if (state["usefulness_status"] == "useful"):
        return "end"

    if (state.get("rewrite_count", 0) >= MAX_REWRITE_COUNT):
        return "end"

    return "rewrite_candidate_query"


# Graph

builder = StateGraph(CandidateCareerState)


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
    "rewrite_candidate_query",
    rewrite_candidate_query_node,
)

# Edges

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
        "rewrite_candidate_query": "rewrite_candidate_query",
        "end": END,
    },
)


builder.add_edge(
    "rewrite_candidate_query",
    "retrieve_candidate_context",
)

candidate_career_agent = builder.compile()