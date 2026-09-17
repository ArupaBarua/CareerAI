import json
from typing import Any, Literal, NotRequired, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.components.evaluation.response_evaluation import (
    revision_chain,
    support_evaluator_chain,
    usefulness_evaluator_chain,
)
from src.components.llm.model import llm
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


MAX_TOOL_ROUNDS = 2
MAX_REVISION_COUNT = 2
MAX_SEARCH_RETRIES = 1


job_search_tool = DuckDuckGoSearchResults(
    max_results=10,
    output_format="list",
)


class JobSearchState(TypedDict):

    query: str

    messages: list[BaseMessage]
    conversation_summary: NotRequired[str]
    long_term_memories: NotRequired[list[dict]]

    search_query: NotRequired[str]

    job_results: NotRequired[list[dict]]

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
    search_retry_count: NotRequired[int]
    search_failed: NotRequired[bool]


job_search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Job Search Agent of CareerAI.

CareerAI is an AI-powered career assistant that helps users with:

- job searching
- resume and CV analysis
- skill-gap analysis
- job recommendations
- career-related questions and guidance

Your task is to find current job opportunities that satisfy the user's query.

You have access to a web-search tool specifically for finding current
job opportunities.

For job-search requests, you must obtain current job information through
the provided search tool before providing job listings.

When you receive search results from a tool:

- inspect the returned results carefully,
- determine whether they provide enough relevant job information to
  answer the user's request,
- if the results are sufficient, produce the final answer using those
  results,
- if the results are insufficient, request another tool call using a
  refined search query,
- do not request another search unnecessarily.

Do not rely on your internal knowledge for current job openings.

Use the user's latest request as the primary source of search criteria.

Conversation context and relevant long-term memories may help resolve
information the user does not repeat, such as:

- target job roles
- preferred locations
- remote, hybrid, or onsite preferences
- industries of interest
- persistent career preferences

The latest user request always overrides older conversation context
or long-term memories when they conflict.

When searching, formulate focused search queries suitable for finding
actual job postings.

Prefer direct job postings and reputable company career pages when available.

Important rules:

- Do not invent job openings
- Do not invent companies
- Do not invent locations
- Do not invent salaries
- Do not invent requirements
- Do not invent employment conditions
- Do not invent application links
- Include only factual job information supported by search results
- Preserve links from search results accurately
- If a detail is unavailable, say that it was not available rather
  than guessing
- If the first search is insufficient, you may perform another search
- Once sufficient evidence has been retrieved, answer the user's
  request directly
""",
        ),
        (
            "human",
            """
Original user query:
{query}

Current search focus:
{search_query}

Conversation summary:
{conversation_summary}

Recent conversation:
{recent_messages}

Relevant long-term memories:
{long_term_memories}
""",
        ),
    ]
)


job_search_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the job-search query rewriting component of CareerAI.

The previous job-search attempt did not produce a sufficiently useful
answer.

Rewrite the search focus so that a web-search agent is more likely to
find relevant current job postings.

Preserve the user's original intent and constraints.

Use any useful information from the previous response to understand
what may have been missing.

Do not answer the user's question.
Do not invent job information.
Return only the rewritten search query.
""",
        ),
        (
            "human",
            """
Original user query:
{query}

Previous search focus:
{search_query}

Previous response:
{response}
""",
        ),
    ]
)

job_search_rewrite_chain = job_search_rewrite_prompt | llm


def format_messages(
    messages: list[BaseMessage],
) -> str:

    if not messages:
        return ""

    return "\n".join(
        f"{message.type}: {message.content}"
        for message in messages
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
    )


def serialize_tool_result(
    result: Any,
) -> str:

    if isinstance(result, str):
        return result

    return json.dumps(
        result,
        ensure_ascii=False,
        default=str,
        indent=2,
    )


def extract_job_results(
    result: Any,
) -> list[dict]:

    if not isinstance(result, list):
        return []

    return [
        item
        for item in result
        if isinstance(item, dict)
    ]


def deduplicate_job_results(
    results: list[dict],
) -> list[dict]:

    seen: set[str] = set()
    unique_results: list[dict] = []

    for result in results:

        identifier = (
            result.get("link")
            or result.get("url")
            or json.dumps(
                result,
                sort_keys=True,
                default=str,
            )
        )

        if identifier in seen:
            continue

        seen.add(identifier)
        unique_results.append(result)

    return unique_results



async def search_jobs_node(
    state: JobSearchState,
) -> dict:

    search_query = state.get(
        "search_query",
        state["query"],
    )

    messages = job_search_prompt.format_messages(
        query=state["query"],
        search_query=search_query,
        conversation_summary=state.get(
            "conversation_summary",
            "",
        ),
        recent_messages=format_messages(
            state.get("messages",[])
        ),
        long_term_memories=format_memories(
            state.get(
                "long_term_memories",
                [],
            )
        ),
    )

    
    search_required_llm = llm.bind_tools(
        [job_search_tool],
        tool_choice="required",
    )

    
    agent_llm = llm.bind_tools([job_search_tool])

    job_results: list[dict] = []

    try:

        for round_index in range(MAX_TOOL_ROUNDS):

            if round_index == 0:
                response = await search_required_llm.ainvoke(messages)

            else:
                response = await agent_llm.ainvoke(messages)

            tool_calls = response.tool_calls

            # No tool call
            # the LLM has generated the final response.
            if not tool_calls:

                return {
                    "job_results": (
                        deduplicate_job_results(job_results)
                    ),
                    "final_response": response.content,
                    "search_failed": False,
                }

            messages.append(response)

            for tool_call in tool_calls:

                if (tool_call["name"] != job_search_tool.name):

                    logger.warning(
                        "Unsupported job-search tool requested: %s",
                        tool_call["name"],
                    )

                    messages.append(
                        ToolMessage(
                            content=(
                                "The requested tool is not available."
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )

                    continue

                tool_result = await job_search_tool.ainvoke(tool_call["args"])
                
                tool_result_text = serialize_tool_result(tool_result)

                job_results.extend(
                    extract_job_results(tool_result)
                )

                # Send the search results back to the LLM.
                messages.append(
                    ToolMessage(
                        content=tool_result_text,
                        tool_call_id=tool_call["id"],
                        name=job_search_tool.name,
                    )
                )

                logger.info("Supported tool call successful.")

        final_response = await llm.ainvoke(messages)

        return {
            "job_results": (
                deduplicate_job_results(
                    job_results
                )
            ),
            "final_response": final_response.content,
            "search_failed": False,
        }

    except Exception:

        logger.exception(
            "DuckDuckGo job search failed."
        )

        return {
            "job_results": [],
            "final_response": (
                "I couldn't access the job-search service, "
                "so I can't reliably provide current job "
                "openings right now."
            ),
            "search_failed": True,
        }


async def evaluate_support_node(
    state: JobSearchState,
) -> dict:

    evaluation = (
        await support_evaluator_chain.ainvoke(
            {
                "query": state["query"],
                "text_context": "",
                "graph_context": "",
                "job_context": json.dumps(
                    state.get("job_results",[]),
                    ensure_ascii=False,
                    indent=2,
                ),
                "response": state["final_response"],
            }
        )
    )

    return {
        "support_status": evaluation.status
    }


async def revise_response_node(
    state: JobSearchState,
) -> dict:

    response = await revision_chain.ainvoke(
        {
            "query": state["query"],
            "text_context": "",
            "graph_context": "",
            "job_context": json.dumps(
                state.get("job_results",[]),
                ensure_ascii=False,
                indent=2,
            ),
            "response": state["final_response"],
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
    state: JobSearchState,
) -> dict:

    evaluation = (
        await usefulness_evaluator_chain.ainvoke(
            {
                "query": state["query"],
                "response": state["final_response"],
            }
        )
    )

    return {
        "usefulness_status": evaluation.status
    }


async def rewrite_search_query_node(
    state: JobSearchState,
) -> dict:

    response = await job_search_rewrite_chain.ainvoke(
        {
            "query": state["query"],
            "search_query": state.get(
                "search_query",
                state["query"],
            ),
            "response": state[
                "final_response"
            ],
        }
    )

    return {
        "search_query": (
            response.content.strip()
        ),

        "search_retry_count": (
            state.get(
                "search_retry_count",
                0,
            )
            + 1
        ),

        "job_results": [],
        "final_response": "",
        "support_status": None,
        "usefulness_status": None,
        "revision_count": 0,
        "search_failed": False,
    }


def route_after_search(
    state: JobSearchState,
) -> str:

    if state["search_failed"]:
        return "end"

    return "evaluate_support"


def route_support(
    state: JobSearchState,
) -> str:

    if (state["support_status"] == "supported"):
        return "evaluate_usefulness"

    if (state.get("revision_count", 0) >= MAX_REVISION_COUNT):

        return "evaluate_usefulness"

    return "revise_response"


def route_usefulness(
    state: JobSearchState,
) -> str:

    if (state["usefulness_status"] == "useful"):
        return "end"

    if (state.get("search_retry_count", 0) >= MAX_SEARCH_RETRIES):
        return "end"

    return "rewrite_search_query"


builder = StateGraph(JobSearchState)


builder.add_node(
    "search_jobs",
    search_jobs_node,
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
    "rewrite_search_query",
    rewrite_search_query_node,
)


builder.add_edge(
    START,
    "search_jobs",
)

builder.add_conditional_edges(
    "search_jobs",
    route_after_search,
    {
        "evaluate_support": "evaluate_support",
        "end": END,
    },
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
        "rewrite_search_query": "rewrite_search_query",
        "end": END,
    },
)

builder.add_edge(
    "rewrite_search_query",
    "search_jobs",
)


job_search_agent = builder.compile()