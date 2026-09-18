import json
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.components.evaluation.response_evaluation import (
    usefulness_evaluator_chain,
)
from src.components.llm.model import llm
from src.components.mcp.client import get_exa_search_tools
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


MAX_TOOL_ROUNDS = 2
MAX_REGENERATION_COUNT = 2


class GeneralAgentState(TypedDict):

    query: str

    # Conversation / memory context
    messages: NotRequired[list[BaseMessage]]
    conversation_summary: NotRequired[str]
    long_term_memories: NotRequired[list[dict]]

    # External evidence obtained through Exa when needed
    external_context: NotRequired[str]

    final_response: NotRequired[str]

    usefulness_status: NotRequired[
        Literal[
            "useful",
            "not_useful",
        ]
        | None
    ]

    regeneration_count: NotRequired[int]


general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the General Career Agent of CareerAI.

CareerAI is an AI-powered career assistant that helps users with:

- job searching
- resume and CV analysis
- skill-gap analysis
- job recommendations
- career-related questions and guidance

You handle career-related questions or any general question that do not require
candidate resume analysis or current job-posting search.

Examples include:

- career advice,
- interview preparation,
- learning roadmaps,
- explanations of career roles,
- career-related technical concepts,
- professional development,
- workplace and career guidance,
- comparisons between career paths,
- general questions about skills and technologies.

You have access to an external web-search tool.

Use external search whenever current, specific, unfamiliar, or
externally verifiable information is needed to answer the user's
request accurately or when external information would materially
improve the answer.

You do NOT need to use external search when your existing knowledge
and the supplied conversation context are sufficient.

When you receive external search results:

- inspect the results carefully,
- request another search only when it would materially improve the
  answer,
- do not perform unnecessary searches.

Conversation summary, recent messages, and long-term memories may be
used to understand the user's:

- current question,
- preferences,
- career interests,
- goals,
- constraints,
- and previously discussed context.

The latest user request has priority if it conflicts with older
conversation context or long-term memories.

Important rules:

- Answer the user's actual question directly.
- Do not invent current facts.
- Do not invent personal facts about the user.
- Clearly distinguish general guidance from current externally
  researched information when relevant.
- If information is uncertain or unavailable, say so.
""",
        ),
        (
            "human",
            """
Current user query:
{query}

Conversation summary:
{conversation_summary}

Recent conversation:
{recent_messages}

Relevant long-term memories:
{long_term_memories}

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


async def generate_response_node(
    state: GeneralAgentState,
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

    messages = general_prompt.format_messages(
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

        external_context=external_context,
    )


    for _ in range(MAX_TOOL_ROUNDS):

        response = await agent_llm.ainvoke(messages)

        tool_calls = response.tool_calls

        # Final response produced.
        if not tool_calls:

            return {
                "final_response": response.content,
                "external_context": external_context,
            }

        # This AIMessage contains the tool call and must appear
        # before the ToolMessage containing its result.
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
                            "The external search failed and returned no usable evidence."
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

    final_response = await llm.ainvoke(messages)

    return {
        "final_response": final_response.content,
        "external_context": external_context,
    }


# Evaluation

async def evaluate_usefulness_node(
    state: GeneralAgentState,
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


async def regenerate_response_node(
    state: GeneralAgentState,
) -> dict:

    regeneration_count = (
        state.get(
            "regeneration_count",
            0,
        )
        + 1
    )

    return {
        "final_response": "",
        "usefulness_status": None,
        "regeneration_count": regeneration_count,
    }



# Routing

def route_usefulness(
    state: GeneralAgentState,
) -> str:

    if (state["usefulness_status"] == "useful"):
        return "end"

    if (state.get("regeneration_count", 0) >= MAX_REGENERATION_COUNT):
        return "end"

    return "regenerate_response"

# Graph

builder = StateGraph(GeneralAgentState)

builder.add_node(
    "generate_response",
    generate_response_node,
)

builder.add_node(
    "evaluate_usefulness",
    evaluate_usefulness_node,
)

builder.add_node(
    "regenerate_response",
    regenerate_response_node,
)


builder.add_edge(
    START,
    "generate_response",
)

builder.add_edge(
    "generate_response",
    "evaluate_usefulness",
)

builder.add_conditional_edges(
    "evaluate_usefulness",
    route_usefulness,
    {
        "regenerate_response": "regenerate_response",
        "end": END,
    },
)

builder.add_edge(
    "regenerate_response",
    "generate_response",
)


general_agent = builder.compile()