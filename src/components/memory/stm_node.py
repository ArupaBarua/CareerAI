from langchain_core.messages import RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.components.graph.state import CareerAIState
from src.components.llm.model import llm
from src.components.memory.stm_service import (
    format_messages_for_summary,
    should_summarize,
    split_messages_for_summary,
)


conversation_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the conversation-memory summarizer for CareerAI.

CareerAI is an AI-powered career assistant.

Your task is to create or update a concise running summary of the user's conversation.

You may receive:
1. an existing conversation summary from earlier summarized messages
2. a new group of older messages that are now being removed from the active conversation.

If no existing summary is provided, create the first conversation summary from the supplied messages.

If an existing summary is provided, preserve its important information and update it with the newly supplied messages.

Preserve information that may be important for understanding future conversation, including:

- the user's goals and intentions,
- questions and requests previously discussed,
- important decisions,
- relevant preferences,
- important facts established during the conversation,
- conclusions or recommendations that later messages may refer to,
- unresolved topics,
- references whose meaning may matter later.

Rules:

- Do not invent information.
- Do not include information that is not supported by the conversation.
- Do not mention that messages were summarized or removed.
- Keep the summary concise but sufficiently informative.
- Return only the conversation summary.
"""
        ),
        (
            "human",
            """
Existing conversation summary:
{existing_summary}

New older messages to incorporate:
{messages}
""",
        ),
    ]
)


conversation_summary_chain = conversation_summary_prompt | llm


async def summarize_conversation_node(
    state: CareerAIState,
) -> dict:

    messages = state["messages"]

    if not should_summarize(messages):
        return {}

    messages_to_summarize, recent_messages = split_messages_for_summary(messages)

    formatted_messages = format_messages_for_summary(messages_to_summarize)

    response = await conversation_summary_chain.ainvoke(
        {
            "existing_summary": state.get(
                "conversation_summary",
                ""
            ),
            "messages": formatted_messages,
        }
    )

    updated_summary = response.content.strip()

    return {
        "conversation_summary": updated_summary,

        "messages": [
            RemoveMessage(
                id=REMOVE_ALL_MESSAGES
            ),
            *recent_messages,
        ]
    }