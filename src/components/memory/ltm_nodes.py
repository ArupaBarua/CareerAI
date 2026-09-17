from langchain_core.messages import HumanMessage
from langgraph.config import get_store

from src.components.graph.state import CareerAIState
from src.components.memory.service import (
    search_relevant_user_memories,
    update_long_term_memory,
)


def get_latest_user_message(
    state: CareerAIState,
) -> str | None:

    for message in reversed(
        state["messages"]
    ):

        if isinstance(
            message,
            HumanMessage,
        ):
            return str(message.content)

    return None


async def update_long_term_memory_node(
    state: CareerAIState,
) -> dict:

    latest_user_message = get_latest_user_message(state)

    if latest_user_message is None:
        return {}

    store = get_store()

    await update_long_term_memory(
        store=store,
        user_id=state["user_id"],
        message=latest_user_message,
    )

    return {}


async def load_relevant_memories_node(
    state: CareerAIState,
) -> dict:

    latest_user_message = get_latest_user_message(state)

    if latest_user_message is None:
        return {
            "long_term_memories": []
        }

    store = get_store()

    memories = await search_relevant_user_memories(
        store=store,
        user_id=state["user_id"],
        query=latest_user_message,
        limit=5,
    )

    return {
        "long_term_memories": memories
    }