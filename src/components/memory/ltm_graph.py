from langchain_core.messages import HumanMessage
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from src.components.graph.state import CareerAIState
from src.components.memory.ltm_service import (
    search_relevant_user_memories,
    update_long_term_memory,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

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

    logger.info("Updated long-term memories.")

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

    if memories is not None:
        logger.info("Relevant long-term memories retrieved.")

    return {
        "long_term_memories": memories
    }


builder = StateGraph(CareerAIState)


builder.add_node(
    "update_long_term_memory",
    update_long_term_memory_node,
)

builder.add_node(
    "load_relevant_memories",
    load_relevant_memories_node,
)


builder.add_edge(
    START,
    "update_long_term_memory",
)

builder.add_edge(
    "update_long_term_memory",
    "load_relevant_memories",
)

builder.add_edge(
    "load_relevant_memories",
    END,
)


ltm_graph = builder.compile()