from uuid import uuid4
import json
from langgraph.store.base import BaseStore

from src.components.memory.ltm_extractor import MemoryExtractionResult, memory_extractor_chain
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_user_memory_namespace(
    user_id: int,
) -> tuple[str, str, str]:

    return (
        "users",
        str(user_id),
        "memories",
    )


async def save_user_memory(
    store: BaseStore,
    user_id: int,
    memory_type: str,
    content: str,
) -> str:

    memory_id = str(uuid4())

    namespace = get_user_memory_namespace(user_id)

    await store.aput(
        namespace,
        memory_id,
        {
            "type": memory_type,
            "content": content,
        },
    )

    return memory_id


async def update_user_memory(
    store: BaseStore,
    user_id: int,
    memory_id: str,
    memory_type: str,
    content: str
) -> None:

    namespace = get_user_memory_namespace(user_id)

    await store.aput(
        namespace,
        memory_id,
        {
            "type": memory_type,
            "content": content
        }
    )


async def get_user_memories(
    store: BaseStore,
    user_id: int,
    limit: int = 50,
) -> list[dict]:

    namespace = get_user_memory_namespace(user_id)

    memories = await store.asearch(
        namespace,
        limit=limit,
    )

    return [
        {
            "id": memory.key,
            **memory.value,
        }
        for memory in memories
    ]

async def search_relevant_user_memories(
    store: BaseStore,
    user_id: int,
    query: str,
    limit: int = 5,
) -> list[dict]:

    namespace = get_user_memory_namespace(user_id)

    memories = await store.asearch(
        namespace,
        query=query,
        limit=limit,
    )

    return [
        {
            "id": memory.key,
            **memory.value,
        }
        for memory in memories
    ]

async def get_user_memory(
    store: BaseStore,
    user_id: int,
    memory_id: str
) -> dict | None:

    namespace = get_user_memory_namespace(user_id)

    memory = await  store.aget(
        namespace, 
        memory_id
    )

    if memory is None:
        return None
    
    return {
        "id": memory.key,
        **memory.value
    }


async def delete_user_memory(
    store: BaseStore,
    user_id: int,
    memory_id: str,
) -> None:

    namespace = get_user_memory_namespace(user_id)

    await store.adelete(
        namespace,
        memory_id,
    )


async def extract_memory_operations(
    store: BaseStore,
    user_id: int,
    message: str
) -> MemoryExtractionResult:

    existing_memories = await get_user_memories(
        store=store,
        user_id=user_id
    )

    result = await memory_extractor_chain.ainvoke(
        {
            "memories": json.dumps(
                existing_memories,
                ensure_ascii=False,
                indent=2
            ),
            "message": message
        }
    )

    return result


async def update_long_term_memory(
    store: BaseStore,
    user_id: int,
    message: str,
) -> None:

    result = await extract_memory_operations(
        store=store,
        user_id=user_id,
        message=message,
    )

    for operation in result.operations:

        if operation.action == "add":

            if (
                operation.memory_type is None
                or operation.content is None
            ):
                continue

            await save_user_memory(
                store=store,
                user_id=user_id,
                memory_type=operation.memory_type,
                content=operation.content,
            )

            logger.info("Added new long-term memory.")


        elif operation.action == "update":

            if (
                operation.memory_id is None
                or operation.memory_type is None
                or operation.content is None
            ):
                continue

            existing_memory = await get_user_memory(
                store=store,
                user_id=user_id,
                memory_id=operation.memory_id,
            )

            if existing_memory is None:
                continue

            await update_user_memory(
                store=store,
                user_id=user_id,
                memory_id=operation.memory_id,
                memory_type=operation.memory_type,
                content=operation.content,
            )

            logger.info("Updated existing long term memory.")


        elif operation.action == "delete":

            if operation.memory_id is None:
                continue

            existing_memory = await get_user_memory(
                store=store,
                user_id=user_id,
                memory_id=operation.memory_id,
            )

            if existing_memory is None:
                continue

            await delete_user_memory(
                store=store,
                user_id=user_id,
                memory_id=operation.memory_id,
            )

            logger.info("Deleted an existing memory.")