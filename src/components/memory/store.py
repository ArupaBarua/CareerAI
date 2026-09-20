from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_openai import OpenAIEmbeddings
from langgraph.store.postgres.aio import (
    AsyncPostgresStore,
)
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.constants.settings import settings


memory_embeddings = OpenAIEmbeddings(
    model=settings.MEMORY_EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY,
)


@asynccontextmanager
async def get_memory_store() -> AsyncIterator[
    AsyncPostgresStore
]:

    postgres_url = settings.DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
    )

    async with AsyncConnectionPool(
        conninfo=postgres_url,
        min_size=0,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        check=AsyncConnectionPool.check_connection,
    ) as pool:

        store = AsyncPostgresStore(
            pool,
            index={
                "dims": 1536,
                "embed": memory_embeddings,
                "fields": ["content"],
            },
        )

        await store.setup()

        yield store