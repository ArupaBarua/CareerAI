from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_openai import OpenAIEmbeddings

from src.constants.settings import settings

memory_embeddings = OpenAIEmbeddings(
    model=settings.MEMORY_EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY
)

@asynccontextmanager
async def get_memory_store() -> AsyncIterator[AsyncPostgresStore]:

    connection_string = settings.DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://"
    )

    async with AsyncPostgresStore.from_conn_string(
        connection_string,
        index={
            "dims": 1536,
            "embed": memory_embeddings,
            "fields": [
                "content"
            ]
        }
    ) as store:

        await store.setup()

        yield store