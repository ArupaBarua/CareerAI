from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.constants.settings import settings

@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:

    postgres_url = settings.DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://"
    )

    async with AsyncPostgresSaver.from_conn_string(postgres_url) as checkpointer:

        await checkpointer.setup()

        yield checkpointer