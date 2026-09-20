from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.constants.settings import settings


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[
    AsyncPostgresSaver
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

        checkpointer = AsyncPostgresSaver(
            pool
        )

        await checkpointer.setup()

        yield checkpointer