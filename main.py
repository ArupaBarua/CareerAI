import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.components.database.connection import engine
from src.components.graph.main_graph import build_career_ai_graph
from src.components.graphrag.connection import (
    close_neo4j_connection,
    verify_neo4j_connection,
)
from src.components.memory.checkpointer import get_checkpointer
from src.components.memory.store import get_memory_store
from src.constants.settings import settings
from src.routes.auth import router as auth_router
from src.routes.chat import router as chat_router
from src.routes.conversations import router as conversations_router
from src.routes.health import router as health_router
from src.routes.messages import router as messages_router
from src.routes.resume import router as resume_router
from src.utils.logger import setup_logger


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI,):

    logger.info("CareerAI application started")

    await verify_neo4j_connection()

    try:

        async with get_checkpointer() as checkpointer:

            async with get_memory_store() as store:

                app.state.career_ai_graph = (
                    build_career_ai_graph(
                        checkpointer=checkpointer,
                        store=store,
                    )
                )

                logger.info("CareerAI graph initialized")

                yield

    finally:

        await close_neo4j_connection()

        await engine.dispose()

        logger.info("CareerAI application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade Agentic AI career assistant"
    ),
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_ORIGIN
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)

app.include_router(auth_router)

app.include_router(conversations_router)

app.include_router(messages_router)

app.include_router(resume_router)

app.include_router(chat_router)