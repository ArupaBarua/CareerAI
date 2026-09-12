from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.components.database.connection import engine
from src.constants.settings import settings
from src.routes.health import router as health_router
from src.routes.auth import router as auth_router
from src.routes.conversations import router as conversations_router
from src.routes.messages import router as messages_router
from src.utils.logger import setup_logger

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
    
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CareerAI application started")

    yield

    await engine.dispose()

    logger.info("CareerAI application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Agentic AI career assistant",
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