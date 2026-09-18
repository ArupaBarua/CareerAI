import asyncio
import json
import re
from collections.abc import AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.auth.dependencies import get_current_user
from src.components.conversations.service import get_conversation
from src.components.database.connection import get_db
from src.components.database.models import User
from src.components.memory.stm_service import get_thread_config
from src.components.messages.service import create_message
from src.components.resume.service import get_resume
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
    )

    resume_id: int | None = None


# SSE helpers

def create_sse_event(
    event: str,
    data: dict,
) -> str:

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def split_response_into_chunks(
    text: str,
) -> list[str]:

    return re.findall(
        r"\S+\s*",
        text,
    )

# Status messages

NODE_STATUS_MESSAGES = {

    # Main graph
    "long_term_memory": (
        "Preparing your context..."
    ),

    "summarize_conversation": (
        "Preparing conversation context..."
    ),

    "routing": (
        "Understanding your request..."
    ),

    # Candidate career workflow
    "candidate_career": (
        "Analyzing your career context..."
    ),

    "retrieve_candidate_context": (
        "Reviewing your resume and profile..."
    ),

    # Job-search workflow
    "job_search": (
        "Searching for relevant opportunities..."
    ),

    "search_jobs": (
        "Searching current job openings..."
    ),

    # General workflow
    "general": (
        "Preparing your answer..."
    ),

    # Evaluation
    "evaluate_support": (
        "Checking the response..."
    ),

    "revise_response": (
        "Refining the response..."
    ),

    "evaluate_usefulness": (
        "Reviewing the response..."
    ),

    "rewrite_candidate_query": (
        "Refining the analysis..."
    ),

    "rewrite_search_query": (
        "Refining the job search..."
    ),
}


def get_tool_status(
    tool_name: str,
) -> str:

    normalized_name = (
        tool_name.casefold()
    )

    if ("duckduckgo"
        in normalized_name
        or "duck" in normalized_name
    ):
        return (
            "Searching current job openings..."
        )

    if ("exa" in normalized_name
        or "web_search" in normalized_name
    ):
        return (
            "Searching the web..."
        )

    return "Using an external tool..."


# Streaming chat endpoint

@router.post(
    "/conversations/{conversation_id}/stream"
)
async def stream_chat(
    conversation_id: int,
    payload: ChatRequest,
    app_request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):


    user_message = payload.message.strip()

    if not user_message:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )

    conversation = await get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    if conversation is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    if payload.resume_id is not None:

        resume = await get_resume(
            db=db,
            resume_id=payload.resume_id,
            user_id=current_user.id,
        )

        if resume is None:

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found",
            )

    await create_message(
        db=db,
        conversation=conversation,
        role="user",
        content=user_message,
    )


    graph = app_request.app.state.career_ai_graph

    graph_config = get_thread_config(conversation_id)

    graph_config["run_name"] = "career_ai_graph"
    
    # Streaming generator
    
    async def event_generator() -> AsyncGenerator[str,None]:

        last_status: str | None = None

        try:

            initial_status = (
                "Understanding your request..."
            )

            last_status = initial_status

            yield create_sse_event(
                event="status",
                data={"message": initial_status},
            )

            graph_input = {

                "messages": [HumanMessage(content=user_message)],

                "user_id": current_user.id,

                "conversation_id": conversation_id,

                "final_response": "",
            }

            if payload.resume_id is not None:
                graph_input["resume_id"] = payload.resume_id

            async for event in graph.astream_events(
                graph_input,
                config=graph_config,
                version="v2",
            ):

                event_type = event.get("event","")

                event_name = str(event.get("name",""))

                metadata = event.get("metadata", {})

                if (event_type == "on_chain_start"):

                    node_name = metadata.get("langgraph_node")

                    if node_name is None:

                        node_name = event_name

                    status_message = NODE_STATUS_MESSAGES.get(node_name)

                    if (status_message
                        and status_message != last_status
                    ):

                        last_status = status_message

                        yield create_sse_event(
                            event="status",
                            data={"message": status_message},
                        )

                # External tool activity
                
                elif (event_type == "on_tool_start"):

                    status_message = get_tool_status(event_name)
                    

                    if (status_message != last_status):

                        last_status = status_message

                        yield create_sse_event(
                            event="status",
                            data={"message": status_message},
                        )

            snapshot = await graph.aget_state(graph_config)

            final_response = (
                snapshot.values.get(
                    "final_response",
                    "",
                )
            )

            if not final_response:

                raise RuntimeError(
                    "CareerAI graph completed "
                    "without a final response."
                )

            job_results = (
                snapshot.values.get(
                    "job_results",
                    [],
                )
            )

            assistant_message = (
                await create_message(
                    db=db,
                    conversation=conversation,
                    role="assistant",
                    content=final_response,

                    tool_data=(
                        job_results
                        if job_results
                        else None
                    ),
                )
            )

            yield create_sse_event(
                event="response_start",
                data={},
            )

            for chunk in split_response_into_chunks(final_response):

                yield create_sse_event(
                    event="token",
                    data={"content": chunk},
                )

                await asyncio.sleep(0)

            if job_results:

                yield create_sse_event(
                    event="job_results",
                    data={"results": job_results},
                )

            yield create_sse_event(
                event="done",
                data={"conversation_id": conversation_id,
                    "message_id": assistant_message.id
                },
            )

        except asyncio.CancelledError:

            logger.info(
                "Chat stream cancelled for conversation %s",
                conversation_id,
            )

            raise

        except Exception:

            logger.exception(
                "CareerAI streaming request "
                "failed for conversation %s",
                conversation_id,
            )

            yield create_sse_event(
                event="error",
                data={
                    "message": (
                        "CareerAI could not complete "
                        "this request."
                    )
                },
            )

    # Streaming response

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )