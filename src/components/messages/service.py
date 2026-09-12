from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.database.models import (
    Conversation,
    Message,
)

async def create_message(
    db: AsyncSession,
    conversation: Conversation,
    role: str,
    content: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_data: dict[str, Any] | list[Any] | None = None,
) -> Message:

    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_data=tool_data,
    )

    conversation.updated_at = datetime.now(timezone.utc)

    db.add(message)

    await db.commit()
    await db.refresh(message)

    return message


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: int,
) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id
        )
        .order_by(
            Message.created_at.desc(),
            Message.id.desc(),
        )
    )

    return list(result.scalars().all())