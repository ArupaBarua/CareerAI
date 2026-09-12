from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.database.models import Conversation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def create_conversation(
    db: AsyncSession,
    user_id: int,
    title: str | None = None
) -> Conversation:

    conversation = Conversation(
        user_id=user_id,
        title=title
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    logger.info(f"Created a new conversation with id {conversation.id}")

    return conversation

async def get_user_conversations(
    db: AsyncSession,
    user_id: int
) -> list[Conversation]:

    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
    )

    logger.info(f"Retrieved all conversations with user id {user_id}")

    return list(result.scalars().all())

async def get_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Conversation | None:
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    logger.info(f"Retrieved conversation id {conversation_id} with user id {user_id}")

    return result.scalar_one_or_none()


async def delete_conversation(
    db: AsyncSession,
    conversation: Conversation,
) -> None:
    
    await db.delete(conversation)
    await db.commit()

    logger.info(f"Deleted conversation id {conversation.id}")