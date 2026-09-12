from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.auth.dependencies import get_current_user
from src.components.conversations.service import get_conversation
from src.components.database.connection import get_db
from src.components.database.models import User
from src.components.messages.schemas import MessageResponse
from src.components.messages.service import get_conversation_messages

router = APIRouter(
    prefix="/conversations",
    tags=["Messages"]
)

@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    conversation = await get_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return await get_conversation_messages(
        db=db,
        conversation_id=conversation_id
    )