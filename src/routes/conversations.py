from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.components.auth.dependencies import get_current_user
from src.components.conversations.schemas import (
    ConversationCreate,
    ConversationResponse,
)
from src.components.conversations.service import (
    create_conversation,
    delete_conversation,
    get_conversation,
    get_user_conversations,
)
from src.components.database.connection import get_db
from src.components.database.models import User

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_new_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    return await create_conversation(
        db=db,
        user_id=current_user.id,
        title=request.title
    )


@router.get(
    "",
    response_model=list[ConversationResponse]
)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    return await get_user_conversations(
        db=db,
        user_id=current_user.id
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse
)
async def retrieve_conversation(
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

    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
async def remove_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> None:
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

    await delete_conversation(
        db=db,
        conversation=conversation
    )