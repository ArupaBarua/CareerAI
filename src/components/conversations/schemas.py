from datetime import datetime
from pydantic import BaseModel, ConfigDict

class ConversationCreate(BaseModel):
    title: str | None = None

class ConversationResponse(BaseModel):
    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)