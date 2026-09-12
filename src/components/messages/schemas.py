from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict

class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str | None
    tool_name: str | None
    tool_call_id: str | None
    tool_data: dict[str, Any] | list[Any] | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)