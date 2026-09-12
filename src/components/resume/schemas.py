from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ResumeResponse(BaseModel):
    id: int
    filename: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )