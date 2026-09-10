from typing import Annotated, NotRequired, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class CareerAIState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: int
    conversation_id: int
    resume_id: Optional[int]
    intent: NotRequired[str]
    retrieved_context: NotRequired[list[dict]]
    job_results: NotRequired[list[dict]]
    final_response: NotRequired[str]
