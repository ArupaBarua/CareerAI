from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llm.model import llm


class MemoryOperation(BaseModel):

    action: Literal[
        "add",
        "update",
        "delete",
    ]

    memory_id: str | None = Field(
        default=None,
        description=(
            "The ID of an existing memory. "
            "Required for update and delete actions. "
            "Must be null for add actions."
        ),
    )

    memory_type: str | None = Field(
        default=None,
        description=(
            "The category of the memory, such as target_role, "
            "career_goal, work_preference, location_preference, "
            "or professional_interest."
        ),
    )

    content: str | None = Field(
        default=None,
        description=(
            "The concise long-term fact to remember. "
            "Required for add and update."
        ),
    )


class MemoryExtractionResult(BaseModel):

    operations: list[MemoryOperation] = Field(
        default_factory=list,
        description=(
            "Memory changes that should be applied. "
            "Return an empty list if no memory changes are needed."
        ),
    )


memory_extractor_llm = llm.with_structured_output(MemoryExtractionResult)


memory_extractor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the long-term memory manager for CareerAI.

CareerAI is an AI-powered career assistant.

Your task is to examine:

1. the user's existing long-term memories,
2. the user's latest message,

and determine whether the long-term memory store should change.

Store only information that is likely to remain useful across future
career-related conversations.

Good long-term memories include:

- target job roles,
- career goals,
- preferred work arrangements,
- preferred job locations,
- preferred industries,
- persistent professional interests,
- stable career preferences.

Do NOT store:

- greetings or casual conversation,
- temporary questions,
- one-time search requests,
- requests to explain code,
- transient job-search results,
- assistant-generated conclusions,
- facts that are already accurately represented in existing memory,
- detailed resume information that belongs in the resume knowledge base.

Actions:

add:
Use when the latest message contains a useful long-term fact that is not
already represented.

update:
Use when the latest message changes or replaces an existing memory.
For update, use the exact existing memory ID.

delete:
Use when the user explicitly contradicts, withdraws, or makes an
existing memory obsolete.
For delete, use the exact existing memory ID.

If nothing needs to change, return an empty operations list.

Rules:

- Do not invent information.
- Base memory changes only on the latest user message.
- Do not create duplicate memories.
- Prefer updating an existing memory instead of adding a conflicting duplicate.
- Keep memory content concise and self-contained.
- One user message may produce multiple memory operations.
""",
        ),
        (
            "human",
            """
Existing long-term memories:
{memories}

Latest user message:
{message}"""
        )
    ]
)


memory_extractor_chain = memory_extractor_prompt | memory_extractor_llm