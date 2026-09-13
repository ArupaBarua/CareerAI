from typing import Literal, NotRequired

from langchain_core.documents import Document
from typing_extensions import TypedDict

class CRAGState(TypedDict):
    query: str
    user_id: int
    resume_id: int

    retrieved_documents: NotRequired[list[Document]]
    retrieval_status: NotRequired[
        Literal[
            "correct",
            "ambiguous",
            "incorrect",
        ] | None
    ]
    refined_context: NotRequired[str]
    web_context: NotRequired[str]
    final_response: NotRequired[str]

    support_status: NotRequired[Literal["supported", "unsupported"]] | None
    usefulness_status: NotRequired[Literal["useful", "not_useful"]] | None

    revision_count: NotRequired[int]
    rewrite_count: NotRequired[int]