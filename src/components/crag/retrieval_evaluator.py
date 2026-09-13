from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llm.model import judge_llm
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

class RetrievalEvaluation(BaseModel):

    status: Literal["correct", 
                    "ambigous", 
                    "incorrect"] = Field(description="Evaluation of how relevant and sufficient the retrieved context is for the user's query.")

retrieval_evaluator_llm = judge_llm.with_structured_output(RetrievalEvaluation)

retrieval_evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the retrieval evaluator for CareerAI.

CareerAI is an AI-powered career assistant that helps users with:

- resume and CV analysis
- skill-gap analysis
- job recommendations based on a user's background
- job searching
- career-related questions and guidance

For resume-related workflows, CareerAI retrieves relevant chunks from
the user's stored resume using semantic vector search.

Your task is to evaluate the quality of those retrieved resume chunks
relative to the user's current query.

You are NOT answering the user's query.
You are only evaluating whether the retrieved context is relevant and
sufficient for a later CareerAI agent to answer the query accurately.

Classify the retrieved context into exactly one of these categories:

correct:
The retrieved context is directly relevant to the user's query and
contains enough useful information to answer it reliably.

ambiguous:
The retrieved context is relevant to the query, but the information is
partial, incomplete, unclear, or insufficient to answer the query
confidently without additional information.

incorrect:
The retrieved context is irrelevant to the query, does not contain
meaningful evidence related to the query, or would not help answer it.

Important rules:

- Judge only the provided retrieved context.
- Do not assume information that is not present in the context.
- Do not use outside knowledge to fill missing resume information.
- Do not answer the user's question.
- Do not rewrite or summarize the resume.
- Return only the structured classification.""",
        ),      
        (
            "human",
            """
User query:
{query}

Retrieved resume context:
{context}"""
        )
    ]
)

retrieval_evaluator_chain = retrieval_evaluator_prompt | retrieval_evaluator_llm


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        document.page_content
        for document in documents
    )


async def evaluate_retrieval(query: str, 
                             documents: list[Document]) -> Literal["correct", "ambiguous", "incorrect"]:
    if not documents:
        return "incorrect"

    context = format_documents(documents=documents)

    evaluation = await retrieval_evaluator_chain.ainvoke(
        {
            "query": query,
            "context": context
        }
    )

    logger.info(
        "Retrieval evaluation: %s",
        evaluation.status,
    )

    return evaluation.status