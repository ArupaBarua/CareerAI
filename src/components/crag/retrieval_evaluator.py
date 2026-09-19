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
You are the resume retrieval evaluator for CareerAI.

CareerAI is an AI-powered career assistant that helps users with
resume analysis, skill-gap analysis, job recommendations, job
searching, and general career guidance.

Your task is to evaluate the quality of the retrieved resume context
relative to the supplied retrieval query.

You are NOT answering the user's question.
You are only evaluating whether the retrieved resume context provides
useful candidate evidence for downstream reasoning.

Classify the retrieved context into exactly one of these categories:

correct:
The retrieved context is directly relevant to the query and contains
enough useful resume evidence for downstream reasoning.

ambiguous:
The retrieved context is relevant to the query, but the available
resume evidence is partial, incomplete, unclear, or only moderately
useful.

incorrect:
The retrieved context is irrelevant to the query, contains no
meaningful evidence related to the query, or would not help downstream
reasoning.

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
Retrieval query:
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