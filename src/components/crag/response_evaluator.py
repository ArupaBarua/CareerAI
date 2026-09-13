from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llm.model import judge_llm

class SupportEvaluation(BaseModel):

    status: Literal[
        "supported", 
        "unsupported"
    ] = Field(description="Whether the generated answer is grounded in the provided context.")

class UsefulnessEvaluation(BaseModel):

    status: Literal[
        "useful",
        "not_useful"
    ] = Field(description="Whether the generated answer adequately addresses the user's query.")


support_llm = judge_llm.with_structured_output(SupportEvaluation)

usefulness_llm = judge_llm.with_structured_output(UsefulnessEvaluation)


support_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the grounding evaluator for CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

Evaluate whether the generated answer is supported by the context
provided to the generation model.

The context may contain refined resume information, web-search
information, or both.

Classify the answer as:

supported:
The factual claims in the answer are supported by the provided context.

unsupported:
The answer contains claims that are unsupported, fabricated,
contradictory, or cannot be justified from the provided context.

Rules:

- Judge grounding only.
- Do not judge writing quality.
- Do not answer the user's question.
- Do not use outside knowledge to justify unsupported claims.
- Return only the structured classification."""
        ),
        (
            "human",
            """
User query:
{query}

Refined resume context:
{refined_context}

Web context:
{web_context}

Generated answer:
{answer}"""
        )
    ]
)


support_chain = support_prompt | support_llm


usefulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the response-quality evaluator for CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

Evaluate whether the generated answer actually addresses the user's
career-related query in a clear and useful way.

Classify the answer as:

useful:
The answer directly addresses the user's request and provides an
adequate response based on the available information.

not_useful:
The answer fails to address the request, is substantially incomplete,
or does not provide the information the user was asking for.

Do not evaluate factual grounding here. Grounding is checked separately.

Do not answer the user's question.
Return only the structured classification."""
        ),
        (
            "human",
            """
User query:
{query}

Generated answer:
{answer}"""
        )
    ]
)


usefulness_chain = usefulness_prompt | usefulness_llm