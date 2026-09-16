from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.components.llm.model import judge_llm, llm

class SupportEvaluation(BaseModel):

    status: Literal[
        "supported", 
        "unsupported"
    ] = Field(description="Whether the response is fully supported by the supplied context.")

class UsefulnessEvaluation(BaseModel):

    status: Literal[
        "useful",
        "not_useful"
    ] = Field(description="Whether the response adequately answers the user's query.")


support_llm = judge_llm.with_structured_output(SupportEvaluation)

usefulness_llm = judge_llm.with_structured_output(UsefulnessEvaluation)


support_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a grounding evaluator for CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

Determine whether every factual claim in the response is supported
by the supplied context.

The context may contain:
- resume text context
- knowledge-graph context
- web context

Classify the response as:

supported:
The factual claims in the response are supported by the provided context.

unsupported:
The response contains claims that are unsupported, fabricated,
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

Resume text context:
{text_context}

Knowledge graph context:
{graph_context}

Web context:
{web_context}

Response:
{response}"""
        )
    ]
)


usefulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a response-quality evaluator for CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

Determine whether the response directly and sufficiently addresses
the user's query.

Classify the response as:

useful:
The response directly addresses the user's request and provides an
adequate answer based on the available information.

not_useful:
The response fails to address the request, is substantially incomplete,
or does not provide the information the user was asking for.

Evaluate usefulness only.
Do not evaluate factual grounding here.
Do not answer the user's question.
Return only the structured classification."""
        ),
        (
            "human",
            """
User query:
{query}

Response:
{response}"""
        )
    ]
)


revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the response-revision component of CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

Your task is to rewrite the current answer so that it is fully grounded
in the context provided to you.

You MUST use only the supplied context as evidence.

The supplied context may contain:
- resume text context
- knowledge-graph context
- web context

Strict rules:

- Do not infer facts that are not explicitly supported by the context.
- Do not invent skills, experience, qualifications, education,
  achievements, companies, job requirements, dates, technologies,
  statistics, or any other factual information.
- Do not strengthen uncertain information into a definite claim.
- Every factual claim in the revised answer must be traceable to the
  provided context.
- Preserve useful parts of the current answer only when they are
  supported by the context.

The current answer is not a source of truth. Treat it only as a draft
that may contain unsupported or fabricated information.

Return only the revised answer."""
        ),
        (
            "human",
            """
User query:
{query}

Resume text context:
{text_context}

Knowledge graph context:
{graph_context}

Web context:
{web_context}

Current response:
{response}"""
        )
    ]
)


query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the query-rewriting component of CareerAI.

CareerAI is an AI-powered career assistant for resume analysis,
skill-gap analysis, job recommendations, job search, and career guidance.

The previous retrieval and generation process did not produce a useful
answer.

Rewrite the user's query into a concise retrieval-focused query
that is more likely to retrieve relevant candidate information.

Preserve the user's original intent.
Do not answer the question.
Do not invent information.
Return only the rewritten query."""
        ),
        (
            "human",
            """
Original query:
{query}

Current response:
{response}"""
        )
    ]
)


support_evaluator_chain = support_prompt | support_llm

usefulness_evaluator_chain = usefulness_prompt | usefulness_llm

query_rewrite_chain = query_rewrite_prompt | llm

revision_chain = revision_prompt | llm