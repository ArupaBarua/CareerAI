import operator
import re
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.components.llm.model import judge_llm
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StripEvaluation(BaseModel):
    keep: bool = Field(
        description="Whether this text strip contains useful information for answering the user's query."
    )

strip_evaluator_llm = judge_llm.with_structured_output(StripEvaluation)

strip_evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the resume knowledge-refinement evaluator for CareerAI.

CareerAI is an AI-powered career assistant that helps users with:

- resume and CV analysis
- skill-gap analysis
- job recommendations based on a user's background
- job searching
- career-related questions and guidance

For resume-related workflows, CareerAI retrieves relevant chunks
from the candidate's stored resume using semantic vector search.

The retrieved resume context has been decomposed into smaller
text strips.

Your task is to determine whether the supplied text strip contains
useful candidate evidence relative to the retrieval query.

Return keep=true when the strip contains relevant candidate
information that could help downstream reasoning.

Return keep=false when the strip is irrelevant, unrelated, or
provides no useful candidate evidence for the retrieval query.

Important rules:

- Evaluate only the supplied text strip.
- Judge relevance relative to the retrieval query.
- Do not answer the query.
- Do not rewrite or summarize the text.
- Do not invent candidate information.
- Do not use outside knowledge.
- Return only the structured decision."""
        ),
        (
            "human",
            """
Retrieval query:
{query}

Resume text strip:
{strip}"""
        )
    ]
)

strip_evaluator_chain = strip_evaluator_prompt | strip_evaluator_llm


class KnowledgeRefinementState(TypedDict):
    query: str
    documents: list[Document]
    strips: list[str]
    kept_strips: Annotated[list[tuple[int, str]], operator.add]
    refined_context: str

class StripEvaluationState(TypedDict):
    query: str
    index: int
    strip: str


async def decompose_node(state: KnowledgeRefinementState) -> dict:

    context = "\n\n".join(
        document.page_content
        for document in state["documents"]
    )
    context = re.sub(
        r"\s+",
        " ",
        context,
    ).strip()

    strips = re.split(
        r"(?<=[.!?])\s+",
        context,
    )

    strips = [strip.strip() for strip in strips if strip.strip()]

    logger.info(
        "Knowledge refinement decomposed context into %d strips.",
        len(strips),
    )

    return {
        "strips": strips
    }


def distribute_strips(state: KnowledgeRefinementState) -> str | list[Send]:

    if not state["strips"]:
        return "recompose"

    return [
        Send(
            "evaluate_strip",
            {
                "query": state["query"],
                "index": index,
                "strip": strip
            }
        )
        for index, strip in enumerate(state["strips"])
    ]


async def evaluate_strip_node(state: StripEvaluationState) -> dict:

    evaluation = await strip_evaluator_chain.ainvoke(
        {
            "query": state["query"],
            "strip": state["strip"]
        }
    )

    if evaluation.keep:
        return {
            "kept_strips": [
                (
                    state["index"],
                    state["strip"]
                )
            ]
        }

    return {
        "kept_strips": []
    }


async def recompose_node(
    state: KnowledgeRefinementState
) -> dict:

    kept_strips = sorted(
        state.get("kept_strips", []),
        key=lambda item: item[0],
    )

    refined_context = "\n".join(strip for _, strip in kept_strips).strip()

    logger.info(
        "Knowledge refinement retained %d/%d strips.",
        len(kept_strips),
        len(state["strips"]),
    )

    return {
        "refined_context": refined_context
    }

# Build Knowledge Refinement Subgraph

builder = StateGraph(
    KnowledgeRefinementState
)

builder.add_node(
    "decompose",
    decompose_node
)

builder.add_node(
    "evaluate_strip",
    evaluate_strip_node
)

builder.add_node(
    "recompose",
    recompose_node
)

builder.add_edge(
    START, "decompose"
)

builder.add_conditional_edges(
    "decompose", distribute_strips
)

builder.add_edge(
    "evaluate_strip", "recompose"
)

builder.add_edge(
    "recompose", END
)

knowledge_refinement_graph = builder.compile()