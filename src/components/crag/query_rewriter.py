from langchain_core.prompts import ChatPromptTemplate

from src.components.llm.model import llm


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

Rewrite the user's query into a clearer retrieval-oriented query that
preserves the user's original intent but may retrieve more relevant
career or resume information.

Do not answer the query.
Return only the rewritten query."""
        ),
        (
            "human",
            "{query}"
        )
    ]
)


query_rewrite_chain = query_rewrite_prompt | llm